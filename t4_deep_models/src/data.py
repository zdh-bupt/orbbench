from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import GROUP_IDS, read_json
from .representation import encode_window


GROUPS = GROUP_IDS


def ready_entries(manifest_path: Path, split: str, group: str | None = None) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    entries = [entry for entry in manifest["entries"] if entry["split"] == split]
    if group is not None:
        entries = [entry for entry in entries if entry["group"] == group]
    not_ready = [entry for entry in entries if entry.get("status") != "ready"]
    if not_ready:
        raise RuntimeError(f"{len(not_ready)} {split}/{group or '*'} arrays are not ready")
    if not entries:
        raise RuntimeError(f"manifest contains no {split}/{group or '*'} entries")
    return sorted(entries, key=lambda entry: (entry["group"], int(entry["rank_in_group"])))


def select_entries(entries: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected = [entry for entry in entries if int(entry["rank_in_group"]) < count]
    if len(selected) != count:
        raise ValueError(f"found {len(selected)} objects; {count} requested")
    return selected


def grouped_entries(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {group: [] for group in GROUPS}
    for entry in entries:
        result[entry["group"]].append(entry)
    for values in result.values():
        values.sort(key=lambda entry: int(entry["rank_in_group"]))
    return result


def origins_for_rows(rows: int, input_steps: int, maximum_horizon_steps: int, stride: int) -> range:
    if min(rows, input_steps, maximum_horizon_steps, stride) <= 0:
        raise ValueError("row and step counts must be positive")
    last = int(rows) - int(maximum_horizon_steps)
    if last < input_steps:
        return range(0)
    return range(int(input_steps), last + 1, int(stride))


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        entries: list[dict[str, Any]],
        input_steps: int,
        target_steps: list[int],
        stride: int,
    ) -> None:
        if not entries:
            raise ValueError("window dataset requires entries")
        self.entries = entries
        self.input_steps = int(input_steps)
        self.target_steps = np.asarray(target_steps, dtype=np.int64)
        self.stride = int(stride)
        if self.input_steps <= 0 or self.stride <= 0:
            raise ValueError("window dimensions and stride must be positive")
        if (
            self.target_steps.ndim != 1
            or len(self.target_steps) == 0
            or np.any(self.target_steps <= 0)
            or np.any(np.diff(self.target_steps) <= 0)
        ):
            raise ValueError("target steps must be unique, positive, and increasing")
        self.counts: list[int] = []
        self.cumulative: list[int] = []
        running = 0
        for entry in entries:
            count = len(
                origins_for_rows(
                    int(entry["rows"]), self.input_steps, int(self.target_steps[-1]), self.stride
                )
            )
            if count == 0:
                raise ValueError(f"no windows fit {entry['array_path']}")
            self.counts.append(count)
            running += count
            self.cumulative.append(running)

    def __len__(self) -> int:
        return self.cumulative[-1]

    @staticmethod
    @lru_cache(maxsize=512)
    def _open(path: str) -> np.ndarray:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.ndim != 2 or value.shape[1] != 6 or value.dtype != np.float32:
            raise ValueError(f"invalid prepared array {path}: shape={value.shape}, dtype={value.dtype}")
        return value

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index = bisect_right(self.cumulative, index)
        previous = self.cumulative[file_index - 1] if file_index else 0
        local_index = index - previous
        origin = self.input_steps + local_index * self.stride
        array = self._open(self.entries[file_index]["array_path"])
        context = np.asarray(array[origin - self.input_steps : origin], dtype=np.float32)
        targets = np.asarray(array[origin + self.target_steps - 1], dtype=np.float32)
        encoded_context, encoded_targets = encode_window(context, targets)
        return (
            torch.from_numpy(encoded_context),
            torch.from_numpy(encoded_targets),
            torch.from_numpy(context[-1].copy()),
        )


def iter_evaluation_batches(
    entries: list[dict[str, Any]],
    input_steps: int,
    target_steps: list[int],
    stride: int,
    batch_size: int,
    max_origins_per_entry: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if target_steps != sorted(set(target_steps)) or not target_steps:
        raise ValueError("target_steps must be unique and increasing")
    contexts: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for entry in entries:
        array = WindowDataset._open(entry["array_path"])
        if len(array) != int(entry["rows"]):
            raise ValueError(f"manifest row count does not match {entry['array_path']}")
        origins = origins_for_rows(len(array), input_steps, target_steps[-1], stride)
        if max_origins_per_entry is not None:
            if max_origins_per_entry <= 0:
                raise ValueError("max_origins_per_entry must be positive")
            origins = origins[:max_origins_per_entry]
        if not origins:
            raise ValueError(f"no evaluation origins fit {entry['array_path']}")
        for origin in origins:
            contexts.append(np.asarray(array[origin - input_steps : origin], dtype=np.float32).copy())
            truths.append(np.asarray(array[origin + np.asarray(target_steps) - 1], dtype=np.float32).copy())
            if len(contexts) == batch_size:
                yield np.stack(contexts), np.stack(truths)
                contexts.clear()
                truths.clear()
    if contexts:
        yield np.stack(contexts), np.stack(truths)
