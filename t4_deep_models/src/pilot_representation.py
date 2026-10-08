from __future__ import annotations

from bisect import bisect_right
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import WindowDataset
from .representation import encode_window


PILOT_REPRESENTATIONS = (
    "origin_rtn_relative",
    "window_zscore_teme",
    "window_zscore_rtn",
)


def _zscore(
    context: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(context, dtype=np.float64).mean(axis=0)
    scale = np.asarray(context, dtype=np.float64).std(axis=0)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    encoded_context = (np.asarray(context, dtype=np.float64) - mean) / scale
    encoded_targets = (np.asarray(targets, dtype=np.float64) - mean) / scale
    return (
        np.asarray(encoded_context, dtype=np.float32),
        np.asarray(encoded_targets, dtype=np.float32),
        np.asarray(scale, dtype=np.float32),
    )


def encode_pilot_window(
    context: np.ndarray,
    targets: np.ndarray,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context_values = np.asarray(context, dtype=np.float32)
    target_values = np.asarray(targets, dtype=np.float32)
    origin = context_values[-1]
    radius = float(np.linalg.norm(np.asarray(origin[:3], dtype=np.float64)))
    speed = float(np.linalg.norm(np.asarray(origin[3:], dtype=np.float64)))
    rtn_physical_scale = np.asarray([radius] * 3 + [speed] * 3, dtype=np.float32)
    if representation == "origin_rtn_relative":
        encoded_context, encoded_targets = encode_window(context_values, target_values)
        return encoded_context, encoded_targets, rtn_physical_scale
    if representation == "window_zscore_teme":
        return _zscore(context_values, target_values)
    if representation == "window_zscore_rtn":
        encoded_context, encoded_targets = encode_window(context_values, target_values)
        normalized_context, normalized_targets, scale = _zscore(
            encoded_context, encoded_targets
        )
        return normalized_context, normalized_targets, scale * rtn_physical_scale
    raise ValueError(f"unsupported pilot representation: {representation}")


class PilotWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        entries: list[dict[str, Any]],
        input_steps: int,
        target_steps: list[int],
        stride: int,
        representation: str,
    ) -> None:
        if representation not in PILOT_REPRESENTATIONS:
            raise ValueError(f"unsupported pilot representation: {representation}")
        self.base = WindowDataset(entries, input_steps, target_steps, stride)
        self.entries = self.base.entries
        self.input_steps = self.base.input_steps
        self.target_steps = self.base.target_steps
        self.stride = self.base.stride
        self.representation = representation

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index = bisect_right(self.base.cumulative, index)
        previous = self.base.cumulative[file_index - 1] if file_index else 0
        local_index = index - previous
        origin = self.input_steps + local_index * self.stride
        array = WindowDataset._open(self.entries[file_index]["array_path"])
        context = np.asarray(array[origin - self.input_steps : origin], dtype=np.float32)
        targets = np.asarray(array[origin + self.target_steps - 1], dtype=np.float32)
        encoded_context, encoded_targets, physical_scale = encode_pilot_window(
            context, targets, self.representation
        )
        return (
            torch.from_numpy(encoded_context),
            torch.from_numpy(encoded_targets),
            torch.from_numpy(physical_scale),
        )
