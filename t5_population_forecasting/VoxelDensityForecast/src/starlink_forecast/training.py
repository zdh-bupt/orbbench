from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .closed_cohort import ClosedCohortDataset
from .config import config_digest, resolve_project_path
from .model import PhysicsResidualUNetConvGRU3D
from .splits import TimeRange, chronological_fraction_ranges


@dataclass(frozen=True)
class ShardSelection:
    """A shard containing one or more origins in a requested time range."""

    path: Path
    origin_count: int


def shard_manifest(config: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    root = resolve_project_path(config, config["paths"]["shard_dir"])
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("closed-cohort manifest is missing; run prepare_closed_cohort.py first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    shards = [
        Path(entry["path"])
        for entry in manifest.get("shards", [])
        if entry.get("path") and entry.get("status") in {"written", "reused"}
    ]
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest lists missing shards: {missing[:3]}")
    if not shards:
        raise ValueError("No usable closed-cohort shards are listed")
    return manifest, shards


def all_origin_times(shards: list[Path]) -> np.ndarray:
    values: list[np.ndarray] = []
    for path in shards:
        with np.load(path, allow_pickle=False) as archive:
            values.append(archive["time_ns"][archive["origin_indices"]])
    return np.unique(np.concatenate(values))


def split_ranges(config: dict[str, Any], shards: list[Path]) -> dict[str, TimeRange]:
    split = config["splits"]
    if str(split.get("mode")) != "chronological_fractions":
        names = ("train", "validation", "calibration", "test")
        return {
            name: TimeRange.from_strings(split[name]["start"], split[name]["end"])
            for name in names
        }
    fractions = {
        "train": float(split["train_fraction"]),
        "validation": float(split["validation_fraction"]),
        "calibration": float(split["calibration_fraction"]),
        "test": float(split["test_fraction"]),
    }
    return chronological_fraction_ranges(
        all_origin_times(shards),
        fractions,
        purge=pd.Timedelta(hours=int(split["purge_hours"])),
    )


def shard_selections_for_range(
    shards: list[Path],
    time_range: TimeRange,
) -> list[ShardSelection]:
    """Select shards with a metadata-only read.

    Reading only ``time_ns`` and ``origin_indices`` avoids expanding state and
    physical-prior arrays while computing train/validation/calibration splits.
    """

    selections: list[ShardSelection] = []
    for path in shards:
        with np.load(path, allow_pickle=False) as archive:
            times_ns = archive["time_ns"]
            origins = archive["origin_indices"]
        mask = (times_ns[origins] >= int(time_range.start.value)) & (
            times_ns[origins] <= int(time_range.end.value)
        )
        origin_count = int(mask.sum())
        if origin_count:
            selections.append(ShardSelection(path=path, origin_count=origin_count))
    return selections


def dataset_for_selection(
    selection: ShardSelection,
    config: dict[str, Any],
    time_range: TimeRange,
    *,
    return_tensors: bool,
) -> ClosedCohortDataset:
    """Load one selected shard and verify its metadata count did not drift."""

    dataset = ClosedCohortDataset(
        selection.path,
        config,
        start=time_range.start,
        end=time_range.end,
        return_tensors=return_tensors,
    )
    if len(dataset) != selection.origin_count:
        raise RuntimeError(
            f"Shard metadata changed while loading {selection.path}: "
            f"expected {selection.origin_count} origins, got {len(dataset)}"
        )
    return dataset


def serialize_ranges(ranges: dict[str, TimeRange]) -> dict[str, dict[str, str]]:
    return {
        name: {"start": value.start.isoformat(), "end": value.end.isoformat()}
        for name, value in ranges.items()
    }


_MODEL_RUNTIME_OVERRIDE_KEYS = frozenset({"base_channels", "hidden_channels"})
_TRAINING_RUNTIME_OVERRIDE_KEYS = frozenset(
    {
        "learning_rate",
        "weight_decay",
        "multiscale_loss_weight",
        "residual_l2_weight",
        "early_stopping_patience",
    }
)


def config_with_runtime_overrides(
    shard_config: dict[str, Any],
    *,
    model_overrides: dict[str, int | float] | None = None,
    training_overrides: dict[str, int | float] | None = None,
) -> dict[str, Any]:
    """Build a run-only config without changing the config used to load shards."""

    model_overrides = model_overrides or {}
    training_overrides = training_overrides or {}
    unknown_model = set(model_overrides) - _MODEL_RUNTIME_OVERRIDE_KEYS
    unknown_training = set(training_overrides) - _TRAINING_RUNTIME_OVERRIDE_KEYS
    if unknown_model or unknown_training:
        details: list[str] = []
        if unknown_model:
            details.append(f"model: {', '.join(sorted(unknown_model))}")
        if unknown_training:
            details.append(f"training: {', '.join(sorted(unknown_training))}")
        raise ValueError("Unsupported runtime overrides: " + "; ".join(details))

    run_config = deepcopy(shard_config)
    run_config["model"].update(model_overrides)
    run_config["training"].update(training_overrides)
    return run_config


def checkpoint_run_config(
    checkpoint: dict[str, Any], shard_config: dict[str, Any]
) -> dict[str, Any]:
    """Validate a checkpoint against shard semantics and return its model config.

    Legacy checkpoints stored one config digest. New experiment checkpoints keep
    a separate immutable shard digest and a run-config digest so width and loss
    ablations can still load the existing shards safely.
    """

    shard_digest = config_digest(shard_config)
    checkpoint_shard_digest = checkpoint.get("shard_config_digest", checkpoint.get("config_digest"))
    if checkpoint_shard_digest != shard_digest:
        raise ValueError("Checkpoint and shard configuration digests do not match")

    run_config = checkpoint.get("run_config", checkpoint.get("config", shard_config))
    if not isinstance(run_config, dict):
        raise ValueError("Checkpoint has an invalid run configuration")
    recorded_run_digest = checkpoint.get("run_config_digest", checkpoint.get("config_digest"))
    if recorded_run_digest != config_digest(run_config):
        raise ValueError("Checkpoint run configuration digest does not match its stored configuration")
    return run_config


def model_from_config(config: dict[str, Any]) -> PhysicsResidualUNetConvGRU3D:
    model = config["model"]
    return PhysicsResidualUNetConvGRU3D(
        input_channels=int(model["input_channels"]),
        lead_count=len(config["forecast"]["horizons"]),
        base_channels=int(model["base_channels"]),
        hidden_channels=int(model["hidden_channels"]),
        residual_scale=float(model["residual_logit_limit"]),
        epsilon=float(model["epsilon"]),
    )
