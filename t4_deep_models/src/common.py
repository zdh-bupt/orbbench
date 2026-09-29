from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_COLUMNS = ("X_m", "Y_m", "Z_m", "VX_m_s", "VY_m_s", "VZ_m_s")
CSV_COLUMNS = ("Epoch_UTC", *TARGET_COLUMNS)
GROUP_IDS = ("leo", "meo", "heo")
MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "dlinear": "DLinear",
    "tcn": "TCN",
    "informer": "Informer",
    "fedformer": "FEDformer",
    "chronos": "Chronos",
    "chronos2": "Chronos2",
}
RESULT_FIELDS = (
    "schema_version",
    "benchmark_id",
    "protocol_sha256",
    "model",
    "model_label",
    "group",
    "group_label",
    "horizon_hours",
    "horizon_steps",
    "sample_seconds",
    "context_steps",
    "complete",
    "quality_gate_passed",
    "n_files",
    "n_origins",
    "n_seeds",
    "position_error_mean_m",
    "velocity_error_mean_m_s",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, group: str, norad_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group}:{norad_id}".encode("ascii")).hexdigest()


def parse_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def horizon_steps(hours: list[float], sample_seconds: int) -> list[int]:
    result: list[int] = []
    for hour in hours:
        raw = float(hour) * 3600.0 / sample_seconds
        step = int(round(raw))
        if abs(raw - step) > 1.0e-9:
            raise ValueError(f"horizon {hour} h is not aligned to {sample_seconds} s")
        result.append(step)
    return result


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    required = {
        "benchmark_id", "sample_seconds", "input_steps", "target_steps",
        "horizons_hours", "cohort", "splits", "train_stride_steps",
        "validation_origin_stride_steps", "test_origin_stride_steps",
        "representation", "training_size_study", "training", "quality_gates",
        "evaluation", "groups", "models", "metrics",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"protocol is missing fields: {missing}")
    if int(protocol.get("schema_version", -1)) != 3:
        raise ValueError("protocol schema_version must be 3")
    for key in (
        "sample_seconds", "input_steps", "train_stride_steps",
        "validation_origin_stride_steps", "test_origin_stride_steps",
    ):
        if int(protocol[key]) <= 0:
            raise ValueError(f"protocol {key} must be positive")
    expected_steps = horizon_steps(
        [float(value) for value in protocol["horizons_hours"]],
        int(protocol["sample_seconds"]),
    )
    target_steps = [int(value) for value in protocol["target_steps"]]
    if target_steps != expected_steps or target_steps != sorted(set(target_steps)):
        raise ValueError("target_steps must exactly match the unique increasing horizons")
    if protocol["representation"].get("id") != "origin_rtn_relative_v1":
        raise ValueError("unsupported representation")
    if protocol["cohort"].get("same_objects_across_splits") is not True:
        raise ValueError("the reproduction requires one cohort followed through time")
    if int(protocol["cohort"].get("prepared_objects_per_group", 0)) <= 0:
        raise ValueError("prepared_objects_per_group must be positive")
    if tuple(item.get("id") for item in protocol["groups"]) != GROUP_IDS:
        raise ValueError("groups must be the locked leo/meo/heo sequence")
    model_ids = [item.get("id") for item in protocol["models"]]
    if len(model_ids) != len(set(model_ids)) or set(model_ids) != set(MODEL_LABELS):
        raise ValueError("models must contain each of the eight locked methods exactly once")
    modes = {item["id"]: item.get("mode") for item in protocol["models"]}
    for model in ("lstm", "gru", "dlinear", "tcn", "informer", "fedformer"):
        if modes[model] != "retrain":
            raise ValueError(f"{model} must be retrained")
    for model in ("chronos", "chronos2"):
        if modes[model] != "pretrained_test_only":
            raise ValueError(f"{model} must be pretrained test only")
    seeds = [int(value) for value in protocol["training"].get("seeds", [])]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("the paper-aligned protocol requires three unique seeds")
    split_order = ("train", "val", "test")
    if tuple(protocol["splits"]) != split_order:
        raise ValueError("splits must be ordered train, val, test")
    previous_end: datetime | None = None
    for split in split_order:
        spec = protocol["splits"][split]
        start = parse_utc_datetime(spec["start_utc"])
        end = parse_utc_datetime(spec["end_utc"])
        if start >= end or (previous_end is not None and start != previous_end):
            raise ValueError("split intervals must be positive and contiguous")
        previous_end = end
    return protocol
