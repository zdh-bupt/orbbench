from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    return {key: archive[key] for key in archive.files}


def validate_dataset_index_alignment(
    data: dict[str, np.ndarray],
    index: pd.DataFrame,
    label_scope_mask: np.ndarray | None = None,
) -> None:
    required_arrays = {
        "contexts_raw",
        "next_raw",
        "labels",
        "primary_mask",
        "norad_ids",
        "target_unix_us",
        "sample_ids",
        "split_codes",
    }
    missing_arrays = sorted(required_arrays - set(data))
    if missing_arrays:
        raise ValueError(f"Dataset archive is missing arrays: {missing_arrays}")
    required_columns = {
        "SAMPLE_ID",
        "NORAD_CAT_ID",
        "TARGET_EPOCH_UTC",
        "LABEL_ID",
        "IS_PRIMARY_LABEL",
        "SPLIT",
    }
    missing_columns = sorted(required_columns - set(index.columns))
    if missing_columns:
        raise ValueError(f"Dataset index is missing columns: {missing_columns}")
    count = len(index)
    mismatched_lengths = {
        key: int(value.shape[0]) for key, value in data.items() if int(value.shape[0]) != count
    }
    if mismatched_lengths:
        raise ValueError(f"Dataset/index length mismatch: index={count}, arrays={mismatched_lengths}")
    expected_split = index["SPLIT"].map({"train": 0, "validation": 1, "test": 2})
    if expected_split.isna().any():
        raise ValueError("Dataset index contains an unknown SPLIT value")
    primary = index["IS_PRIMARY_LABEL"]
    if primary.dtype != bool:
        primary = primary.astype(str).str.lower().isin({"1", "true", "yes"})
    expected_time = (
        pd.to_datetime(index["TARGET_EPOCH_UTC"], utc=True).astype("int64").to_numpy(np.int64)
        // 1000
    )
    scope = (
        np.ones(count, dtype=bool)
        if label_scope_mask is None
        else np.asarray(label_scope_mask, dtype=bool)
    )
    if scope.shape != (count,):
        raise ValueError(f"label_scope_mask must have shape {(count,)}, got {scope.shape}")
    checks = {
        "sample_ids": np.array_equal(
            data["sample_ids"].astype(str), index["SAMPLE_ID"].astype(str).to_numpy()
        ),
        "split_codes": np.array_equal(data["split_codes"], expected_split.to_numpy(np.int8)),
        "norad_ids": np.array_equal(
            data["norad_ids"].astype(np.int64), index["NORAD_CAT_ID"].to_numpy(np.int64)
        ),
        "labels": np.array_equal(
            data["labels"][scope].astype(np.int64), index.loc[scope, "LABEL_ID"].to_numpy(np.int64)
        ),
        "primary_mask": np.array_equal(
            data["primary_mask"][scope].astype(bool), primary.to_numpy(bool)[scope]
        ),
        "target_unix_us": np.array_equal(data["target_unix_us"].astype(np.int64), expected_time),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Dataset/index positional alignment failed for: {failed}")


def fit_normalizer(
    contexts: np.ndarray, mask: np.ndarray, clip_after_normalization: float = 20.0
) -> dict[str, object]:
    values = np.asarray(contexts[mask], dtype=np.float64).reshape(-1, contexts.shape[-1])
    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return {
        "center_median": center.astype(float).tolist(),
        "scale_iqr_or_std": scale.astype(float).tolist(),
        "fit_window_count": int(mask.sum()),
        "fit_observation_count": int(mask.sum() * contexts.shape[1]),
        "fit_scope": "selected training NORAD IDs only",
        "clip_after_normalization": float(clip_after_normalization),
    }


def normalize(values: np.ndarray, normalizer: dict[str, object]) -> np.ndarray:
    center = np.asarray(normalizer["center_median"], dtype=np.float32)
    scale = np.asarray(normalizer["scale_iqr_or_std"], dtype=np.float32)
    result = (np.asarray(values, dtype=np.float32) - center) / scale
    clip = float(normalizer.get("clip_after_normalization", 20.0))
    result = np.nan_to_num(result, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(result, -clip, clip)


def denormalize(values: np.ndarray, normalizer: dict[str, object]) -> np.ndarray:
    center = np.asarray(normalizer["center_median"], dtype=np.float32)
    scale = np.asarray(normalizer["scale_iqr_or_std"], dtype=np.float32)
    return np.asarray(values, dtype=np.float32) * scale + center


def save_normalizer(normalizer: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(normalizer, indent=2), encoding="utf-8")


def ranked_training_ids(norad_ids: np.ndarray, train_mask: np.ndarray, seed: int) -> list[int]:
    unique_ids = sorted(set(np.asarray(norad_ids[train_mask], dtype=int).tolist()))

    def rank(value: int) -> bytes:
        return hashlib.sha256(f"fraction:{seed}:{value}".encode("ascii")).digest()

    return sorted(unique_ids, key=rank)


def fraction_training_mask(
    norad_ids: np.ndarray,
    split_codes: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    train_mask = np.asarray(split_codes) == 0
    ranked = ranked_training_ids(norad_ids, train_mask, seed)
    count = max(1, math.ceil(len(ranked) * fraction))
    selected_ids = ranked[:count]
    selected_mask = train_mask & np.isin(norad_ids, selected_ids)
    return selected_mask, selected_ids
