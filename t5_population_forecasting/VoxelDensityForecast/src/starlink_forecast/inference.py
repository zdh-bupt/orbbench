from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .closed_cohort import _state_cache_paths
from .config import grid_from_config
from .coordinates import teme_to_itrs_wgs84
from .physics import propagate_kepler
from .voxel import aggregate_cic, build_feature_frame


def _exact_index(times: np.ndarray, value: int) -> int | None:
    index = int(np.searchsorted(times, value))
    return index if index < len(times) and int(times[index]) == value else None


def build_inference_sample(
    config: dict[str, Any],
    *,
    origin_time: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    paths = _state_cache_paths(config)
    cadence_minutes = int(config["sampling"]["cadence_minutes"])
    cadence_ns = np.int64(cadence_minutes * 60 * 1_000_000_000)
    history_steps = int(config["forecast"]["history_steps"])
    horizons = tuple(int(value) for value in config["forecast"]["horizons"])
    if origin_time is None:
        latest: list[int] = []
        for _, path in paths:
            parquet = pq.ParquetFile(path)
            column = parquet.read_row_group(parquet.num_row_groups - 1, columns=["time_ns"])["time_ns"]
            latest.append(int(column[len(column) - 1].as_py()))
        origin_ns = max(latest)
    else:
        timestamp = pd.Timestamp(origin_time)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")
        origin_ns = int(timestamp.value)
        if origin_ns % int(cadence_ns):
            raise ValueError("origin_time must lie exactly on the configured UTC cadence")

    history_start = origin_ns - (history_steps - 1) * int(cadence_ns)
    caches: list[dict[str, Any]] = []
    columns = [
        "time_ns", "state_age_seconds",
        "x_teme_km", "y_teme_km", "z_teme_km",
        "vx_teme_km_s", "vy_teme_km_s", "vz_teme_km_s",
        "latitude_deg", "longitude_deg", "altitude_km",
        "v_east_km_s", "v_north_km_s", "v_up_km_s",
    ]
    for satellite_id, path in paths:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("time_ns", ">=", history_start), ("time_ns", "<=", origin_ns)],
        ).sort_values("time_ns")
        expected = origin_ns - np.arange(history_steps - 1, -1, -1, dtype=np.int64) * cadence_ns
        if len(frame) == history_steps and np.array_equal(frame["time_ns"].to_numpy(np.int64), expected):
            caches.append({"satellite_id": satellite_id, "path": path, "frame": frame.reset_index(drop=True)})

    if len(caches) < int(config["forecast"].get("min_origin_members", 1)):
        raise ValueError("No sufficient LEO members have a complete causal history at the requested origin")

    grid = grid_from_config(config)
    member_count = len(caches)
    features: list[np.ndarray] = []
    for history_index in range(history_steps):
        geodetic = np.stack(
            [cache["frame"].loc[history_index, ["latitude_deg", "longitude_deg", "altitude_km"]].to_numpy(np.float64) for cache in caches]
        )
        velocity = np.stack(
            [cache["frame"].loc[history_index, ["v_east_km_s", "v_north_km_s", "v_up_km_s"]].to_numpy(np.float64) for cache in caches]
        )
        age = np.asarray([cache["frame"].loc[history_index, "state_age_seconds"] for cache in caches])
        aggregate = aggregate_cic(
            geodetic[:, 0], geodetic[:, 1], geodetic[:, 2], velocity, age, grid
        )
        features.append(
            build_feature_frame(
                aggregate,
                origin_member_count=member_count,
                observed_member_count=member_count,
                time_ns=history_start + history_index * int(cadence_ns),
                velocity_scale_km_s=float(config["forecast"]["velocity_scale_km_s"]),
                maximum_age_seconds=float(config["sampling"]["causal_max_age_seconds"]),
                grid=grid,
            )
        )

    origin_position = np.stack(
        [cache["frame"].loc[history_steps - 1, ["x_teme_km", "y_teme_km", "z_teme_km"]].to_numpy(np.float64) for cache in caches]
    )
    origin_velocity = np.stack(
        [cache["frame"].loc[history_steps - 1, ["vx_teme_km_s", "vy_teme_km_s", "vz_teme_km_s"]].to_numpy(np.float64) for cache in caches]
    )
    prior_counts: list[np.ndarray] = []
    prior_outside: list[np.ndarray] = []
    zero_velocity = np.zeros((member_count, 3), dtype=np.float64)
    zero_age = np.zeros(member_count, dtype=np.float64)
    for horizon in horizons:
        position, velocity = propagate_kepler(
            origin_position, origin_velocity, float(horizon * 3600)
        )
        transformed = teme_to_itrs_wgs84(
            np.full(member_count, origin_ns + horizon * 3_600_000_000_000, dtype=np.int64),
            position,
            velocity,
            batch_size=int(config["sampling"].get("coordinate_batch_rows", 50_000)),
        )
        aggregate = aggregate_cic(
            transformed["latitude_deg"],
            transformed["longitude_deg"],
            transformed["altitude_km"],
            zero_velocity,
            zero_age,
            grid,
        )
        prior_counts.append(aggregate.count_mass.astype(np.float32))
        prior_outside.append(aggregate.outside_count.astype(np.float32))
    return {
        "features": np.stack(features),
        "physics_prior": np.stack(prior_counts),
        "physics_prior_outside": np.stack(prior_outside),
        "future_total": np.full(len(horizons), member_count, dtype=np.float32),
        "origin_time_ns": np.int64(origin_ns),
        "satellite_ids": np.asarray([cache["satellite_id"] for cache in caches], dtype="U32"),
        "horizons": np.asarray(horizons, dtype=np.int16),
    }


def normalized_physics_prediction(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    voxel = np.asarray(sample["physics_prior"], dtype=np.float64)
    outside = np.asarray(sample["physics_prior_outside"], dtype=np.float64)
    totals = np.asarray(sample["future_total"], dtype=np.float64)
    result_voxel = np.empty_like(voxel)
    result_outside = np.empty_like(outside)
    for index in range(len(totals)):
        mass = voxel[index].sum() + outside[index].sum()
        if mass <= 0:
            raise ValueError("physical prior contains no mass")
        result_voxel[index] = voxel[index] * (totals[index] / mass)
        result_outside[index] = outside[index] * (totals[index] / mass)
    return result_voxel.astype(np.float32), result_outside.astype(np.float32)
