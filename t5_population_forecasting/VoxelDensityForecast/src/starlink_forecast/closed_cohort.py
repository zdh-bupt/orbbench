from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import config_digest, grid_from_config, resolve_project_path
from .coordinates import teme_to_itrs_wgs84
from .physics import propagate_kepler
from .progress import ProgressBar
from .voxel import FEATURE_NAMES, aggregate_cic, build_feature_frame


def _utc(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tz is None else result.tz_convert("UTC")


def _state_cache_paths(config: dict[str, Any]) -> list[tuple[str, Path]]:
    state_dir = resolve_project_path(config, config["paths"]["state_cache_dir"])
    manifest_path = state_dir / "state_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("state_manifest.json is missing; run scripts/prepare_states.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values: list[tuple[str, Path]] = []
    for entry in manifest.get("satellites", []):
        if entry.get("status") not in {"processed", "reused"}:
            continue
        path = Path(entry["cache_path"])
        if not path.is_file():
            raise FileNotFoundError(f"State cache listed by manifest is missing: {path}")
        values.append((str(entry["satellite_id"]), path))
    if not values:
        raise ValueError("No processed LEO state caches are available")
    return sorted(values)


def _cache_range(path: Path) -> tuple[int, int]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0:
        raise ValueError(f"Empty state cache: {path}")
    first_column = parquet.read_row_group(0, columns=["time_ns"])["time_ns"]
    last_column = parquet.read_row_group(parquet.num_row_groups - 1, columns=["time_ns"])["time_ns"]
    return int(first_column[0].as_py()), int(last_column[len(last_column) - 1].as_py())


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    first = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    return pd.date_range(first, end, freq="MS", inclusive="left")


def _load_month_cube(
    paths: list[tuple[str, Path]],
    times_ns: np.ndarray,
) -> dict[str, np.ndarray]:
    count_times = len(times_ns)
    count_satellites = len(paths)
    position_teme = np.full((count_times, count_satellites, 3), np.nan, dtype=np.float64)
    velocity_teme = np.full_like(position_teme, np.nan)
    geodetic = np.full_like(position_teme, np.nan)
    velocity_enu = np.full_like(position_teme, np.nan)
    state_age = np.full((count_times, count_satellites), np.nan, dtype=np.float32)
    time_start, time_end = int(times_ns[0]), int(times_ns[-1])
    step_ns = int(times_ns[1] - times_ns[0]) if len(times_ns) > 1 else 3_600_000_000_000
    for satellite_index, (_, path) in enumerate(paths):
        frame = pd.read_parquet(
            path,
            columns=[
                "time_ns", "state_age_seconds",
                "x_teme_km", "y_teme_km", "z_teme_km",
                "vx_teme_km_s", "vy_teme_km_s", "vz_teme_km_s",
                "latitude_deg", "longitude_deg", "altitude_km",
                "v_east_km_s", "v_north_km_s", "v_up_km_s",
            ],
            filters=[("time_ns", ">=", time_start), ("time_ns", "<=", time_end)],
        )
        if frame.empty:
            continue
        selected_times = frame["time_ns"].to_numpy(np.int64)
        destination = ((selected_times - time_start) // step_ns).astype(np.int64)
        exact = times_ns[destination] == selected_times
        destination = destination[exact]
        frame = frame.iloc[np.flatnonzero(exact)]
        position_teme[destination, satellite_index] = frame[["x_teme_km", "y_teme_km", "z_teme_km"]]
        velocity_teme[destination, satellite_index] = frame[["vx_teme_km_s", "vy_teme_km_s", "vz_teme_km_s"]]
        geodetic[destination, satellite_index] = frame[["latitude_deg", "longitude_deg", "altitude_km"]]
        velocity_enu[destination, satellite_index] = frame[["v_east_km_s", "v_north_km_s", "v_up_km_s"]]
        state_age[destination, satellite_index] = frame["state_age_seconds"].to_numpy(np.float32)
    valid = (
        np.isfinite(position_teme).all(axis=-1)
        & np.isfinite(velocity_teme).all(axis=-1)
        & np.isfinite(geodetic).all(axis=-1)
        & np.isfinite(velocity_enu).all(axis=-1)
        & np.isfinite(state_age)
    )
    return {
        "position_teme_km": position_teme,
        "velocity_teme_km_s": velocity_teme,
        "geodetic": geodetic,
        "velocity_enu_km_s": velocity_enu,
        "state_age_seconds": state_age,
        "valid": valid,
    }


def _valid_origin_indices(
    times_ns: np.ndarray,
    valid: np.ndarray,
    core_start_ns: int,
    core_end_ns: int,
    history_steps: int,
    horizons: tuple[int, ...],
    min_members: int,
    require_complete_history: bool,
    require_complete_targets: bool,
) -> np.ndarray:
    step_ns = int(times_ns[1] - times_ns[0])
    origins: list[int] = []
    for origin in np.flatnonzero((times_ns >= core_start_ns) & (times_ns < core_end_ns)):
        if origin < history_steps - 1 or origin + max(horizons) >= len(times_ns):
            continue
        members = valid[origin]
        if int(members.sum()) < min_members:
            continue
        if require_complete_history and not valid[origin - history_steps + 1 : origin + 1, members].all():
            continue
        if require_complete_targets and not valid[[origin + horizon for horizon in horizons]][:, members].all():
            continue
        if any(times_ns[origin + horizon] != times_ns[origin] + horizon * step_ns for horizon in horizons):
            continue
        origins.append(int(origin))
    return np.asarray(origins, dtype=np.int64)


def _physical_prior_geodetic(
    times_ns: np.ndarray,
    position_teme: np.ndarray,
    velocity_teme: np.ndarray,
    valid: np.ndarray,
    origins: np.ndarray,
    horizons: tuple[int, ...],
    coordinate_batch: int,
) -> np.ndarray:
    result = np.full((len(origins), len(horizons), position_teme.shape[1], 3), np.nan, dtype=np.float32)
    for lead_index, horizon in enumerate(horizons):
        origin_position = position_teme[origins]
        origin_velocity = velocity_teme[origins]
        member_valid = valid[origins]
        propagated_position, propagated_velocity = propagate_kepler(
            origin_position[member_valid],
            origin_velocity[member_valid],
            float(horizon * 3600),
        )
        target_times = np.broadcast_to(
            times_ns[origins, None] + np.int64(horizon * 3_600_000_000_000),
            member_valid.shape,
        )[member_valid]
        transformed = teme_to_itrs_wgs84(
            target_times,
            propagated_position,
            propagated_velocity,
            batch_size=coordinate_batch,
        )
        values = np.stack(
            (
                transformed["latitude_deg"],
                transformed["longitude_deg"],
                transformed["altitude_km"],
            ),
            axis=-1,
        ).astype(np.float32)
        result[:, lead_index][member_valid] = values
    return result


def build_closed_cohort_shards(
    config: dict[str, Any],
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    overwrite: bool = False,
    max_months: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    paths = _state_cache_paths(config)
    ranges = [_cache_range(path) for _, path in paths]
    cadence_minutes = int(config["sampling"]["cadence_minutes"])
    cadence = pd.Timedelta(minutes=cadence_minutes)
    available_start = pd.Timestamp(min(value[0] for value in ranges), unit="ns", tz="UTC")
    available_end = pd.Timestamp(max(value[1] for value in ranges), unit="ns", tz="UTC") + cadence
    requested_start = _utc(start) or available_start
    requested_end = _utc(end) or available_end
    requested_start = requested_start.ceil(cadence)
    requested_end = requested_end.floor(cadence)
    if requested_start >= requested_end:
        raise ValueError("Requested shard interval is empty")

    output_dir = resolve_project_path(config, config["paths"]["shard_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history_steps = int(config["forecast"]["history_steps"])
    horizons = tuple(int(value) for value in config["forecast"]["horizons"])
    step_ns = np.int64(cadence.value)
    coordinate_batch = int(config["sampling"].get("coordinate_batch_rows", 50_000))
    digest = config_digest(config)
    entries: list[dict[str, Any]] = []
    months = list(_month_starts(requested_start, requested_end))
    if max_months is not None:
        months = months[:max_months]
    bar = ProgressBar("closed cohort shards", len(months), enabled=progress)

    for month_start in months:
        core_start = max(month_start, requested_start)
        core_end = min(month_start + pd.offsets.MonthBegin(1), requested_end)
        if core_start >= core_end:
            bar.advance(detail=f"{month_start.strftime('%Y-%m')} skipped")
            continue
        target = output_dir / f"closed-cohort-{month_start.strftime('%Y-%m')}.npz"
        if target.exists() and not overwrite:
            with np.load(target, allow_pickle=False) as archive:
                if str(archive["config_digest"]) != digest:
                    raise ValueError(f"Existing shard uses a different config: {target}; use --overwrite")
                entries.append(
                    {
                        "path": str(target),
                        "month": month_start.strftime("%Y-%m"),
                        "origin_count": int(len(archive["origin_indices"])),
                        "status": "reused",
                    }
                )
            bar.advance(detail=f"{month_start.strftime('%Y-%m')} reused")
            continue
        context_start = core_start - cadence * (history_steps - 1)
        context_end = core_end + cadence * max(horizons)
        times = pd.date_range(context_start, context_end, freq=cadence, inclusive="left")
        times_ns = times.as_unit("ns").asi8
        cube = _load_month_cube(paths, times_ns)
        origins = _valid_origin_indices(
            times_ns,
            cube["valid"],
            int(core_start.value),
            int(core_end.value),
            history_steps,
            horizons,
            int(config["forecast"].get("min_origin_members", 1)),
            bool(config["forecast"].get("require_complete_history", True)),
            bool(config["forecast"].get("require_complete_targets", True)),
        )
        if not len(origins):
            entries.append(
                {
                    "path": None,
                    "month": month_start.strftime("%Y-%m"),
                    "origin_count": 0,
                    "status": "no_valid_origins",
                }
            )
            bar.advance(detail=f"{month_start.strftime('%Y-%m')} no origins")
            continue
        prior = _physical_prior_geodetic(
            times_ns,
            cube["position_teme_km"],
            cube["velocity_teme_km_s"],
            cube["valid"],
            origins,
            horizons,
            coordinate_batch,
        )
        np.savez_compressed(
            target,
            schema_version=np.int16(config["dataset"]["schema_version"]),
            config_digest=np.str_(digest),
            feature_names=np.asarray(FEATURE_NAMES, dtype="U40"),
            satellite_ids=np.asarray([satellite_id for satellite_id, _ in paths], dtype="U32"),
            horizons_hours=np.asarray(horizons, dtype=np.int16),
            history_steps=np.int16(history_steps),
            cadence_minutes=np.int16(cadence_minutes),
            time_ns=times_ns,
            geodetic=cube["geodetic"].astype(np.float32),
            velocity_enu_km_s=cube["velocity_enu_km_s"].astype(np.float32),
            state_age_seconds=cube["state_age_seconds"].astype(np.float32),
            valid=cube["valid"],
            origin_indices=origins,
            physical_prior_geodetic=prior,
            core_start_ns=np.int64(core_start.value),
            core_end_ns=np.int64(core_end.value),
        )
        entries.append(
            {
                "path": str(target),
                "month": month_start.strftime("%Y-%m"),
                "origin_count": int(len(origins)),
                "origin_time_start": pd.Timestamp(times_ns[origins[0]], unit="ns", tz="UTC").isoformat(),
                "origin_time_end": pd.Timestamp(times_ns[origins[-1]], unit="ns", tz="UTC").isoformat(),
                "status": "written",
            }
        )
        bar.advance(detail=f"{month_start.strftime('%Y-%m')} written")
    bar.finish(detail=f"usable {sum(1 for entry in entries if entry.get('path'))}")

    usable = [entry for entry in entries if entry["path"]]
    manifest = {
        "schema_version": int(config["dataset"]["schema_version"]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_digest": digest,
        "population_semantics": "origin-known members with complete causal history and targets",
        "coordinate_contract": (
            "TEME source propagated causally to UTC anchor, then Astropy TEME to ITRS/WGS84; bundled IERS tables, "
            "auto_download=false, auto_max_age=None"
        ),
        "physical_prior": "causal two-body Kepler propagation from each origin TEME state",
        "feature_names": list(FEATURE_NAMES),
        "satellite_ids": [satellite_id for satellite_id, _ in paths],
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "horizons_hours": list(horizons),
        "history_steps": history_steps,
        "shards": entries,
        "usable_shard_count": len(usable),
        "total_origin_count": sum(entry["origin_count"] for entry in usable),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


class ClosedCohortDataset:
    def __init__(
        self,
        shard_path: str | Path,
        config: dict[str, Any],
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        return_tensors: bool = True,
    ) -> None:
        # A compressed .npz cannot be memory-mapped by member.  These arrays are
        # deliberately materialized for one active shard only; callers must not
        # construct datasets for every shard at once.
        with np.load(Path(shard_path), allow_pickle=False) as archive:
            if str(archive["config_digest"]) != config_digest(config):
                raise ValueError("Shard and runtime configuration digests do not match")
            self.times_ns = archive["time_ns"]
            self.geodetic = archive["geodetic"]
            self.velocity_enu = archive["velocity_enu_km_s"]
            self.state_age = archive["state_age_seconds"]
            self.valid = archive["valid"]
            all_origins = archive["origin_indices"]
            all_prior = archive["physical_prior_geodetic"]
            mask = np.ones(len(all_origins), dtype=bool)
            if start is not None:
                mask &= self.times_ns[all_origins] >= int(start.value)
            if end is not None:
                mask &= self.times_ns[all_origins] <= int(end.value)
            if bool(mask.all()):
                self.origins = all_origins
                self.prior_geodetic = all_prior
            else:
                self.origins = all_origins[mask]
                self.prior_geodetic = all_prior[mask]
            self.horizons = tuple(int(value) for value in archive["horizons_hours"])
            self.history_steps = int(archive["history_steps"])
        self.grid = grid_from_config(config)
        self.velocity_scale = float(config["forecast"]["velocity_scale_km_s"])
        self.max_age = float(config["sampling"]["causal_max_age_seconds"])
        self.return_tensors = return_tensors
        if len(FEATURE_NAMES) != int(config["model"]["input_channels"]):
            raise ValueError("model.input_channels does not match the formal feature contract")

    def __len__(self) -> int:
        return len(self.origins)

    @staticmethod
    def _positions(geodetic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return geodetic[:, 0], geodetic[:, 1], geodetic[:, 2]

    def __getitem__(self, item: int) -> dict[str, Any]:
        origin = int(self.origins[item])
        members = self.valid[origin]
        member_count = int(members.sum())
        features: list[np.ndarray] = []
        for history_index in range(origin - self.history_steps + 1, origin + 1):
            observed = members & self.valid[history_index]
            geodetic = self.geodetic[history_index, observed]
            aggregate = aggregate_cic(
                *self._positions(geodetic),
                self.velocity_enu[history_index, observed],
                self.state_age[history_index, observed],
                self.grid,
            )
            features.append(
                build_feature_frame(
                    aggregate,
                    origin_member_count=member_count,
                    observed_member_count=int(observed.sum()),
                    time_ns=int(self.times_ns[history_index]),
                    velocity_scale_km_s=self.velocity_scale,
                    maximum_age_seconds=self.max_age,
                    grid=self.grid,
                )
            )

        prior_counts: list[np.ndarray] = []
        prior_outside: list[np.ndarray] = []
        target_counts: list[np.ndarray] = []
        target_outside: list[np.ndarray] = []
        target_hard_counts: list[np.ndarray] = []
        target_hard_outside: list[np.ndarray] = []
        zeros_velocity = np.zeros((member_count, 3), dtype=np.float32)
        zeros_age = np.zeros(member_count, dtype=np.float32)
        for lead_index, horizon in enumerate(self.horizons):
            prior_geo = self.prior_geodetic[item, lead_index, members]
            prior = aggregate_cic(
                *self._positions(prior_geo), zeros_velocity, zeros_age, self.grid
            )
            target_index = origin + horizon
            target_geo = self.geodetic[target_index, members]
            target = aggregate_cic(
                *self._positions(target_geo),
                self.velocity_enu[target_index, members],
                self.state_age[target_index, members],
                self.grid,
            )
            prior_counts.append(prior.count_mass.astype(np.float32))
            prior_outside.append(prior.outside_count.astype(np.float32))
            target_counts.append(target.count_mass.astype(np.float32))
            target_outside.append(target.outside_count.astype(np.float32))
            target_hard_counts.append(target.hard_count.astype(np.float32))
            target_hard_outside.append(target.hard_outside_count.astype(np.float32))

        output: dict[str, Any] = {
            "features": np.stack(features),
            "physics_prior": np.stack(prior_counts),
            "physics_prior_outside": np.stack(prior_outside),
            "future_total": np.full(len(self.horizons), member_count, dtype=np.float32),
            "target_counts": np.stack(target_counts),
            "target_outside": np.stack(target_outside),
            "target_hard_counts": np.stack(target_hard_counts),
            "target_hard_outside": np.stack(target_hard_outside),
            "origin_time_ns": np.int64(self.times_ns[origin]),
            "origin_member_count": np.int32(member_count),
        }
        if self.return_tensors:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("PyTorch is required when return_tensors=True") from exc
            return {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in output.items()
            }
        return output
