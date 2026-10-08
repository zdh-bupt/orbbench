from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    config_digest,
    leo_from_config,
    resolve_project_path,
    sampling_from_config,
)
from .coordinates import teme_to_itrs_wgs84
from .physics import propagate_kepler
from .raw import (
    audit_state_directory,
    extract_causal_anchor_states,
    hash_file,
    read_selection_file,
)
from .progress import ProgressBar


def propagate_states_to_anchors(
    position_teme_km: np.ndarray,
    velocity_teme_km_s: np.ndarray,
    state_age_seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    age = np.asarray(state_age_seconds, dtype=np.float64)
    if np.any(age < 0) or not np.isfinite(age).all():
        raise ValueError("state_age_seconds must be finite and non-negative")
    return propagate_kepler(position_teme_km, velocity_teme_km_s, age)


def _selection_path(config: dict[str, Any], override: str | Path | None) -> Path | None:
    value = override if override is not None else config["selection"].get("selection_file")
    if value in (None, "", "null"):
        return None
    path = Path(value)
    return path if path.is_absolute() else resolve_project_path(config, str(path))


def _cache_is_current(
    path: Path,
    source: Path,
    digest: str,
) -> bool:
    try:
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        return (
            int(metadata[b"source_size_bytes"]) == source.stat().st_size
            and int(metadata[b"source_mtime_ns"]) == source.stat().st_mtime_ns
            and metadata[b"config_digest"].decode("utf-8") == digest
        )
    except (OSError, KeyError, ValueError, TypeError):
        return False


def _cache_summary(path: Path, source: Path, satellite_id: str, reused: bool) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    first = parquet.read_row_group(0, columns=["time_ns"])["time_ns"][0].as_py()
    last_group = parquet.num_row_groups - 1
    last_column = parquet.read_row_group(last_group, columns=["time_ns"])["time_ns"]
    last = last_column[len(last_column) - 1].as_py()
    return {
        "satellite_id": satellite_id,
        "source_path": str(source),
        "cache_path": str(path),
        "status": "reused" if reused else "processed",
        "sample_count": int(parquet.metadata.num_rows),
        "time_start_utc": np.datetime_as_string(np.datetime64(first, "ns"), unit="s") + "Z",
        "time_end_utc": np.datetime_as_string(np.datetime64(last, "ns"), unit="s") + "Z",
        "source_sha256": metadata[b"source_sha256"].decode("utf-8"),
        "input_rows": int(metadata[b"input_rows"]),
        "invalid_rows": int(metadata[b"invalid_rows"]),
    }


def _write_state_parquet(
    target: Path,
    satellite_id: str,
    source: Path,
    digest: str,
    source_sha256: str,
    extracted: dict[str, Any],
    transformed: dict[str, np.ndarray],
    schema_version: int,
) -> None:
    source_position_teme = np.asarray(extracted["position_teme_km"])
    source_velocity_teme = np.asarray(extracted["velocity_teme_km_s"])
    position_teme = np.asarray(extracted["anchor_position_teme_km"])
    velocity_teme = np.asarray(extracted["anchor_velocity_teme_km_s"])
    position_itrs = transformed["position_itrs_km"]
    velocity_itrs = transformed["velocity_itrs_km_s"]
    velocity_enu = transformed["velocity_enu_km_s"]
    table = pa.table(
        {
            "time_ns": np.asarray(extracted["time_ns"], dtype=np.int64),
            "selected_epoch_ns": np.asarray(extracted["selected_epoch_ns"], dtype=np.int64),
            "state_age_seconds": np.asarray(extracted["state_age_seconds"], dtype=np.float32),
            "source_x_teme_km": source_position_teme[:, 0],
            "source_y_teme_km": source_position_teme[:, 1],
            "source_z_teme_km": source_position_teme[:, 2],
            "source_vx_teme_km_s": source_velocity_teme[:, 0],
            "source_vy_teme_km_s": source_velocity_teme[:, 1],
            "source_vz_teme_km_s": source_velocity_teme[:, 2],
            "x_teme_km": position_teme[:, 0],
            "y_teme_km": position_teme[:, 1],
            "z_teme_km": position_teme[:, 2],
            "vx_teme_km_s": velocity_teme[:, 0],
            "vy_teme_km_s": velocity_teme[:, 1],
            "vz_teme_km_s": velocity_teme[:, 2],
            "x_itrs_km": position_itrs[:, 0],
            "y_itrs_km": position_itrs[:, 1],
            "z_itrs_km": position_itrs[:, 2],
            "vx_itrs_km_s": velocity_itrs[:, 0],
            "vy_itrs_km_s": velocity_itrs[:, 1],
            "vz_itrs_km_s": velocity_itrs[:, 2],
            "latitude_deg": transformed["latitude_deg"],
            "longitude_deg": transformed["longitude_deg"],
            "altitude_km": transformed["altitude_km"],
            "v_east_km_s": velocity_enu[:, 0],
            "v_north_km_s": velocity_enu[:, 1],
            "v_up_km_s": velocity_enu[:, 2],
        }
    )
    metadata = {
        b"schema_version": str(schema_version).encode("ascii"),
        b"satellite_id": satellite_id.encode("utf-8"),
        b"source_path": str(source).encode("utf-8"),
        b"source_size_bytes": str(source.stat().st_size).encode("ascii"),
        b"source_mtime_ns": str(source.stat().st_mtime_ns).encode("ascii"),
        b"source_sha256": source_sha256.encode("ascii"),
        b"config_digest": digest.encode("ascii"),
        b"coordinate_frame": b"TEME source propagated causally to UTC anchor",
        b"output_frame": b"ITRS/WGS84",
        b"input_rows": str(extracted["input_rows"]).encode("ascii"),
        b"invalid_rows": str(extracted["invalid_rows"]).encode("ascii"),
    }
    pq.write_table(
        table.replace_schema_metadata(metadata),
        target,
        compression="zstd",
        row_group_size=744,
        write_statistics=True,
    )


def prepare_state_caches(
    config: dict[str, Any],
    *,
    data_root: str | Path | None = None,
    selection_file: str | Path | None = None,
    overwrite: bool = False,
    max_files: int | None = None,
    hash_sources: bool = True,
    progress: bool = False,
    chunksize_rows: int | None = None,
    coordinate_batch_rows: int | None = None,
) -> dict[str, Any]:
    selection = config["selection"]
    raw_root = Path(data_root or config["paths"]["raw_state_dir"])
    if not raw_root.is_absolute():
        raw_root = resolve_project_path(config, str(raw_root))
    selected_ids = read_selection_file(_selection_path(config, selection_file))
    audit = audit_state_directory(
        raw_root,
        pattern=str(selection["file_pattern"]),
        recursive=bool(selection.get("recursive", True)),
        satellite_id_regex=str(selection["satellite_id_regex"]),
        leo=leo_from_config(config),
        selected_ids=selected_ids,
        max_files=max_files,
        progress=progress,
    )
    output_dir = resolve_project_path(config, config["paths"]["state_cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sampling = sampling_from_config(config)
    if chunksize_rows is not None:
        if chunksize_rows <= 0:
            raise ValueError("chunksize_rows must be positive")
        sampling = replace(sampling, chunksize_rows=int(chunksize_rows))
    digest = config_digest(config)
    coordinate_batch = int(coordinate_batch_rows or config["sampling"].get("coordinate_batch_rows", 50_000))
    if coordinate_batch <= 0:
        raise ValueError("coordinate_batch_rows must be positive")
    entries: list[dict[str, Any]] = []
    bar = ProgressBar("state caches", len(audit["files"]), enabled=progress)

    for inspected in audit["files"]:
        if inspected["status"] != "leo":
            entries.append(
                {
                    "satellite_id": inspected["satellite_id"],
                    "source_path": inspected["path"],
                    "status": inspected["status"],
                    "reason": inspected["reason"],
                    "orbit": inspected["orbit"],
                }
            )
            bar.advance(detail=f"{inspected['satellite_id']} {inspected['status']}")
            continue
        source = Path(inspected["path"])
        satellite_id = str(inspected["satellite_id"])
        target = output_dir / f"satellite-{satellite_id}.parquet"
        if target.exists() and not overwrite and _cache_is_current(target, source, digest):
            entries.append(_cache_summary(target, source, satellite_id, reused=True))
            bar.advance(detail=f"{satellite_id} reused")
            continue
        try:
            extracted = extract_causal_anchor_states(source, sampling)
            time_ns = np.asarray(extracted["time_ns"], dtype=np.int64)
            if not len(time_ns):
                raise ValueError("no rows satisfy the causal anchor and maximum-age rules")
            source_position = np.asarray(extracted["position_teme_km"])
            source_velocity = np.asarray(extracted["velocity_teme_km_s"])
            anchor_position, anchor_velocity = propagate_states_to_anchors(
                source_position,
                source_velocity,
                np.asarray(extracted["state_age_seconds"], dtype=np.float64),
            )
            extracted["anchor_position_teme_km"] = anchor_position
            extracted["anchor_velocity_teme_km_s"] = anchor_velocity
            transformed = teme_to_itrs_wgs84(
                time_ns,
                anchor_position,
                anchor_velocity,
                batch_size=coordinate_batch,
            )
            source_sha256 = hash_file(source) if hash_sources else "not_computed"
            _write_state_parquet(
                target,
                satellite_id,
                source,
                digest,
                source_sha256,
                extracted,
                transformed,
                int(config["dataset"]["schema_version"]),
            )
            entries.append(_cache_summary(target, source, satellite_id, reused=False))
            bar.advance(detail=f"{satellite_id} processed")
        except Exception as exc:
            entries.append(
                {
                    "satellite_id": satellite_id,
                    "source_path": str(source),
                    "status": "processing_error",
                    "reason": str(exc),
                }
            )
            bar.advance(detail=f"{satellite_id} error")
    bar.finish(detail=f"processed/reused {sum(1 for entry in entries if entry['status'] in {'processed', 'reused'})}")

    processed = [entry for entry in entries if entry["status"] in {"processed", "reused"}]
    manifest = {
        "schema_version": int(config["dataset"]["schema_version"]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_digest": digest,
        "raw_root": str(raw_root),
        "selection_file": str(_selection_path(config, selection_file) or ""),
        "coordinate_contract": (
            "input TEME metres and metres/second; preserve source state; causally propagate "
            "to each UTC anchor before caching anchored TEME and ITRS in km; "
            "Astropy bundled IERS tables with auto_download=false and auto_max_age=None"
        ),
        "causal_sampling": {
            "cadence_minutes": sampling.cadence_minutes,
            "max_age_seconds": sampling.causal_max_age_seconds,
            "rule": "latest source epoch not later than each UTC anchor",
        },
        "processed_satellite_count": len(processed),
        "audit": audit,
        "satellites": entries,
    }
    manifest_path = output_dir / "state_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
