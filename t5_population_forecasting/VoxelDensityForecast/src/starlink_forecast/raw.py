from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .config import LeoSpec, SamplingSpec
from .physics import OrbitSummary, osculating_orbit_summary
from .progress import ProgressBar


REQUIRED_COLUMNS = (
    "Epoch_UTC",
    "X_m",
    "Y_m",
    "Z_m",
    "VX_m_s",
    "VY_m_s",
    "VZ_m_s",
)

EPOCH_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)


@dataclass(frozen=True)
class SatelliteInspection:
    satellite_id: str | None
    path: str
    size_bytes: int
    status: str
    reason: str | None
    first_epoch_utc: str | None
    orbit: OrbitSummary | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value


def discover_state_files(
    root: str | Path,
    pattern: str = "*_pruned.csv",
    *,
    recursive: bool = True,
) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Raw state directory does not exist: {directory}")
    values = directory.rglob(pattern) if recursive else directory.glob(pattern)
    return sorted(path for path in values if path.is_file())


def satellite_id_from_path(path: Path, pattern: str) -> str:
    match = re.fullmatch(pattern, path.name)
    if match is None or "satellite_id" not in match.groupdict():
        raise ValueError(f"Filename does not match satellite_id_regex: {path.name}")
    return str(match.group("satellite_id"))


def read_selection_file(path: str | Path | None) -> set[str] | None:
    if path in (None, "", "null"):
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Selection file does not exist: {source}")
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("satellite_ids")
        if not isinstance(value, list):
            raise ValueError("JSON selection must be a list or contain satellite_ids")
        return {str(item).strip() for item in value if str(item).strip()}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        handle.seek(0)
        if "," in first and "satellite_id" in first.lower():
            reader = csv.DictReader(handle)
            column = next((name for name in reader.fieldnames or [] if name.lower() == "satellite_id"), None)
            if column is None:
                raise ValueError("CSV selection is missing satellite_id")
            return {str(row[column]).strip() for row in reader if str(row.get(column, "")).strip()}
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def parse_epoch_utc(values: pd.Series) -> pd.Series:
    """Parse source UTC epochs without falling into pandas' slow per-row inference path."""

    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, utc=True, errors="coerce", format=EPOCH_FORMATS[0])
    missing = parsed.isna() & text.notna()
    for epoch_format in EPOCH_FORMATS[1:]:
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], utc=True, errors="coerce", format=epoch_format)
        missing = parsed.isna() & text.notna()
    if missing.any():
        # Some pandas versions warn loudly when falling back to dateutil. Keep that
        # path only for genuinely irregular timestamps and preserve invalid rows.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
            try:
                fallback = pd.to_datetime(text.loc[missing], utc=True, errors="coerce", format="mixed")
            except ValueError:
                fallback = pd.to_datetime(text.loc[missing], utc=True, errors="coerce")
        parsed.loc[missing] = fallback
    return parsed


def _first_valid_row(path: Path) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS), nrows=64)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    epoch = parse_epoch_utc(frame["Epoch_UTC"])
    numeric = frame[list(REQUIRED_COLUMNS[1:])].apply(pd.to_numeric, errors="coerce")
    valid = epoch.notna() & np.isfinite(numeric.to_numpy(np.float64)).all(axis=1)
    if not valid.any():
        raise ValueError("first 64 rows contain no valid Cartesian state")
    index = int(np.flatnonzero(valid.to_numpy())[0])
    values = numeric.iloc[index].to_numpy(np.float64)
    return pd.Timestamp(epoch.iloc[index]), values[:3] / 1000.0, values[3:] / 1000.0


def inspect_satellite_file(
    path: str | Path,
    satellite_id_regex: str,
    leo: LeoSpec,
) -> SatelliteInspection:
    source = Path(path)
    try:
        satellite_id = satellite_id_from_path(source, satellite_id_regex)
    except Exception as exc:
        return SatelliteInspection(None, str(source), source.stat().st_size, "invalid", str(exc), None, None)
    try:
        epoch, position, velocity = _first_valid_row(source)
        orbit = osculating_orbit_summary(position, velocity)
        is_leo = (
            orbit.bound_elliptic
            and orbit.perigee_altitude_km >= leo.min_perigee_altitude_km
            and orbit.apogee_altitude_km <= leo.max_apogee_altitude_km
        )
        status = "leo" if is_leo else "skipped_non_leo"
        reason = None if is_leo else (
            f"osculating perigee/apogee {orbit.perigee_altitude_km:.3f}/"
            f"{orbit.apogee_altitude_km:.3f} km outside configured LEO bounds"
        )
        return SatelliteInspection(
            satellite_id, str(source), source.stat().st_size, status, reason, epoch.isoformat(), orbit
        )
    except Exception as exc:
        return SatelliteInspection(
            satellite_id, str(source), source.stat().st_size, "invalid", str(exc), None, None
        )


def audit_state_directory(
    root: str | Path,
    *,
    pattern: str,
    recursive: bool,
    satellite_id_regex: str,
    leo: LeoSpec,
    selected_ids: set[str] | None = None,
    max_files: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    files = discover_state_files(root, pattern, recursive=recursive)
    inspections: list[SatelliteInspection] = []
    seen: dict[str, Path] = {}
    bar = ProgressBar("audit raw files", len(files), enabled=progress)
    for path in files:
        try:
            satellite_id = satellite_id_from_path(path, satellite_id_regex)
        except ValueError:
            satellite_id = None
        if selected_ids is not None and satellite_id not in selected_ids:
            bar.advance(detail=path.name)
            continue
        if satellite_id is not None and satellite_id in seen:
            inspections.append(
                SatelliteInspection(
                    satellite_id,
                    str(path),
                    path.stat().st_size,
                    "invalid",
                    f"duplicate satellite ID also found at {seen[satellite_id]}",
                    None,
                    None,
                )
            )
            bar.advance(detail=f"{satellite_id} duplicate")
            continue
        if satellite_id is not None:
            seen[satellite_id] = path
        inspections.append(inspect_satellite_file(path, satellite_id_regex, leo))
        bar.advance(detail=path.name)
        if max_files is not None and len(inspections) >= max_files:
            break
    bar.finish(detail=f"inspected {len(inspections)}")
    found_ids = {item.satellite_id for item in inspections if item.satellite_id is not None}
    missing_selected = sorted(selected_ids - found_ids) if selected_ids is not None else []
    counts: dict[str, int] = {}
    for item in inspections:
        counts[item.status] = counts.get(item.status, 0) + 1
    return {
        "schema_version": 1,
        "root": str(Path(root)),
        "pattern": pattern,
        "recursive": recursive,
        "discovered_file_count": len(files),
        "inspected_file_count": len(inspections),
        "status_counts": counts,
        "missing_selected_ids": missing_selected,
        "files": [item.to_dict() for item in inspections],
    }


def hash_file(path: str | Path, algorithm: str = "sha256", block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def extract_causal_anchor_states(path: str | Path, sampling: SamplingSpec) -> dict[str, np.ndarray | int]:
    source = Path(path)
    cadence_ns = np.int64(sampling.cadence_minutes * 60 * 1_000_000_000)
    max_age_ns = sampling.causal_max_age_seconds * 1.0e9
    best_epoch_by_anchor: dict[int, int] = {}
    best_state_by_anchor: dict[int, np.ndarray] = {}
    input_rows = 0
    invalid_rows = 0
    for chunk in pd.read_csv(
        source,
        usecols=list(REQUIRED_COLUMNS),
        chunksize=sampling.chunksize_rows,
    ):
        input_rows += len(chunk)
        parsed = parse_epoch_utc(chunk["Epoch_UTC"])
        numeric = chunk[list(REQUIRED_COLUMNS[1:])].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(np.float64)
        valid = parsed.notna().to_numpy() & np.isfinite(values).all(axis=1)
        invalid_rows += int(np.count_nonzero(~valid))
        if not np.any(valid):
            continue
        epoch_ns = pd.DatetimeIndex(parsed[valid]).as_unit("ns").asi8
        valid_values = values[valid]
        anchor_ns = -np.floor_divide(-epoch_ns, cadence_ns) * cadence_ns
        age_ns = anchor_ns - epoch_ns
        causal = (age_ns >= 0) & (age_ns <= max_age_ns)
        if not np.any(causal):
            continue
        causal_anchor = anchor_ns[causal]
        causal_epoch = epoch_ns[causal]
        causal_state = valid_values[causal]
        order = np.lexsort((causal_epoch, causal_anchor))
        causal_anchor = causal_anchor[order]
        causal_epoch = causal_epoch[order]
        causal_state = causal_state[order]
        keep = np.r_[causal_anchor[1:] != causal_anchor[:-1], True]
        for anchor_value, epoch_value, state_value in zip(
            causal_anchor[keep],
            causal_epoch[keep],
            causal_state[keep],
            strict=True,
        ):
            anchor_key = int(anchor_value)
            epoch_int = int(epoch_value)
            previous_epoch = best_epoch_by_anchor.get(anchor_key)
            if previous_epoch is None or epoch_int >= previous_epoch:
                best_epoch_by_anchor[anchor_key] = epoch_int
                best_state_by_anchor[anchor_key] = state_value.astype(np.float64, copy=True)
    if not best_epoch_by_anchor:
        return {
            "time_ns": np.empty(0, dtype=np.int64),
            "selected_epoch_ns": np.empty(0, dtype=np.int64),
            "state_age_seconds": np.empty(0, dtype=np.float32),
            "position_teme_km": np.empty((0, 3), dtype=np.float64),
            "velocity_teme_km_s": np.empty((0, 3), dtype=np.float64),
            "input_rows": input_rows,
            "invalid_rows": invalid_rows,
        }
    all_anchor = np.asarray(sorted(best_epoch_by_anchor), dtype=np.int64)
    all_epoch = np.asarray([best_epoch_by_anchor[int(anchor)] for anchor in all_anchor], dtype=np.int64)
    all_state = np.vstack([best_state_by_anchor[int(anchor)] for anchor in all_anchor]).astype(np.float64)
    return {
        "time_ns": all_anchor,
        "selected_epoch_ns": all_epoch,
        "state_age_seconds": ((all_anchor - all_epoch) / 1.0e9).astype(np.float32),
        "position_teme_km": all_state[:, :3] / 1000.0,
        "velocity_teme_km_s": all_state[:, 3:] / 1000.0,
        "input_rows": input_rows,
        "invalid_rows": invalid_rows,
    }
