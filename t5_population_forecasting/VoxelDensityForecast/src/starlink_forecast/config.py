from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GridSpec:
    height_min_km: float
    height_max_km: float
    height_step_km: float
    lat_step_deg: float
    lon_step_deg: float

    @property
    def shape(self) -> tuple[int, int, int]:
        values = (
            (self.height_max_km - self.height_min_km) / self.height_step_km,
            180.0 / self.lat_step_deg,
            360.0 / self.lon_step_deg,
        )
        rounded = tuple(round(value) for value in values)
        if any(abs(value - integer) > 1.0e-9 for value, integer in zip(values, rounded)):
            raise ValueError("Grid ranges must be exactly divisible by their step sizes")
        return rounded


@dataclass(frozen=True)
class LeoSpec:
    min_perigee_altitude_km: float
    max_apogee_altitude_km: float


@dataclass(frozen=True)
class SamplingSpec:
    cadence_minutes: int
    causal_max_age_seconds: float
    chunksize_rows: int


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    value["_config_path"] = str(config_path)
    value["_project_dir"] = str(config_path.parent.parent)
    return value


def grid_from_config(config: dict[str, Any]) -> GridSpec:
    grid = config["grid"]
    result = GridSpec(
        height_min_km=float(grid["height_min_km"]),
        height_max_km=float(grid["height_max_km"]),
        height_step_km=float(grid["height_step_km"]),
        lat_step_deg=float(grid["lat_step_deg"]),
        lon_step_deg=float(grid["lon_step_deg"]),
    )
    if min(result.shape) <= 0:
        raise ValueError(f"Invalid grid shape: {result.shape}")
    if result.height_min_km >= result.height_max_km:
        raise ValueError("height_min_km must be below height_max_km")
    if result.lat_step_deg <= 0 or result.lon_step_deg <= 0:
        raise ValueError("latitude and longitude steps must be positive")
    return result


def leo_from_config(config: dict[str, Any]) -> LeoSpec:
    selection = config["selection"]
    result = LeoSpec(
        min_perigee_altitude_km=float(selection["min_perigee_altitude_km"]),
        max_apogee_altitude_km=float(selection["max_apogee_altitude_km"]),
    )
    if result.min_perigee_altitude_km < 0:
        raise ValueError("min_perigee_altitude_km cannot be negative")
    if result.min_perigee_altitude_km >= result.max_apogee_altitude_km:
        raise ValueError("LEO perigee threshold must be below the apogee threshold")
    return result


def sampling_from_config(config: dict[str, Any]) -> SamplingSpec:
    sampling = config["sampling"]
    result = SamplingSpec(
        cadence_minutes=int(sampling["cadence_minutes"]),
        causal_max_age_seconds=float(sampling["causal_max_age_seconds"]),
        chunksize_rows=int(sampling["chunksize_rows"]),
    )
    if result.cadence_minutes <= 0 or 60 % result.cadence_minutes:
        raise ValueError("cadence_minutes must be a positive divisor of 60")
    if result.causal_max_age_seconds < 0:
        raise ValueError("causal_max_age_seconds cannot be negative")
    if result.chunksize_rows <= 0:
        raise ValueError("chunksize_rows must be positive")
    return result


def resolve_project_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_project_dir"]) / path


def config_digest(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
