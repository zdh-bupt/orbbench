from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import pandas as pd

from .config import GridSpec


FEATURE_NAMES = (
    "count_share",
    "log1p_count_normalized",
    "sum_v_e_normalized",
    "sum_v_n_normalized",
    "sum_v_u_normalized",
    "sum_v_e2_normalized",
    "sum_v_n2_normalized",
    "sum_v_u2_normalized",
    "state_age_mass_normalized",
    "frame_coverage_fraction",
    "outside_below_share",
    "outside_above_share",
    "utc_hour_sin",
    "utc_hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "altitude_center_normalized",
    "latitude_sin",
    "latitude_cos",
    "longitude_sin",
    "longitude_cos",
)


@dataclass
class VoxelAggregate:
    count_mass: np.ndarray
    sum_velocity: np.ndarray
    sum_velocity_squared: np.ndarray
    sum_state_age: np.ndarray
    outside_count: np.ndarray
    hard_count: np.ndarray
    hard_outside_count: np.ndarray
    sum_velocity_cross: np.ndarray | None = None


class CICAssignments(NamedTuple):
    category_index: np.ndarray
    weight: np.ndarray


def _pole_map(latitude_index: np.ndarray, longitude_index: np.ndarray, lat_bins: int, lon_bins: int) -> tuple[np.ndarray, np.ndarray]:
    latitude = latitude_index.copy()
    longitude = longitude_index.copy()
    low = latitude < 0
    latitude[low] = -latitude[low] - 1
    longitude[low] += lon_bins // 2
    high = latitude >= lat_bins
    latitude[high] = 2 * lat_bins - latitude[high] - 1
    longitude[high] += lon_bins // 2
    return np.clip(latitude, 0, lat_bins - 1), longitude % lon_bins


def cic_assignments(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_km: np.ndarray,
    grid: GridSpec,
) -> CICAssignments:
    """Return the eight soft CIC category assignments for each position."""

    latitude = np.asarray(latitude_deg, dtype=np.float64)
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    altitude = np.asarray(altitude_km, dtype=np.float64)
    if latitude.shape != longitude.shape or latitude.shape != altitude.shape or latitude.ndim != 1:
        raise ValueError("CIC assignment coordinates must share one [N] shape")
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all() or not np.isfinite(altitude).all():
        raise ValueError("CIC assignment coordinates must be finite")

    shape = grid.shape
    voxel_category_count = int(np.prod(shape))
    height_coordinate = (altitude - grid.height_min_km) / grid.height_step_km - 0.5
    latitude_coordinate = (latitude + 90.0) / grid.lat_step_deg - 0.5
    longitude_coordinate = (longitude + 180.0) / grid.lon_step_deg - 0.5
    h0 = np.floor(height_coordinate).astype(np.int64)
    lat0 = np.floor(latitude_coordinate).astype(np.int64)
    lon0 = np.floor(longitude_coordinate).astype(np.int64)
    fractions = (
        height_coordinate - h0,
        latitude_coordinate - lat0,
        longitude_coordinate - lon0,
    )
    category_index = np.empty((len(latitude), 8), dtype=np.int64)
    weight = np.empty((len(latitude), 8), dtype=np.float64)
    corner = 0
    for dh in (0, 1):
        height_weight = fractions[0] if dh else 1.0 - fractions[0]
        height_index = h0 + dh
        below = height_index < 0
        above = height_index >= shape[0]
        for dlat in (0, 1):
            latitude_weight = fractions[1] if dlat else 1.0 - fractions[1]
            for dlon in (0, 1):
                longitude_weight = fractions[2] if dlon else 1.0 - fractions[2]
                contribution = height_weight * latitude_weight * longitude_weight
                categories = np.empty(len(latitude), dtype=np.int64)
                categories[below] = voxel_category_count
                categories[above] = voxel_category_count + 1
                inside = ~(below | above)
                if np.any(inside):
                    latitude_index, longitude_index = _pole_map(
                        lat0[inside] + dlat,
                        lon0[inside] + dlon,
                        shape[1],
                        shape[2],
                    )
                    categories[inside] = np.ravel_multi_index(
                        (height_index[inside], latitude_index, longitude_index),
                        shape,
                    )
                category_index[:, corner] = categories
                weight[:, corner] = contribution
                corner += 1
    if not np.allclose(weight.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("CIC assignment weights do not sum to one")
    if np.any(weight < -1.0e-15):
        raise RuntimeError("CIC assignment produced negative weights")
    return CICAssignments(category_index, np.maximum(weight, 0.0))


def aggregate_cic(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_km: np.ndarray,
    velocity_enu_km_s: np.ndarray,
    state_age_seconds: np.ndarray,
    grid: GridSpec,
    *,
    weights: np.ndarray | None = None,
) -> VoxelAggregate:
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    altitude = np.asarray(altitude_km, dtype=np.float64)
    velocity = np.asarray(velocity_enu_km_s, dtype=np.float64)
    age = np.asarray(state_age_seconds, dtype=np.float64)
    if latitude.shape != longitude.shape or latitude.shape != altitude.shape or latitude.shape != age.shape:
        raise ValueError("position and age arrays must share one shape")
    if velocity.shape != (*latitude.shape, 3):
        raise ValueError("velocity must have shape [N, 3]")
    mass = np.ones(latitude.shape, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if mass.shape != latitude.shape or np.any(mass < 0):
        raise ValueError("weights must be non-negative and match positions")
    if not (
        np.isfinite(latitude).all()
        and np.isfinite(longitude).all()
        and np.isfinite(altitude).all()
        and np.isfinite(velocity).all()
        and np.isfinite(age).all()
        and np.isfinite(mass).all()
    ):
        raise ValueError("voxel inputs must be finite")

    shape = grid.shape
    count = np.zeros(shape, dtype=np.float64)
    sum_velocity = np.zeros((3, *shape), dtype=np.float64)
    sum_velocity2 = np.zeros((3, *shape), dtype=np.float64)
    sum_velocity_cross = np.zeros((3, *shape), dtype=np.float64)
    sum_age = np.zeros(shape, dtype=np.float64)
    outside = np.zeros(2, dtype=np.float64)
    hard = np.zeros(shape, dtype=np.float64)
    hard_outside = np.zeros(2, dtype=np.float64)

    hard_below = altitude < grid.height_min_km
    hard_above = altitude >= grid.height_max_km
    hard_outside[0] = mass[hard_below].sum()
    hard_outside[1] = mass[hard_above].sum()
    hard_inside = ~(hard_below | hard_above)
    if np.any(hard_inside):
        hi = np.floor((altitude[hard_inside] - grid.height_min_km) / grid.height_step_km).astype(np.int64)
        lati = np.floor((latitude[hard_inside] + 90.0) / grid.lat_step_deg).astype(np.int64)
        loni = np.floor((longitude[hard_inside] + 180.0) / grid.lon_step_deg).astype(np.int64)
        lati = np.clip(lati, 0, shape[1] - 1)
        loni %= shape[2]
        np.add.at(hard, (hi, lati, loni), mass[hard_inside])

    assignments = cic_assignments(latitude, longitude, altitude, grid)
    voxel_category_count = int(np.prod(shape))
    count_flat = count.reshape(-1)
    sum_age_flat = sum_age.reshape(-1)
    sum_velocity_flat = sum_velocity.reshape(3, -1)
    sum_velocity2_flat = sum_velocity2.reshape(3, -1)
    for corner in range(assignments.category_index.shape[1]):
        category = assignments.category_index[:, corner]
        contribution = mass * assignments.weight[:, corner]
        below = category == voxel_category_count
        above = category == voxel_category_count + 1
        outside[0] += contribution[below].sum()
        outside[1] += contribution[above].sum()
        inside = category < voxel_category_count
        if not np.any(inside):
            continue
        index = category[inside]
        values = contribution[inside]
        np.add.at(count_flat, index, values)
        np.add.at(sum_age_flat, index, values * age[inside])
        for component in range(3):
            component_velocity = velocity[inside, component]
            np.add.at(sum_velocity_flat[component], index, values * component_velocity)
            np.add.at(sum_velocity2_flat[component], index, values * component_velocity**2)

        cross_velocity = (
            velocity[inside, 0] * velocity[inside, 1],
            velocity[inside, 0] * velocity[inside, 2],
            velocity[inside, 1] * velocity[inside, 2],
        )
        sum_velocity_cross_flat = sum_velocity_cross.reshape(3, -1)
        for component, product in enumerate(cross_velocity):
            np.add.at(sum_velocity_cross_flat[component], index, values * product)

    return VoxelAggregate(
        count,
        sum_velocity,
        sum_velocity2,
        sum_age,
        outside,
        hard,
        hard_outside,
        sum_velocity_cross,
    )


@lru_cache(maxsize=16)
def static_geometry_channels(grid: GridSpec) -> np.ndarray:
    depth, latitude_bins, longitude_bins = grid.shape
    altitude = grid.height_min_km + (np.arange(depth) + 0.5) * grid.height_step_km
    latitude = -90.0 + (np.arange(latitude_bins) + 0.5) * grid.lat_step_deg
    longitude = -180.0 + (np.arange(longitude_bins) + 0.5) * grid.lon_step_deg
    altitude_normalized = 2.0 * (altitude - grid.height_min_km) / (
        grid.height_max_km - grid.height_min_km
    ) - 1.0
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)
    channels = [
        np.broadcast_to(altitude_normalized[:, None, None], grid.shape),
        np.broadcast_to(np.sin(lat_rad)[None, :, None], grid.shape),
        np.broadcast_to(np.cos(lat_rad)[None, :, None], grid.shape),
        np.broadcast_to(np.sin(lon_rad)[None, None, :], grid.shape),
        np.broadcast_to(np.cos(lon_rad)[None, None, :], grid.shape),
    ]
    return np.stack(channels).astype(np.float32)


def build_feature_frame(
    aggregate: VoxelAggregate,
    *,
    origin_member_count: int,
    observed_member_count: int,
    time_ns: int,
    velocity_scale_km_s: float,
    maximum_age_seconds: float,
    grid: GridSpec,
) -> np.ndarray:
    if origin_member_count <= 0 or velocity_scale_km_s <= 0 or maximum_age_seconds <= 0:
        raise ValueError("feature scales and origin_member_count must be positive")
    total = float(origin_member_count)
    timestamp = pd.Timestamp(time_ns, unit="ns", tz="UTC")
    hour_phase = 2.0 * np.pi * (
        timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    ) / 24.0
    days_in_year = 366.0 if timestamp.is_leap_year else 365.0
    day_phase = 2.0 * np.pi * (timestamp.dayofyear - 1 + timestamp.hour / 24.0) / days_in_year
    scalar_values = (
        observed_member_count / total,
        aggregate.outside_count[0] / total,
        aggregate.outside_count[1] / total,
        np.sin(hour_phase),
        np.cos(hour_phase),
        np.sin(day_phase),
        np.cos(day_phase),
    )
    dynamic = [
        aggregate.count_mass / total,
        np.log1p(aggregate.count_mass) / np.log1p(total + 1.0),
        *(aggregate.sum_velocity[index] / (total * velocity_scale_km_s) for index in range(3)),
        *(
            aggregate.sum_velocity_squared[index] / (total * velocity_scale_km_s**2)
            for index in range(3)
        ),
        aggregate.sum_state_age / (total * maximum_age_seconds),
        *(np.full(grid.shape, value, dtype=np.float64) for value in scalar_values),
    ]
    features = np.concatenate(
        (np.stack(dynamic).astype(np.float32), static_geometry_channels(grid)), axis=0
    )
    if features.shape[0] != len(FEATURE_NAMES):
        raise RuntimeError("feature implementation and FEATURE_NAMES disagree")
    return features
