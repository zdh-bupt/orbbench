from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


EARTH_MU_KM3_S2 = 398600.4418
EARTH_EQUATORIAL_RADIUS_KM = 6378.137
NON_ELLIPTIC_PROPAGATOR_VERSION = "scipy_solve_ivp_dop853_rtol1e-11_v1"


@dataclass(frozen=True)
class OrbitSummary:
    radius_km: float
    speed_km_s: float
    semi_major_axis_km: float
    eccentricity: float
    perigee_altitude_km: float
    apogee_altitude_km: float
    bound_elliptic: bool


def osculating_orbit_summary(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    earth_radius_km: float = EARTH_EQUATORIAL_RADIUS_KM,
) -> OrbitSummary:
    position = np.asarray(position_km, dtype=np.float64)
    velocity = np.asarray(velocity_km_s, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("position and velocity must each contain three values")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("state vector must be finite")
    radius = float(np.linalg.norm(position))
    speed = float(np.linalg.norm(velocity))
    if radius <= earth_radius_km * 0.5 or speed <= 0:
        raise ValueError("state vector is not a plausible Earth orbit")
    angular_momentum = np.cross(position, velocity)
    eccentricity_vector = np.cross(velocity, angular_momentum) / mu_km3_s2 - position / radius
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    energy = 0.5 * speed * speed - mu_km3_s2 / radius
    bound = bool(energy < 0 and eccentricity < 1)
    if not bound:
        return OrbitSummary(radius, speed, float("nan"), eccentricity, float("nan"), float("nan"), False)
    semi_major_axis = -mu_km3_s2 / (2.0 * energy)
    perigee = semi_major_axis * (1.0 - eccentricity) - earth_radius_km
    apogee = semi_major_axis * (1.0 + eccentricity) - earth_radius_km
    return OrbitSummary(
        radius_km=radius,
        speed_km_s=speed,
        semi_major_axis_km=float(semi_major_axis),
        eccentricity=eccentricity,
        perigee_altitude_km=float(perigee),
        apogee_altitude_km=float(apogee),
        bound_elliptic=True,
    )


def propagate_kepler(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    delta_seconds: float | np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate bound Cartesian states with a vectorized two-body solution."""

    position = np.asarray(position_km, dtype=np.float64)
    velocity = np.asarray(velocity_km_s, dtype=np.float64)
    if position.shape != velocity.shape or position.shape[-1] != 3:
        raise ValueError("position and velocity must have matching [..., 3] shapes")
    delta = np.asarray(delta_seconds, dtype=np.float64)
    target_shape = position.shape[:-1]
    try:
        delta = np.broadcast_to(delta, target_shape)
    except ValueError as exc:
        raise ValueError("delta_seconds is not broadcastable to the state shape") from exc

    radius = np.linalg.norm(position, axis=-1)
    speed2 = np.sum(velocity * velocity, axis=-1)
    angular_momentum = np.cross(position, velocity)
    h_norm = np.linalg.norm(angular_momentum, axis=-1)
    energy = 0.5 * speed2 - mu_km3_s2 / radius
    semi_major = -mu_km3_s2 / (2.0 * energy)
    eccentricity_vector = (
        np.cross(velocity, angular_momentum) / mu_km3_s2
        - position / radius[..., None]
    )
    eccentricity = np.linalg.norm(eccentricity_vector, axis=-1)
    valid = (
        np.isfinite(position).all(axis=-1)
        & np.isfinite(velocity).all(axis=-1)
        & np.isfinite(delta)
        & (radius > 0)
        & (h_norm > 0)
        & (energy < 0)
        & (semi_major > 0)
        & (eccentricity < 0.999999)
    )

    safe_e = np.where(eccentricity > 1.0e-10, eccentricity, 1.0)
    p_axis = eccentricity_vector / safe_e[..., None]
    circular = eccentricity <= 1.0e-10
    p_axis = np.where(circular[..., None], position / radius[..., None], p_axis)
    h_axis = angular_momentum / h_norm[..., None]
    q_axis = np.cross(h_axis, p_axis)

    sqrt_one_minus_e2 = np.sqrt(np.maximum(1.0 - eccentricity * eccentricity, 1.0e-15))
    x0 = np.sum(position * p_axis, axis=-1)
    y0 = np.sum(position * q_axis, axis=-1)
    cos_e0 = np.clip(x0 / semi_major + eccentricity, -1.0, 1.0)
    sin_e0 = np.clip(y0 / (semi_major * sqrt_one_minus_e2), -1.0, 1.0)
    eccentric_anomaly_0 = np.arctan2(sin_e0, cos_e0)
    mean_anomaly_0 = eccentric_anomaly_0 - eccentricity * np.sin(eccentric_anomaly_0)
    mean_motion = np.sqrt(mu_km3_s2 / semi_major**3)
    mean_anomaly = mean_anomaly_0 + mean_motion * delta
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(10):
        residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
        derivative = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        eccentric_anomaly -= residual / np.maximum(derivative, 1.0e-12)

    cos_e = np.cos(eccentric_anomaly)
    sin_e = np.sin(eccentric_anomaly)
    propagated_position = (
        semi_major[..., None] * (cos_e - eccentricity)[..., None] * p_axis
        + semi_major[..., None] * (sqrt_one_minus_e2 * sin_e)[..., None] * q_axis
    )
    denominator = np.maximum(1.0 - eccentricity * cos_e, 1.0e-12)
    velocity_factor = (mean_motion * semi_major / denominator)[..., None]
    propagated_velocity = velocity_factor * (
        -sin_e[..., None] * p_axis
        + (sqrt_one_minus_e2 * cos_e)[..., None] * q_axis
    )
    propagated_position = np.where(valid[..., None], propagated_position, np.nan)
    propagated_velocity = np.where(valid[..., None], propagated_velocity, np.nan)
    return propagated_position, propagated_velocity


def propagate_two_body_numerical(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    delta_seconds: float | np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    relative_tolerance: float = 1.0e-11,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate finite elliptic or non-elliptic states with SciPy DOP853.

    This is a sparse fallback for states outside the analytic elliptic solver's
    contract. Invalid inputs or failed integrations remain NaN so callers can
    retain their contextual fail-fast behavior.
    """

    position = np.asarray(position_km, dtype=np.float64)
    velocity = np.asarray(velocity_km_s, dtype=np.float64)
    if position.shape != velocity.shape or position.shape[-1] != 3:
        raise ValueError("position and velocity must have matching [..., 3] shapes")
    if mu_km3_s2 <= 0.0 or relative_tolerance <= 0.0:
        raise ValueError("mu_km3_s2 and relative_tolerance must be positive")
    delta = np.asarray(delta_seconds, dtype=np.float64)
    target_shape = position.shape[:-1]
    try:
        delta = np.broadcast_to(delta, target_shape)
    except ValueError as exc:
        raise ValueError("delta_seconds is not broadcastable to the state shape") from exc

    flat_position = position.reshape(-1, 3)
    flat_velocity = velocity.reshape(-1, 3)
    flat_delta = delta.reshape(-1)
    propagated_position = np.full_like(flat_position, np.nan)
    propagated_velocity = np.full_like(flat_velocity, np.nan)
    absolute_tolerance = np.asarray([1.0e-9] * 3 + [1.0e-12] * 3, dtype=np.float64)

    for index, (initial_position, initial_velocity, duration) in enumerate(
        zip(flat_position, flat_velocity, flat_delta)
    ):
        if (
            not np.isfinite(initial_position).all()
            or not np.isfinite(initial_velocity).all()
            or not np.isfinite(duration)
            or np.linalg.norm(initial_position) <= 0.0
        ):
            continue
        if duration == 0.0:
            propagated_position[index] = initial_position
            propagated_velocity[index] = initial_velocity
            continue

        def derivative(_: float, state: np.ndarray) -> np.ndarray:
            radius_vector = state[:3]
            radius = float(np.linalg.norm(radius_vector))
            if not np.isfinite(radius) or radius <= 0.0:
                raise FloatingPointError("two-body integration reached an invalid radius")
            acceleration = -mu_km3_s2 * radius_vector / radius**3
            return np.concatenate((state[3:], acceleration))

        try:
            solution = solve_ivp(
                derivative,
                (0.0, float(duration)),
                np.concatenate((initial_position, initial_velocity)),
                method="DOP853",
                rtol=float(relative_tolerance),
                atol=absolute_tolerance,
                t_eval=[float(duration)],
            )
        except (FloatingPointError, ValueError):
            continue
        if not solution.success or solution.y.shape != (6, 1):
            continue
        final = solution.y[:, -1]
        if not np.isfinite(final).all():
            continue
        propagated_position[index] = final[:3]
        propagated_velocity[index] = final[3:]

    return (
        propagated_position.reshape(position.shape),
        propagated_velocity.reshape(velocity.shape),
    )
