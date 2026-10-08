from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _slices(length: int, batch_size: int) -> Iterator[slice]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def ecef_velocity_to_enu(
    velocity_ecef_km_s: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
) -> np.ndarray:
    velocity = np.asarray(velocity_ecef_km_s, dtype=np.float64)
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    if velocity.shape[-1] != 3 or velocity.shape[:-1] != latitude.shape or latitude.shape != longitude.shape:
        raise ValueError("velocity, latitude, and longitude shapes do not agree")
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    east = -sin_lon * velocity[..., 0] + cos_lon * velocity[..., 1]
    north = (
        -sin_lat * cos_lon * velocity[..., 0]
        - sin_lat * sin_lon * velocity[..., 1]
        + cos_lat * velocity[..., 2]
    )
    up = (
        cos_lat * cos_lon * velocity[..., 0]
        + cos_lat * sin_lon * velocity[..., 1]
        + sin_lat * velocity[..., 2]
    )
    return np.stack((east, north, up), axis=-1)


def enu_velocity_to_ecef(
    velocity_enu_km_s: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
) -> np.ndarray:
    """Rotate local ENU velocity components into the Earth-fixed basis."""

    velocity = np.asarray(velocity_enu_km_s, dtype=np.float64)
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    if velocity.shape[-1] != 3 or velocity.shape[:-1] != latitude.shape or latitude.shape != longitude.shape:
        raise ValueError("velocity, latitude, and longitude shapes do not agree")
    if not np.isfinite(velocity).all() or not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("ENU velocity and geodetic angles must be finite")

    east, north, up = (velocity[..., index] for index in range(3))
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    x = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    y = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    z = cos_lat * north + sin_lat * up
    return np.stack((x, y, z), axis=-1)


def itrs_wgs84_to_teme(
    times_ns: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_km: np.ndarray,
    velocity_enu_km_s: np.ndarray,
    *,
    batch_size: int = 50_000,
) -> dict[str, np.ndarray]:
    """Transform WGS84 positions and local ENU velocities into TEME states."""

    from astropy import units as u
    from astropy.coordinates import (
        CartesianDifferential,
        CartesianRepresentation,
        EarthLocation,
        ITRS,
        TEME,
    )
    from astropy.time import Time
    from astropy.utils import iers

    times = np.asarray(times_ns, dtype=np.int64)
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    altitude = np.asarray(altitude_km, dtype=np.float64)
    velocity_enu = np.asarray(velocity_enu_km_s, dtype=np.float64)
    if latitude.shape != longitude.shape or latitude.shape != altitude.shape or latitude.shape != times.shape:
        raise ValueError("times and WGS84 coordinates must have identical [N] shapes")
    if velocity_enu.shape != (len(times), 3):
        raise ValueError("velocity_enu_km_s must have shape [N, 3]")
    if not (
        np.isfinite(latitude).all()
        and np.isfinite(longitude).all()
        and np.isfinite(altitude).all()
        and np.isfinite(velocity_enu).all()
    ):
        raise ValueError("WGS84/ENU states must be finite")

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    iers.conf.iers_degraded_accuracy = "warn"
    result_position = np.empty((len(times), 3), dtype=np.float64)
    result_velocity = np.empty((len(times), 3), dtype=np.float64)
    result_position_itrs = np.empty_like(result_position)
    result_velocity_itrs = np.empty_like(result_velocity)

    for subset in _slices(len(times), batch_size):
        obstime = Time(times[subset] / 1.0e9, format="unix", scale="utc")
        location = EarthLocation.from_geodetic(
            longitude[subset] * u.deg,
            latitude[subset] * u.deg,
            altitude[subset] * u.km,
            ellipsoid="WGS84",
        )
        x, y, z = location.to_geocentric()
        position_itrs = np.stack(
            (x.to_value(u.km), y.to_value(u.km), z.to_value(u.km)), axis=-1
        )
        velocity_itrs = enu_velocity_to_ecef(
            velocity_enu[subset], latitude[subset], longitude[subset]
        )
        representation = CartesianRepresentation(
            position_itrs[:, 0] * u.km,
            position_itrs[:, 1] * u.km,
            position_itrs[:, 2] * u.km,
            differentials=CartesianDifferential(
                velocity_itrs[:, 0] * u.km / u.s,
                velocity_itrs[:, 1] * u.km / u.s,
                velocity_itrs[:, 2] * u.km / u.s,
            ),
        )
        transformed = ITRS(representation, obstime=obstime).transform_to(TEME(obstime=obstime))
        result_position[subset] = transformed.cartesian.xyz.to_value(u.km).T
        result_velocity[subset] = transformed.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s).T
        result_position_itrs[subset] = position_itrs
        result_velocity_itrs[subset] = velocity_itrs

    return {
        "position_teme_km": result_position,
        "velocity_teme_km_s": result_velocity,
        "position_itrs_km": result_position_itrs,
        "velocity_itrs_km_s": result_velocity_itrs,
    }


def teme_to_itrs_wgs84(
    times_ns: np.ndarray,
    position_teme_km: np.ndarray,
    velocity_teme_km_s: np.ndarray,
    *,
    batch_size: int = 50_000,
) -> dict[str, np.ndarray]:
    """Transform TEME Cartesian states to ITRS and WGS84 without network IERS fetches."""

    from astropy import units as u
    from astropy.coordinates import (
        CartesianDifferential,
        CartesianRepresentation,
        EarthLocation,
        ITRS,
        TEME,
    )
    from astropy.time import Time
    from astropy.utils import iers

    times = np.asarray(times_ns, dtype=np.int64)
    position = np.asarray(position_teme_km, dtype=np.float64)
    velocity = np.asarray(velocity_teme_km_s, dtype=np.float64)
    if position.shape != velocity.shape or position.shape != (len(times), 3):
        raise ValueError("times and TEME states must have shapes [N] and [N, 3]")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("TEME states must be finite")

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    iers.conf.iers_degraded_accuracy = "warn"
    result_position = np.empty_like(position)
    result_velocity = np.empty_like(velocity)
    latitude = np.empty(len(times), dtype=np.float64)
    longitude = np.empty(len(times), dtype=np.float64)
    altitude = np.empty(len(times), dtype=np.float64)

    for subset in _slices(len(times), batch_size):
        obstime = Time(times[subset] / 1.0e9, format="unix", scale="utc")
        representation = CartesianRepresentation(
            position[subset, 0] * u.km,
            position[subset, 1] * u.km,
            position[subset, 2] * u.km,
            differentials=CartesianDifferential(
                velocity[subset, 0] * u.km / u.s,
                velocity[subset, 1] * u.km / u.s,
                velocity[subset, 2] * u.km / u.s,
            ),
        )
        transformed = TEME(representation, obstime=obstime).transform_to(ITRS(obstime=obstime))
        xyz = transformed.cartesian.xyz.to_value(u.km).T
        d_xyz = transformed.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s).T
        location = EarthLocation.from_geocentric(xyz[:, 0], xyz[:, 1], xyz[:, 2], unit=u.km)
        lon, lat, height = location.to_geodetic("WGS84")
        result_position[subset] = xyz
        result_velocity[subset] = d_xyz
        latitude[subset] = lat.to_value(u.deg)
        longitude[subset] = lon.to_value(u.deg)
        altitude[subset] = height.to_value(u.km)

    enu = ecef_velocity_to_enu(result_velocity, latitude, longitude)
    return {
        "position_itrs_km": result_position,
        "velocity_itrs_km_s": result_velocity,
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "altitude_km": altitude,
        "velocity_enu_km_s": enu,
    }
