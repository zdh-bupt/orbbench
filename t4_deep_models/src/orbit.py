from __future__ import annotations

import math
from typing import Sequence


EARTH_MU_M3_S2 = 3.986004418e14
EARTH_RADIUS_M = 6_378_137.0


def orbital_elements(state: Sequence[float]) -> tuple[float, float]:
    if len(state) != 6:
        raise ValueError("an orbit state must contain six values")
    x, y, z, vx, vy, vz = map(float, state)
    radius = math.sqrt(x * x + y * y + z * z)
    speed2 = vx * vx + vy * vy + vz * vz
    energy = speed2 / 2.0 - EARTH_MU_M3_S2 / radius
    if energy >= 0.0:
        raise ValueError("unbound state is outside this benchmark")
    semi_major_axis = -EARTH_MU_M3_S2 / (2.0 * energy)
    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx
    h2 = hx * hx + hy * hy + hz * hz
    eccentricity2 = 1.0 + 2.0 * energy * h2 / (EARTH_MU_M3_S2**2)
    eccentricity = math.sqrt(max(0.0, eccentricity2))
    altitude_km = (semi_major_axis - EARTH_RADIUS_M) / 1000.0
    return altitude_km, eccentricity


def orbit_group(altitude_km: float, eccentricity: float) -> str | None:
    if eccentricity >= 0.1:
        return "heo"
    if 160.0 <= altitude_km < 2000.0:
        return "leo"
    if altitude_km >= 2000.0:
        return "meo"
    return None
