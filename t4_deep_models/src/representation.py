from __future__ import annotations

import numpy as np


def rtn_frame(origin_states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origins = np.asarray(origin_states, dtype=np.float64)
    if origins.shape[-1] != 6:
        raise ValueError(f"origin states must end in six channels, found {origins.shape}")
    positions = origins[..., :3]
    velocities = origins[..., 3:]
    radii = np.linalg.norm(positions, axis=-1)
    speeds = np.linalg.norm(velocities, axis=-1)
    angular_momentum = np.cross(positions, velocities)
    momentum_norm = np.linalg.norm(angular_momentum, axis=-1)
    if np.any(radii < 1.0) or np.any(speeds < 1.0e-3) or np.any(momentum_norm < 1.0):
        raise ValueError("cannot form an RTN frame from a degenerate state")
    radial = positions / radii[..., None]
    normal = angular_momentum / momentum_norm[..., None]
    transverse = np.cross(normal, radial)
    basis = np.stack((radial, transverse, normal), axis=-1)
    return basis, radii, speeds, origins


def encode_states(states: np.ndarray, origin_states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    basis, radii, speeds, origins = rtn_frame(origin_states)
    if values.shape[-1] != 6:
        raise ValueError(f"states must end in six channels, found {values.shape}")
    if values.ndim != origins.ndim + 1:
        raise ValueError("states must have exactly one time dimension beyond origin states")
    local_position = np.einsum(
        "...ti,...ij->...tj", values[..., :3] - origins[..., None, :3], basis
    )
    local_velocity = np.einsum(
        "...ti,...ij->...tj", values[..., 3:] - origins[..., None, 3:], basis
    )
    encoded = np.concatenate(
        (local_position / radii[..., None, None], local_velocity / speeds[..., None, None]),
        axis=-1,
    )
    if not np.isfinite(encoded).all():
        raise FloatingPointError("RTN encoding produced non-finite values")
    return np.asarray(encoded, dtype=np.float32)


def decode_states(encoded: np.ndarray, origin_states: np.ndarray) -> np.ndarray:
    values = np.asarray(encoded, dtype=np.float64)
    basis, radii, speeds, origins = rtn_frame(origin_states)
    if values.shape[-1] != 6 or values.ndim != origins.ndim + 1:
        raise ValueError("encoded states and origins have incompatible shapes")
    local_position = values[..., :3] * radii[..., None, None]
    local_velocity = values[..., 3:] * speeds[..., None, None]
    position = np.einsum("...tj,...ij->...ti", local_position, basis) + origins[..., None, :3]
    velocity = np.einsum("...tj,...ij->...ti", local_velocity, basis) + origins[..., None, 3:]
    decoded = np.concatenate((position, velocity), axis=-1)
    if not np.isfinite(decoded).all():
        raise FloatingPointError("RTN decoding produced non-finite values")
    return np.asarray(decoded, dtype=np.float32)


def encode_window(context: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    context_values = np.asarray(context, dtype=np.float32)
    target_values = np.asarray(targets, dtype=np.float32)
    if context_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("one window must contain [time, channel] arrays")
    origin = context_values[-1]
    return encode_states(context_values, origin), encode_states(target_values, origin)
