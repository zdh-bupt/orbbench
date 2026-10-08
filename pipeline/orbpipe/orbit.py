"""Two-body Kepler propagation (vectorised over states) and small helpers."""
from __future__ import annotations
import numpy as np
from . import config as C

def kepler_propagate(r0: np.ndarray, v0: np.ndarray, dt_s: np.ndarray, mu: float = C.MU_KM3_S2) -> tuple[np.ndarray, np.ndarray]:
    """Universal-variable two-body propagation. r0,v0 [N,3] km, km/s; dt_s [N] s. Returns r,v [N,3]."""
    r0 = np.atleast_2d(r0).astype(float); v0 = np.atleast_2d(v0).astype(float); dt = np.atleast_1d(dt_s).astype(float)
    N = len(dt)
    rn = np.linalg.norm(r0, axis=1); vn2 = np.einsum("ij,ij->i", v0, v0)
    alpha = 2.0 / rn - vn2 / mu                                   # 1/a
    vr0 = np.einsum("ij,ij->i", r0, v0) / rn
    sqmu = np.sqrt(mu)
    chi = sqmu * np.abs(alpha) * dt                              # initial guess (elliptic)
    for _ in range(60):
        z = alpha * chi ** 2
        S, Cc = _stumpff(z)
        F = rn * vr0 / sqmu * chi ** 2 * Cc + (1 - alpha * rn) * chi ** 3 * S + rn * chi - sqmu * dt
        dF = rn * vr0 / sqmu * chi * (1 - alpha * chi ** 2 * S) + (1 - alpha * rn) * chi ** 2 * Cc + rn
        step = F / dF; chi = chi - step
        if np.all(np.abs(step) < 1e-9): break
    z = alpha * chi ** 2; S, Cc = _stumpff(z)
    f = 1 - chi ** 2 / rn * Cc; g = dt - chi ** 3 / sqmu * S
    r = f[:, None] * r0 + g[:, None] * v0
    rnew = np.linalg.norm(r, axis=1)
    fdot = sqmu / (rnew * rn) * (alpha * chi ** 3 * S - chi); gdot = 1 - chi ** 2 / rnew * Cc
    v = fdot[:, None] * r0 + gdot[:, None] * v0
    return r, v

def _stumpff(z: np.ndarray):
    S = np.empty_like(z); Cc = np.empty_like(z)
    pos = z > 1e-8; neg = z < -1e-8; zero = ~(pos | neg)
    sz = np.sqrt(np.abs(z[pos])); S[pos] = (sz - np.sin(sz)) / sz ** 3; Cc[pos] = (1 - np.cos(sz)) / z[pos]
    sz = np.sqrt(np.abs(z[neg])); S[neg] = (np.sinh(sz) - sz) / sz ** 3; Cc[neg] = (np.cosh(sz) - 1) / (-z[neg])
    S[zero] = 1 / 6; Cc[zero] = 0.5
    return S, Cc

def classical_elements(r: np.ndarray, v: np.ndarray, mu: float = C.MU_KM3_S2) -> dict:
    r = np.atleast_2d(r); v = np.atleast_2d(v)
    rn = np.linalg.norm(r, axis=1); vn2 = np.einsum("ij,ij->i", v, v)
    h = np.cross(r, v); hn = np.linalg.norm(h, axis=1)
    a = 1.0 / (2.0 / rn - vn2 / mu)
    evec = (np.cross(v, h) / mu) - r / rn[:, None]; e = np.linalg.norm(evec, axis=1)
    inc = np.degrees(np.arccos(np.clip(h[:, 2] / hn, -1, 1)))
    return dict(a_km=a, e=e, inc_deg=inc, hp_km=a * (1 - e) - C.RE_KM, ha_km=a * (1 + e) - C.RE_KM,
                period_min=2 * np.pi * np.sqrt(np.abs(a) ** 3 / mu) / 60.0)
