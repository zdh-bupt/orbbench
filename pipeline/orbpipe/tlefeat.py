"""Per-object TLE-derived feature series shared by T1 (phase) and T2 (maneuver).

For consecutive TLE pairs (i-1, i) of one object we compute:
  dt_h, a_km, da_km, dn_rel, de, di_deg, dbstar, and the SGP4 consistency residual: propagate TLE_{i-1}
  to epoch_i and compare with TLE_i's own state at its epoch in the RTN frame of TLE_i -> res_r, res_t, res_n (km).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .util import read_tles, elements_from_ne, sgp4_states, to_rtn

def tle_features(norad: int, start=None, end=None, source: str = "raw", with_residual: bool = True) -> pd.DataFrame:
    t = read_tles(norad, start, end, source)
    if len(t) < 3:
        return pd.DataFrame()
    a, hp, ha = elements_from_ne(t.n.values, t.e.values)
    f = pd.DataFrame({"norad": norad, "epoch": t.epoch, "a_km": a, "hp_km": hp, "ha_km": ha, "n": t.n, "e": t.e, "inc": t.inc, "bstar": t.bstar})
    f["dt_h"] = f.epoch.diff().dt.total_seconds() / 3600.0
    f["da_km"] = f.a_km.diff(); f["dn_rel"] = f.n.diff() / f.n; f["de"] = f.e.diff(); f["di_deg"] = f.inc.diff(); f["dbstar"] = f.bstar.diff()
    f["da_rate_km_d"] = f.da_km / (f.dt_h / 24.0)
    if with_residual:
        res = np.full((len(t), 3), np.nan)
        # propagate each TLE_{i-1} to epoch_i (vectorised per TLE is overkill; loop is fine: ~1-3k TLEs/object/year)
        own = np.vstack([sgp4_states(r.line1, r.line2, [r.epoch]) for r in t.itertuples()])
        for i in range(1, len(t)):
            p = t.iloc[i - 1]
            st = sgp4_states(p.line1, p.line2, [t.epoch.iloc[i]])[0]
            if np.isfinite(st).all() and np.isfinite(own[i]).all():
                res[i] = to_rtn((st[:3] - own[i, :3])[None, :], own[i, :3][None, :], own[i, 3:][None, :])[0]
        f["res_r"], f["res_t"], f["res_n"] = res[:, 0], res[:, 1], res[:, 2]
        f["res_3d"] = np.linalg.norm(res, axis=1)
    return f
