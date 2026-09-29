"""Shared utilities: time parsing, TLE parsing, SGP4 propagation, RTN frames, robust stats, status/log."""
from __future__ import annotations
import json, math, time, logging, zipfile, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import config as C

# ---------------------------------------------------------------- logging / status
def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); lg.addHandler(sh)
        fh = logging.FileHandler(C.LOGS / f"{name}.log"); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg

STATUS_FILE = C.ROOT / "status.json"

def _load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except Exception:
            return {}
    return {}

def set_status(stage: str, state: str, **kw) -> None:
    """state in {running, done, failed, skipped}; kw stored alongside (message, n, elapsed_s...)."""
    s = _load_status()
    s.setdefault("stages", {})[stage] = {"state": state, "updated": utcnow_iso(), **kw}
    s["last_update"] = utcnow_iso()
    STATUS_FILE.write_text(json.dumps(s, indent=1, ensure_ascii=False))

def stage_done(stage: str) -> bool:
    return _load_status().get("stages", {}).get(stage, {}).get("state") == "done"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

class Timer:
    def __enter__(self): self.t0 = time.time(); return self
    def __exit__(self, *a): self.elapsed = time.time() - self.t0

# ---------------------------------------------------------------- time
def to_utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

def parse_utc_series(s: pd.Series) -> pd.Series:
    """Robust UTC parsing for mixed ISO strings (with/without fraction, with/without offset)."""
    return pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce")

def load_maneuvers(name: str) -> pd.DataFrame:
    p = C.DATA / f"sentinel_man_{name}.csv"
    if not p.exists(): return pd.DataFrame(columns=["start", "stop"])
    m = pd.read_csv(p); m["start"] = parse_utc_series(m.start); m["stop"] = parse_utc_series(m.stop)
    return m.dropna().sort_values("start").reset_index(drop=True)

def load_cohort() -> pd.DataFrame:
    p = C.DATA / "objects.csv"
    if p.exists(): return pd.read_csv(p)
    ids = []
    for n in ("pruned", "multi"):
        f = C.DATA / f"norad_{n}_extracted.txt"
        if f.exists(): ids += [(int(x), n) for x in f.read_text().split() if x.strip().isdigit()]
    obj = pd.DataFrame(ids, columns=["norad", "source"]).drop_duplicates("norad")
    lab = pd.read_csv(C.LABELS_CSV).rename(columns={"NORAD_CAT_ID": "norad", "OBJECT_TYPE": "type", "domain": "regime"})
    return obj.merge(lab[["norad", "type", "regime", "status"]], on="norad", how="left")

def tle_epoch(line1: str) -> datetime:
    yy = int(line1[18:20]); doy = float(line1[20:32])
    year = 1900 + yy if yy >= 57 else 2000 + yy
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0)

# ---------------------------------------------------------------- TLE archive access
_ZIP = None
def tle_zip() -> zipfile.ZipFile:
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(C.TLE_ZIP)
    return _ZIP

def clean_line(l: str) -> str:
    l = l.rstrip("\n\r").rstrip()
    if l.endswith("\\"): l = l[:-1].rstrip()
    return l[:69].ljust(69)

def read_tles(norad: int, start=None, end=None, source: str = "raw") -> pd.DataFrame:
    """Return DataFrame[epoch, line1, line2, n, e, inc, bstar] sorted by epoch.
    source='raw' reads the archive zip; 'paf' reads the step6 patch pool when available (else raw)."""
    lines = None
    if source == "paf":
        p = C.STEP6_PATCH_TLE / f"{norad}.txt"
        if p.exists():
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if lines is None:
        fn = f"{norad:05d}.txt"
        z = tle_zip()
        if fn not in z.namelist():
            return pd.DataFrame(columns=["epoch", "line1", "line2", "n", "e", "inc", "bstar"])
        lines = z.read(fn).decode("utf-8", "replace").splitlines()
    rows = []
    i = 0
    while i + 1 < len(lines):
        l1, l2 = lines[i], lines[i + 1]
        if l1.startswith("1 ") and l2.startswith("2 "):
            l1c, l2c = clean_line(l1), clean_line(l2)
            try:
                ep = tle_epoch(l1c)
                n = float(l2c[52:63]); e = float("0." + l2c[26:33].strip()); inc = float(l2c[8:16])
                bs = l1c[53:61]; m = bs[:6].strip(); ex = bs[6:8]
                sign = -1 if m.startswith("-") else 1
                bstar = sign * float("0." + m.lstrip("+-")) * 10 ** int(ex)
            except Exception:
                i += 2; continue
            rows.append((ep, l1c, l2c, n, e, inc, bstar)); i += 2
        else:
            i += 1
    df = pd.DataFrame(rows, columns=["epoch", "line1", "line2", "n", "e", "inc", "bstar"])
    if df.empty: return df
    df["epoch"] = pd.to_datetime(df["epoch"], utc=True)
    df = df.sort_values("epoch").drop_duplicates("epoch", keep="last").reset_index(drop=True)
    if start is not None: df = df[df.epoch >= to_utc(start)]
    if end is not None: df = df[df.epoch <= to_utc(end)]
    return df.reset_index(drop=True)

def elements_from_ne(n_rev_day: np.ndarray, e: np.ndarray):
    """a [km], perigee/apogee height [km] from mean motion and eccentricity."""
    w = np.asarray(n_rev_day, float) * 2 * math.pi / 86400.0
    a = (C.MU_KM3_S2 / w ** 2) ** (1 / 3)
    return a, a * (1 - e) - C.RE_KM, a * (1 + e) - C.RE_KM

# ---------------------------------------------------------------- SGP4
def sgp4_states(line1: str, line2: str, times: Sequence[datetime]) -> np.ndarray:
    """Propagate one TLE to many UTC times. Returns [N,6] km, km/s in TEME; NaN rows on error."""
    from sgp4.api import Satrec, jday
    sat = Satrec.twoline2rv(line1, line2)
    out = np.full((len(times), 6), np.nan)
    jd = np.empty(len(times)); fr = np.empty(len(times))
    for k, t in enumerate(times):
        t = pd.Timestamp(t).to_pydatetime()
        jd[k], fr[k] = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
    err, r, v = sat.sgp4_array(jd, fr)
    ok = err == 0
    out[ok, :3] = r[ok]; out[ok, 3:] = v[ok]
    return out

def propagate_nearest(tles: pd.DataFrame, times: pd.DatetimeIndex, causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-TLE (or previous-TLE if causal) SGP4 propagation. Returns states[N,6] TEME and age_h[N]."""
    ep = tles.epoch.values.astype("datetime64[ns]").astype("int64") / 1e9
    tt = times.values.astype("datetime64[ns]").astype("int64") / 1e9
    states = np.full((len(times), 6), np.nan); age = np.full(len(times), np.nan)
    if len(tles) == 0: return states, age
    if causal:
        idx = np.searchsorted(ep, tt, side="right") - 1
    else:
        pos = np.searchsorted(ep, tt); lo = np.clip(pos - 1, 0, len(ep) - 1); hi = np.clip(pos, 0, len(ep) - 1)
        idx = np.where(np.abs(tt - ep[lo]) <= np.abs(tt - ep[hi]), lo, hi)
    valid = idx >= 0
    for i in np.unique(idx[valid]):
        m = (idx == i) & valid
        r = tles.iloc[int(i)]
        states[m] = sgp4_states(r.line1, r.line2, times[m])
        age[m] = (tt[m] - ep[i]) / 3600.0
    return states, age

def propagate_mtf(tles: pd.DataFrame, times: pd.DatetimeIndex, k: int = 3, tau_h: float = 2.0, window_h: float = 12.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-TLE fusion: up to k TLEs within ±window_h, weights exp(-|age|/tau). Falls back to nearest.
    Returns states[N,6], weighted mean |age|[N], ensemble_size[N]."""
    ep = tles.epoch.values.astype("datetime64[ns]").astype("int64") / 1e9
    tt = times.values.astype("datetime64[ns]").astype("int64") / 1e9
    N = len(times); states = np.zeros((N, 6)); wsum = np.zeros(N); agew = np.zeros(N); ens = np.zeros(N, int)
    if len(tles) == 0: return np.full((N, 6), np.nan), np.full(N, np.nan), ens
    # candidate membership: for each TLE, the times within the window; keep top-k per time by |age|
    age_mat = np.abs(tt[:, None] - ep[None, :]) / 3600.0                     # N x M (M small for one object)
    within = age_mat <= window_h
    order = np.argsort(age_mat, axis=1)[:, :k]
    sel = np.zeros_like(within)
    np.put_along_axis(sel, order, True, axis=1)
    sel &= within
    none = ~sel.any(axis=1)
    if none.any():                                                             # fallback: nearest even if outside window
        sel[none, np.argmin(age_mat[none], axis=1)] = True
    for j in range(len(tles)):
        m = sel[:, j]
        if not m.any(): continue
        st = sgp4_states(tles.iloc[j].line1, tles.iloc[j].line2, times[m])
        ok = np.isfinite(st).all(axis=1)
        w = np.exp(-age_mat[m, j] / tau_h); w[~ok] = 0.0; st[~ok] = 0.0
        states[m] += st * w[:, None]; wsum[m] += w; agew[m] += age_mat[m, j] * w; ens[m] += ok.astype(int)
    good = wsum > 0
    states[good] /= wsum[good, None]; states[~good] = np.nan
    agew[good] /= wsum[good]; agew[~good] = np.nan
    return states, agew, ens

# ---------------------------------------------------------------- frames
def rtn_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows: R, T, N unit vectors for each state. r,v: [N,3]. Returns [N,3,3] with basis vectors as rows."""
    R = r / np.linalg.norm(r, axis=1, keepdims=True)
    h = np.cross(r, v); Nn = h / np.linalg.norm(h, axis=1, keepdims=True)
    T = np.cross(Nn, R)
    return np.stack([R, T, Nn], axis=1)

def to_rtn(delta: np.ndarray, r_ref: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
    """Project position differences [N,3] onto the RTN frame of the reference state."""
    B = rtn_basis(r_ref, v_ref)
    return np.einsum("nij,nj->ni", B, delta)

def teme_to_gcrs(states_km: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    """TEME -> GCRS position/velocity using astropy. states [N,6] km, km/s."""
    from astropy import units as u
    from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
    from astropy.time import Time
    ok = np.isfinite(states_km).all(axis=1)
    out = np.full_like(states_km, np.nan)
    if not ok.any(): return out
    t = Time(times[ok].tz_convert(None).values, scale="utc")
    r = states_km[ok, :3]; v = states_km[ok, 3:]
    cart = CartesianRepresentation(r[:, 0] * u.km, r[:, 1] * u.km, r[:, 2] * u.km,
                                   differentials=CartesianDifferential(v[:, 0] * u.km / u.s, v[:, 1] * u.km / u.s, v[:, 2] * u.km / u.s))
    g = TEME(cart, obstime=t).transform_to(GCRS(obstime=t)).cartesian
    d = g.differentials["s"]
    out[ok, 0] = g.x.to_value(u.km); out[ok, 1] = g.y.to_value(u.km); out[ok, 2] = g.z.to_value(u.km)
    out[ok, 3] = d.d_x.to_value(u.km / u.s); out[ok, 4] = d.d_y.to_value(u.km / u.s); out[ok, 5] = d.d_z.to_value(u.km / u.s)
    return out

def itrs_to_gcrs(states_km: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    """ITRS (ECEF) -> GCRS for POD states [N,6] km, km/s."""
    from astropy import units as u
    from astropy.coordinates import ITRS, GCRS, CartesianRepresentation, CartesianDifferential
    from astropy.time import Time
    ok = np.isfinite(states_km).all(axis=1)
    out = np.full_like(states_km, np.nan)
    if not ok.any(): return out
    t = Time(times[ok].tz_convert(None).values, scale="utc")
    r = states_km[ok, :3]; v = states_km[ok, 3:]
    cart = CartesianRepresentation(r[:, 0] * u.km, r[:, 1] * u.km, r[:, 2] * u.km,
                                   differentials=CartesianDifferential(v[:, 0] * u.km / u.s, v[:, 1] * u.km / u.s, v[:, 2] * u.km / u.s))
    g = ITRS(cart, obstime=t).transform_to(GCRS(obstime=t)).cartesian
    d = g.differentials["s"]
    out[ok, 0] = g.x.to_value(u.km); out[ok, 1] = g.y.to_value(u.km); out[ok, 2] = g.z.to_value(u.km)
    out[ok, 3] = d.d_x.to_value(u.km / u.s); out[ok, 4] = d.d_y.to_value(u.km / u.s); out[ok, 5] = d.d_z.to_value(u.km / u.s)
    return out

# ---------------------------------------------------------------- robust stats
def mad(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return float("nan")
    m = np.median(x); return float(np.median(np.abs(x - m)))

def bootstrap_ci(values: np.ndarray, stat=np.median, n: int = 1000, alpha: float = 0.05, seed: int = C.SEED):
    rng = np.random.default_rng(seed); x = np.asarray(values, float); x = x[np.isfinite(x)]
    if x.size == 0: return (float("nan"), float("nan"), float("nan"))
    bs = np.array([stat(rng.choice(x, x.size, replace=True)) for _ in range(n)])
    return float(stat(x)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))

def quantiles(x: np.ndarray, qs=(0.5, 0.9, 0.95)) -> dict:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {f"p{int(q*100)}": (float(np.quantile(x, q)) if x.size else float("nan")) for q in qs}

def save_df(df: pd.DataFrame, name: str) -> Path:
    p = C.RESULTS / name; p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False); return p

def save_json(obj, name: str) -> Path:
    p = C.RESULTS / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    return p
