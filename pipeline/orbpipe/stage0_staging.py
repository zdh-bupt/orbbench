"""Stage 0: stage all inputs for the evaluation window into pipeline/data (read-only sources untouched).

Outputs (data/):
  objects.csv                 cohort table: norad, source(pruned|multi), type, regime, status, DISCOS mass/area, path
  traj_window.npz             cohort trajectories at STAGE_CADENCE_MIN over the window: times[T], norad[N], X[N,T,6] TEME km,km/s
  sentinel_pod_<S>.parquet    POD truth in GCRS (1-min, window ± margin) : time, x..vz km, km/s
  sentinel_man_<S>.csv        maneuver intervals: start, stop (UTC)
  space_weather_daily.csv     date, f107_obs, kp_daily_avg, ap_avg (observed section, label swap applied as documented)
"""
from __future__ import annotations
import re, sys, tarfile, io
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, to_utc, itrs_to_gcrs

log = get_logger("stage0")

def _tail_read(path: Path, start: pd.Timestamp, ncols=7, guess_bytes=130_000_000) -> pd.DataFrame:
    """Read only the tail of a huge 1-min CSV and keep rows >= start."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - guess_bytes)); buf = f.read()
    txt = buf.decode("utf-8", "replace")
    txt = txt[txt.find("\n") + 1:] if size > guess_bytes else txt
    df = pd.read_csv(io.StringIO(txt), header=None if size > guess_bytes else 0)
    if size <= guess_bytes:
        df.columns = list(range(df.shape[1]))
    df = df.iloc[:, :ncols]
    df.columns = ["t", "x", "y", "z", "vx", "vy", "vz"][:ncols]
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce", format="mixed")
    df = df[df.t >= start]
    if len(df) and df.t.iloc[0] > start + pd.Timedelta(minutes=2) and size > guess_bytes:
        # tail was too short: read a bigger tail once
        return _tail_read(path, start, ncols, guess_bytes * 4)
    return df

def _stage_one(args):
    norad, source, path, t0, t1, cadence = args
    try:
        p = Path(path)
        if source == "pruned":
            df = _tail_read(p, t0 - pd.Timedelta(minutes=1)); scale = 1e-3          # m -> km
        else:
            # multi csv: timestamp,strategy,frame,x,y,z,vx,vy,vz,... (km, km/s); rows ~230 bytes -> 264k rows ≈ 60 MB
            size = p.stat().st_size; guess = 220_000_000
            with open(p, "rb") as f:
                f.seek(max(0, size - guess)); buf = f.read()
            txt = buf.decode("utf-8", "replace"); txt = txt[txt.find("\n") + 1:]
            raw = pd.read_csv(io.StringIO(txt), header=None, usecols=[0, 3, 4, 5, 6, 7, 8])
            raw.columns = ["t", "x", "y", "z", "vx", "vy", "vz"]
            raw["t"] = pd.to_datetime(raw["t"], utc=True, errors="coerce", format="mixed")
            df = raw[raw.t >= t0 - pd.Timedelta(minutes=1)]; scale = 1.0
        df = df.dropna(subset=["t"]).set_index("t").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        grid = pd.date_range(t0, t1, freq=f"{cadence}min", tz="UTC")
        # the products are 1-min series with a fixed per-object sub-minute offset (0-60 s): snapping to the grid minute
        # would shift positions by up to 225 km along-track, so interpolate position and velocity with cubic Hermite
        out = _hermite_km(df, grid, scale); ok = np.isfinite(out[:, 0])
        return norad, out, float(ok.mean())
    except Exception as ex:
        return norad, None, str(ex)

def _hermite_km(df: pd.DataFrame, grid: pd.DatetimeIndex, scale: float, max_gap_s: float = 90.0) -> np.ndarray:
    """Cubic Hermite interpolation of a 1-min pos/vel series (columns x..vz, scaled to km, km/s) to grid times. NaN outside gaps."""
    ts = df.index.values.astype("datetime64[ns]").astype("int64") / 1e9; tg = grid.values.astype("datetime64[ns]").astype("int64") / 1e9
    P = df[["x", "y", "z"]].to_numpy(np.float64) * scale; V = df[["vx", "vy", "vz"]].to_numpy(np.float64) * scale
    out = np.full((len(grid), 6), np.nan, dtype=np.float32)
    if len(ts) < 2: return out
    i1 = np.searchsorted(ts, tg, side="right"); i0 = i1 - 1; exact = (i1 > 0) & (i1 <= len(ts)) & np.isclose(ts[np.clip(i1 - 1, 0, len(ts) - 1)], tg)
    ok = (i0 >= 0) & (i1 < len(ts)); i0c = np.clip(i0, 0, len(ts) - 1); i1c = np.clip(i1, 0, len(ts) - 1)
    h = ts[i1c] - ts[i0c]; ok &= (h > 0) & (h <= max_gap_s); hh = np.where(h > 0, h, 1.0); s = np.clip((tg - ts[i0c]) / hh, 0, 1)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    pos = h00[:, None] * P[i0c] + (h10 * hh)[:, None] * V[i0c] + h01[:, None] * P[i1c] + (h11 * hh)[:, None] * V[i1c]
    d00 = 6 * s ** 2 - 6 * s; d10 = 3 * s ** 2 - 4 * s + 1; d01 = -6 * s ** 2 + 6 * s; d11 = 3 * s ** 2 - 2 * s
    vel = d00[:, None] * P[i0c] / hh[:, None] + d10[:, None] * V[i0c] + d01[:, None] * P[i1c] / hh[:, None] + d11[:, None] * V[i1c]
    out[ok, :3] = pos[ok]; out[ok, 3:] = vel[ok]
    ex = exact & ~ok; j = np.clip(i1[ex] - 1, 0, len(ts) - 1); out[ex, :3] = P[j]; out[ex, 3:] = V[j]      # grid time equal to a sample
    return out

def build_objects() -> pd.DataFrame:
    rows = []
    for d in sorted(C.PRUNED_DIR.glob("*/")):
        for f in d.glob("*_pruned.csv"):
            m = re.match(r"(\d+)_pruned\.csv", f.name)
            if m: rows.append((int(m.group(1)), "pruned", str(f)))
    for d in sorted(C.MULTI_DIR.glob("*/")):
        for f in d.glob("*multi.csv"):
            m = re.match(r"(\d+)multi\.csv", f.name)
            if m: rows.append((int(m.group(1)), "multi", str(f)))
    obj = pd.DataFrame(rows, columns=["norad", "source", "path"]).drop_duplicates("norad")
    lab = pd.read_csv(C.LABELS_CSV).rename(columns={"NORAD_CAT_ID": "norad", "OBJECT_TYPE": "type", "domain": "regime"})
    obj = obj.merge(lab[["norad", "type", "regime", "status", "launch_ts"]], on="norad", how="left")
    gaps = pd.read_csv(C.GAPS_CSV).rename(columns={"NORAD_CAT_ID": "norad"})[["norad", "median_gap_hours", "n_gaps"]]
    obj = obj.merge(gaps, on="norad", how="left")
    dis = pd.read_csv(C.DISCOS_CSV, low_memory=False)
    dis = dis[["satno", "name", "objectClass", "mass", "xSectAvg", "xSectMax", "span", "cosparId"]].rename(columns={"satno": "norad"})
    dis = dis.dropna(subset=["norad"]).astype({"norad": int}).drop_duplicates("norad")
    obj = obj.merge(dis, on="norad", how="left")
    # hard-body radius proxy from average cross-section, fallback by type
    r = np.sqrt(obj["xSectAvg"] / np.pi)
    fallback = obj["type"].map({"PAY": 2.0, "R/B": 3.0, "DEB": 0.5, "UNK": 1.0}).fillna(1.0)
    obj["hbr_m"] = r.fillna(fallback)
    obj["hbr_from_discos"] = r.notna()
    return obj.sort_values("norad").reset_index(drop=True)

def stage_trajectories(obj: pd.DataFrame) -> None:
    t0, t1 = to_utc(C.WINDOW_START), to_utc(C.WINDOW_END)
    grid = pd.date_range(t0, t1, freq=f"{C.STAGE_CADENCE_MIN}min", tz="UTC")
    tasks = [(int(r.norad), r.source, r.path, t0, t1, C.STAGE_CADENCE_MIN) for r in obj.itertuples()]
    X = np.full((len(tasks), len(grid), 6), np.nan, dtype=np.float32); cov = np.zeros(len(tasks)); errs = {}
    pos = {n: i for i, (n, *_ ) in enumerate(tasks)}
    with ProcessPoolExecutor(C.N_WORKERS) as ex:
        futs = [ex.submit(_stage_one, t) for t in tasks]
        for k, fu in enumerate(as_completed(futs)):
            n, arr, c = fu.result()
            if arr is None: errs[n] = c
            else: X[pos[n]] = arr; cov[pos[n]] = c
            if (k + 1) % 250 == 0: log.info(f"staged {k+1}/{len(tasks)}")
    np.savez_compressed(C.DATA / "traj_window.npz", times=grid.values.astype("datetime64[ns]"),
                        norad=np.array([t[0] for t in tasks]), X=X)
    obj = obj.copy(); obj["window_coverage"] = [cov[pos[int(n)]] for n in obj.norad]
    obj.to_csv(C.DATA / "objects.csv", index=False)
    log.info(f"trajectories staged: {len(tasks)} objects x {len(grid)} steps; errors={len(errs)}; mean coverage={cov.mean():.3f}")
    if errs: (C.DATA / "stage0_errors.json").write_text(pd.Series(errs).to_json(indent=1))

def stage_sentinels() -> None:
    t0 = to_utc(C.SENTINEL_START) - pd.Timedelta(days=10); t1 = to_utc(C.WINDOW_END) + pd.Timedelta(days=1)
    for name, (norad, sub, man) in C.SENTINELS.items():
        files = sorted((C.POD_ROOT / sub).glob("*.csv"))
        parts = []
        for f in files:
            yr = re.findall(r"(20\d\d)", f.name)
            if yr and not (t0.year <= int(yr[0]) <= t1.year): continue
            d = pd.read_csv(f, usecols=["UTC", "X", "Y", "Z", "VX", "VY", "VZ", "Quality"])
            d["UTC"] = pd.to_datetime(d["UTC"], utc=True, errors="coerce", format="mixed")
            d = d[(d.UTC >= t0) & (d.UTC <= t1)]
            parts.append(d)
        if not parts:
            log.warning(f"{name}: no POD in window"); continue
        d = pd.concat(parts).dropna(subset=["UTC"]).sort_values("UTC").drop_duplicates("UTC")
        # POD is 10 s and not on the minute: cubic-Hermite interpolate (pos+vel) to the 1-min grid in ECEF
        d = d.set_index("UTC")
        grid = pd.date_range(d.index.min().ceil("min"), d.index.max().floor("min"), freq="1min", tz="UTC")
        st, ok = hermite_to_grid(d, grid)
        gcrs = itrs_to_gcrs(st, grid[ok]); teme = gcrs_to_teme(gcrs, grid[ok])
        out = pd.DataFrame(gcrs, columns=["x", "y", "z", "vx", "vy", "vz"]); out.insert(0, "time", grid[ok])
        for j, c in enumerate(["x", "y", "z", "vx", "vy", "vz"]): out[f"{c}_teme"] = teme[:, j]
        qi = d.index.get_indexer(grid[ok], method="nearest"); out["quality"] = d["Quality"].to_numpy()[qi]
        out.to_parquet(C.DATA / f"sentinel_pod_{name}.parquet", index=False)
        log.info(f"{name} POD staged: {len(out)} 1-min GCRS+TEME states {out.time.min()} -> {out.time.max()}")
        if man:
            ev, cur = [], None
            for line in (C.MAN_DIR / man).read_text(errors="replace").splitlines()[1:]:
                m = re.match(r"\s*(\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}\.\d+)\s+.*\s(\d)\s*$", line)
                if not m: continue
                t, flag = _parse_man_ts(m.group(1)), int(m.group(2))
                if flag == 1: cur = t
                elif flag == 0 and cur is not None: ev.append((cur, t)); cur = None
            mdf = pd.DataFrame(ev, columns=["start", "stop"])
            mdf.to_csv(C.DATA / f"sentinel_man_{name}.csv", index=False)
            log.info(f"{name} maneuvers parsed: {len(mdf)} events ({(mdf.start.dt.year==2025).sum()} in 2025)")

def hermite_to_grid(d: pd.DataFrame, grid: pd.DatetimeIndex, max_gap_s: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Hermite interpolation of ECEF pos/vel (m, m/s) to grid times. Returns states [G,6] km,km/s and ok mask."""
    ts = d.index.values.astype("datetime64[ns]").astype("int64") / 1e9
    tg = grid.values.astype("datetime64[ns]").astype("int64") / 1e9
    P = d[["X", "Y", "Z"]].to_numpy(np.float64) / 1000.0; V = d[["VX", "VY", "VZ"]].to_numpy(np.float64) / 1000.0
    i1 = np.searchsorted(ts, tg, side="right"); i0 = i1 - 1
    ok = (i0 >= 0) & (i1 < len(ts))
    i0c = np.clip(i0, 0, len(ts) - 1); i1c = np.clip(i1, 0, len(ts) - 1)
    hgap = ts[i1c] - ts[i0c]; ok &= (hgap > 0) & (hgap <= max_gap_s)     # only interpolate inside a short gap
    h = np.where(hgap > 0, hgap, 1.0); s = np.clip((tg - ts[i0c]) / h, 0, 1)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    pos = h00[:, None] * P[i0c] + (h10 * h)[:, None] * V[i0c] + h01[:, None] * P[i1c] + (h11 * h)[:, None] * V[i1c]
    d00 = 6 * s ** 2 - 6 * s; d10 = 3 * s ** 2 - 4 * s + 1; d01 = -6 * s ** 2 + 6 * s; d11 = 3 * s ** 2 - 2 * s
    vel = (d00[:, None] * P[i0c] / h[:, None] + d10[:, None] * V[i0c] + d01[:, None] * P[i1c] / h[:, None] + d11[:, None] * V[i1c])
    return np.hstack([pos, vel])[ok], ok

def gcrs_to_teme(states_km: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    from astropy import units as u
    from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
    from astropy.time import Time
    okm = np.isfinite(states_km).all(axis=1); out = np.full_like(states_km, np.nan)
    if not okm.any(): return out
    t = Time(times[okm].tz_convert(None).values, scale="utc")
    r = states_km[okm, :3]; v = states_km[okm, 3:]
    cart = CartesianRepresentation(r[:, 0] * u.km, r[:, 1] * u.km, r[:, 2] * u.km,
                                   differentials=CartesianDifferential(v[:, 0] * u.km / u.s, v[:, 1] * u.km / u.s, v[:, 2] * u.km / u.s))
    g = GCRS(cart, obstime=t).transform_to(TEME(obstime=t)).cartesian; dd = g.differentials["s"]
    out[okm, 0] = g.x.to_value(u.km); out[okm, 1] = g.y.to_value(u.km); out[okm, 2] = g.z.to_value(u.km)
    out[okm, 3] = dd.d_x.to_value(u.km / u.s); out[okm, 4] = dd.d_y.to_value(u.km / u.s); out[okm, 5] = dd.d_z.to_value(u.km / u.s)
    return out

def _parse_man_ts(s: str) -> pd.Timestamp:
    # '2014/04/05-05:14:44.802'
    d, t = s.split("-", 1)
    return pd.Timestamp(d.replace("/", "-") + " " + t, tz="UTC")

def stage_space_weather() -> None:
    d = pd.read_parquet(C.SW_PARQUET)
    d = d[d.data_section.astype(str).str.lower() == "observed"].copy()
    kp = d[[f"kp_{a:02d}_{b:02d}" for a, b in zip(range(0, 24, 3), range(3, 27, 3))]].mean(axis=1)
    out = pd.DataFrame({"date": pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"),
                        "f107_obs": d["f107_adj"].astype(float),   # documented label swap in source parquet
                        "f107_adj": d["f107_obs"].astype(float),
                        "kp_daily_avg": kp.astype(float), "ap_avg": d["ap_avg"].astype(float),
                        "isn": d.get("isn", np.nan)})
    out = out.dropna(subset=["f107_obs", "kp_daily_avg", "ap_avg"]).sort_values("date")
    out.to_csv(C.DATA / "space_weather_daily.csv", index=False)
    log.info(f"space weather staged: {len(out)} days {out.date.min()} -> {out.date.max()}")

def main(force: bool = False) -> None:
    if stage_done("S0_staging") and not force:
        log.info("S0 already done"); return
    set_status("S0_staging", "running")
    try:
        with Timer() as T:
            stage_space_weather()
            stage_sentinels()
            obj = build_objects()
            log.info(f"objects: {len(obj)} ({obj.source.value_counts().to_dict()}); types {obj.type.value_counts().to_dict()}")
            stage_trajectories(obj)
        set_status("S0_staging", "done", elapsed_s=round(T.elapsed), n_objects=int(len(obj)))
    except Exception as ex:
        log.exception("S0 failed"); set_status("S0_staging", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main(force="--force" in sys.argv)
