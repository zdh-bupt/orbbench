"""T4 Orbit forecasting with uncertainty (Sentinel truth cohort, POD as ground truth).

Origins every ORIGIN_STEP_H in the window; horizons 0.5..24 h; all methods causal at the origin t0.
Methods
  SGP4-prev          last TLE with epoch <= t0
  SGP4-MTF           causal multi-TLE fusion (<=3 TLEs with epoch in [t0-12h, t0], weights exp(-age/2h))
  Kepler-24h         two-body propagation of the last state of the 24-h causal history (L2-type input)
  Ridge-orbit        SGP4-prev + ridge correction of RTN error from orbit features (trained pre-window on POD)
  Ridge-weather      + causal space-weather features (observed lags at t0)              [independent mode]
  Ridge-weather+T3   + T3 forecast of Ap/F10.7 for t0..t0+h                              [chained mode]
  SGP4-MTF+T2        MTF that drops TLEs older than the last T2-detected maneuver         [chained mode]
  SGP4-MTF+T1+T2     ... and shrinks the fusion window to 6 h when T1 phase != STABLE     [chained mode]
Uncertainty
  σ-fixed            one global RTN σ per component (train residual scale)
  σ-age              σ_c = a_c + b_c·(age+h) fitted robustly pre-window
  σ-OAD              member spread of MTF at target, linearly calibrated to |err| pre-window
Metrics: RTN error P50/P90/P95 per horizon (all sats pooled and per sat), maneuver-affected split,
         coverage 68/95%, NEES, Spearman(σ,|err|).
Outputs: results/t4_forecast_cases.parquet (one row per origin×horizon×method), t4_forecast_metrics.csv, t4_uq_metrics.csv
"""
from __future__ import annotations
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

from . import config as C
from .util import (get_logger, set_status, stage_done, Timer, save_df, to_utc, read_tles, sgp4_states, teme_to_gcrs,
                   to_rtn, load_maneuvers, mad, parse_utc_series)
from .orbit import kepler_propagate

log = get_logger("t4_forecast")
ORIGIN_STEP_H = 12
HZ = C.FORECAST_HORIZONS_H
TRAIN_START = C.SENTINEL_START
RTN = ["r", "t", "n"]

def load_pod(name: str) -> pd.DataFrame:
    """POD truth in TEME (columns x..vz), so that SGP4 outputs need no per-origin frame transform."""
    d = pd.read_parquet(C.DATA / f"sentinel_pod_{name}.parquet"); d["time"] = pd.to_datetime(d.time, utc=True)
    if "x_teme" in d.columns:
        d = d[["time"] + [f"{c}_teme" for c in ("x", "y", "z", "vx", "vy", "vz")] + ["quality"]]
        d.columns = ["time", "x", "y", "z", "vx", "vy", "vz", "quality"]
    return d.set_index("time").sort_index()

def _causal_tles(tles: pd.DataFrame, t0, window_h=12.0, k=3, after=None):
    c = tles[tles.epoch <= t0]
    if after is not None: c = c[c.epoch >= after] if len(c[c.epoch >= after]) else c
    c = c[c.epoch >= t0 - pd.Timedelta(hours=window_h)]
    if len(c) == 0:
        c = tles[tles.epoch <= t0].tail(1)
    return c.tail(k)

def _fuse(cands: pd.DataFrame, t0, targets, tau_h=2.0):
    """MTF fusion of candidate TLEs to target times. Returns states [T,6] TEME, weighted age h, member spread km."""
    if len(cands) == 0: return np.full((len(targets), 6), np.nan), np.nan, np.nan
    S = []; W = []
    for r in cands.itertuples():
        st = sgp4_states(r.line1, r.line2, targets); age = (t0 - r.epoch).total_seconds() / 3600.0
        S.append(st); W.append(np.exp(-age / tau_h))
    S = np.stack(S); W = np.array(W); W = W / W.sum()
    fused = np.einsum("m,mtk->tk", W, S)
    spread = np.sqrt(np.einsum("m,mt->t", W, np.sum((S[:, :, :3] - fused[None, :, :3]) ** 2, axis=2))) if len(cands) > 1 else np.zeros(len(targets))
    age_w = float(np.sum(W * np.array([(t0 - r.epoch).total_seconds() / 3600.0 for r in cands.itertuples()])))
    return fused, age_w, spread

def _one_sat(args):
    name, norad, origins, mode_inputs = args
    """Compute forecasts for one satellite over given origins. Returns DataFrame of cases."""
    pod = load_pod(name)
    tles = read_tles(norad, pd.Timestamp(origins[0]) - pd.Timedelta(days=5), origins[-1] + pd.Timedelta(days=2))
    man = load_maneuvers(name)
    t2flags = mode_inputs.get("t2_flags", {}).get(norad)     # DataFrame epoch/flag (from cohort or sentinel events)
    t1phase = mode_inputs.get("t1_phase", {}).get(norad)     # DataFrame t_start,t_end,label
    rows = []
    for t0 in origins:
        targets = pd.DatetimeIndex([t0 + pd.Timedelta(hours=h) for h in HZ])
        try:
            truth = pod.reindex(targets)[["x", "y", "z", "vx", "vy", "vz"]].to_numpy()
        except Exception:
            continue
        if not np.isfinite(truth).all(): continue
        base = {}
        prev = tles[tles.epoch <= t0].tail(1)
        if len(prev) == 0: continue
        age_prev = (t0 - prev.epoch.iloc[0]).total_seconds() / 3600.0
        base["SGP4-prev"] = (sgp4_states(prev.line1.iloc[0], prev.line2.iloc[0], targets), age_prev, np.zeros(len(HZ)))
        c = _causal_tles(tles, t0); f, a, sp = _fuse(c, t0, targets); base["SGP4-MTF"] = (f, a, sp)
        # Kepler from causal 24-h history end state (history built with MTF causal at t0)
        h_end = _fuse(c, t0, pd.DatetimeIndex([t0]))[0]
        if np.isfinite(h_end).all():
            dt = np.array([h * 3600.0 for h in HZ])
            rk, vk = kepler_propagate(np.repeat(h_end[:, :3], len(HZ), 0), np.repeat(h_end[:, 3:], len(HZ), 0), dt)
            base["Kepler-24h"] = (np.hstack([rk, vk]), a, np.zeros(len(HZ)))
        # chained variants
        after = None
        if t2flags is not None and len(t2flags):
            fl = t2flags[(t2flags.epoch <= t0) & (t2flags.epoch >= t0 - pd.Timedelta(hours=48)) & (t2flags.flag)]
            if len(fl): after = fl.epoch.max()
        c2 = _causal_tles(tles, t0, after=after); f2, a2, sp2 = _fuse(c2, t0, targets); base["SGP4-MTF+T2"] = (f2, a2, sp2)
        win = 12.0
        if t1phase is not None and len(t1phase):
            ph = t1phase[(t1phase.t_start <= t0) & (t1phase.t_end >= t0 - pd.Timedelta(hours=48))]
            if len(ph) and (ph.label != "STABLE_ORBIT").any(): win = 6.0
        c3 = _causal_tles(tles, t0, window_h=win, after=after); f3, a3, sp3 = _fuse(c3, t0, targets); base["SGP4-MTF+T1+T2"] = (f3, a3, sp3)
        man_flag = bool(len(man) and ((man.start <= targets[-1]) & (man.stop >= t0)).any())
        man_recent = bool(len(man) and ((man.start <= t0) & (man.start >= t0 - pd.Timedelta(hours=48))).any())
        for m, (st, age, sp) in base.items():
            if not np.isfinite(st).all(): continue
            err = to_rtn(st[:, :3] - truth[:, :3], truth[:, :3], truth[:, 3:])
            for j, h in enumerate(HZ):
                rows.append(dict(sat=name, norad=norad, t0=t0, h=h, method=m, age_h=age, spread_km=float(sp[j]) if np.ndim(sp) else float(sp),
                                 err_r=err[j, 0], err_t=err[j, 1], err_n=err[j, 2], err_3d=float(np.linalg.norm(err[j])),
                                 man_in_window=man_flag, man_recent=man_recent, bstar=float(prev.bstar.iloc[0]), n=float(prev.n.iloc[0]),
                                 e=float(prev.e.iloc[0]), inc=float(prev.inc.iloc[0])))
    return pd.DataFrame(rows)

def weather_features(cases: pd.DataFrame, chained: bool) -> pd.DataFrame:
    sw = pd.read_csv(C.DATA / "space_weather_daily.csv", parse_dates=["date"]).set_index("date")
    d0 = pd.to_datetime(cases.t0).dt.tz_convert(None).dt.normalize()
    out = pd.DataFrame(index=cases.index)
    for lag in (1, 3, 7):
        idx = d0 - pd.Timedelta(days=lag)
        out[f"f107_lag{lag}"] = sw.f107_obs.reindex(idx).values; out[f"ap_lag{lag}"] = sw.ap_avg.reindex(idx).values; out[f"kp_lag{lag}"] = sw.kp_daily_avg.reindex(idx).values
    if chained:
        fc = pd.read_parquet(C.RESULTS / "t3_weather_forecasts.parquet")
        fc = fc[(fc.method == "HistGB-AR30") & (fc.horizon_d == 1)]
        piv = fc.pivot_table(index="origin_date", columns="variable", values="pred")
        idx = d0 - pd.Timedelta(days=1)
        for v in ("ap_avg", "f107_obs", "kp_daily_avg"):
            out[f"{v}_fc1d"] = piv[v].reindex(idx).values if v in piv else np.nan
    return out.ffill().bfill()

def ridge_correctors(cases: pd.DataFrame) -> pd.DataFrame:
    """Fit RTN error correctors on pre-window SGP4-prev cases; apply in window. Adds methods Ridge-*."""
    base = cases[cases.method == "SGP4-prev"].copy()
    t0w = to_utc(C.SENTINEL_TEST_START)
    tr = base[base.t0 < t0w]; te = base[base.t0 >= t0w]
    wf_ind = weather_features(base, chained=False); wf_ch = weather_features(base, chained=True)
    all_sats = sorted(base.sat.unique())
    def feats(df, wf=None):
        X = pd.DataFrame({"age": df.age_h, "h": df.h, "ah": df.age_h + df.h, "bstar": df.bstar, "n": df.n, "e": df.e, "inc": df.inc,
                          "doy_s": np.sin(2 * np.pi * pd.to_datetime(df.t0).dt.dayofyear / 365.25), "doy_c": np.cos(2 * np.pi * pd.to_datetime(df.t0).dt.dayofyear / 365.25)}, index=df.index)
        for s in all_sats: X[f"sat_{s}"] = (df.sat == s).astype(float)
        if wf is not None: X = pd.concat([X, wf.loc[df.index]], axis=1)
        return X.fillna(0.0)
    outs = []
    for name, wf, kind in (("Ridge-orbit", None, "ridge"), ("Ridge-weather", wf_ind, "ridge"), ("Ridge-weather+T3", wf_ch, "ridge"),
                           ("HistGB-orbit", None, "gb"), ("HistGB-weather", wf_ind, "gb"), ("HistGB-weather+T3", wf_ch, "gb")):
        Xtr, Xte = feats(tr, wf), feats(te, wf)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        corr = te.copy(); corr["method"] = name
        for c in RTN:
            m = (Ridge(alpha=10.0) if kind == "ridge" else HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=C.SEED)).fit(Xtr, tr[f"err_{c}"])
            corr[f"err_{c}"] = te[f"err_{c}"] - m.predict(Xte)
        corr["err_3d"] = np.sqrt(corr.err_r ** 2 + corr.err_t ** 2 + corr.err_n ** 2)
        outs.append(corr)
    return pd.concat(outs, ignore_index=True)

def uq_models(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit σ models on pre-window SGP4-MTF cases, evaluate in-window. Returns (per-case σ table, metrics)."""
    t0w = to_utc(C.SENTINEL_TEST_START)
    base = cases[cases.method == "SGP4-MTF"].copy(); tr = base[base.t0 < t0w]; te = base[base.t0 >= t0w].copy()
    rows = []; sig = {}; calib = {}
    for c in RTN:
        e_tr = tr[f"err_{c}"].values; ah_tr = (tr.age_h + tr.h).values
        s_fixed = 1.4826 * mad(e_tr)
        # σ-age: robust linear fit of |e| ~ a + b·(age+h) via binned MAD
        bins = np.quantile(ah_tr, np.linspace(0, 1, 9)); ctr, sc = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ah_tr >= lo) & (ah_tr <= hi)
            if m.sum() > 20: ctr.append(ah_tr[m].mean()); sc.append(1.4826 * mad(e_tr[m]))
        b, a = np.polyfit(ctr, sc, 1) if len(ctr) > 2 else (0.0, s_fixed)
        # σ-OAD: |e| ~ α + β·spread (only where spread>0), fallback to σ-age
        sp_tr = tr.spread_km.values; m = sp_tr > 0
        if m.sum() > 50:
            bins = np.quantile(sp_tr[m], np.linspace(0, 1, 9)); ctr2, sc2 = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mm = m & (sp_tr >= lo) & (sp_tr <= hi)
                if mm.sum() > 20: ctr2.append(sp_tr[mm].mean()); sc2.append(1.4826 * mad(e_tr[mm]))
            bo, ao = np.polyfit(ctr2, sc2, 1) if len(ctr2) > 2 else (0.0, s_fixed)
        else:
            bo, ao = 0.0, s_fixed
        calib[c] = dict(fixed=float(s_fixed), age_a=float(a), age_b=float(b), oad_a=float(ao), oad_b=float(bo))
        ah_te = (te.age_h + te.h).values; sp_te = te.spread_km.values
        sig[f"fixed_{c}"] = np.full(len(te), max(s_fixed, 1e-3))
        sig[f"age_{c}"] = np.maximum(a + b * ah_te, 1e-3)
        sig[f"oad_{c}"] = np.where(sp_te > 0, np.maximum(ao + bo * sp_te, 1e-3), sig[f"age_{c}"])
    for k, v in sig.items(): te[f"sigma_{k}"] = v
    calib.update({})
    for model in ("fixed", "age", "oad"):
        for c in RTN:
            e = te[f"err_{c}"].values; s = te[f"sigma_{model}_{c}"].values
            z = e / s
            rows.append(dict(task="T4-UQ", model=f"σ-{model}", component=c, n=len(e), cov68=float(np.mean(np.abs(z) <= 1)), cov95=float(np.mean(np.abs(z) <= 1.96)),
                             nees=float(np.mean(z ** 2)), spearman_sigma_abserr=float(spearmanr(s, np.abs(e)).statistic) if np.std(s) > 0 else np.nan,
                             sigma_median_km=float(np.median(s))))
        # per horizon 3D
        for h, g in te.groupby("h"):
            z2 = sum((g[f"err_{c}"] / g[f"sigma_{model}_{c}"]) ** 2 for c in RTN)
            rows.append(dict(task="T4-UQ", model=f"σ-{model}", component="3D", h=h, n=len(g), cov95=float(np.mean(z2 <= 7.815)), nees=float(np.mean(z2) / 3)))
    import json; (C.RESULTS / "t4_uq_calibration.json").write_text(json.dumps(calib, indent=1))
    return te, pd.DataFrame(rows)

def metrics(cases: pd.DataFrame) -> pd.DataFrame:
    t0w = to_utc(C.SENTINEL_TEST_START); w = cases[cases.t0 >= t0w].copy()
    sw = pd.read_csv(C.DATA / "space_weather_daily.csv", parse_dates=["date"]).set_index("date").ap_avg
    d0 = pd.to_datetime(w.t0).dt.tz_convert(None).dt.normalize()
    w["ap_max"] = np.maximum(sw.reindex(d0).to_numpy(), sw.reindex(d0 + pd.Timedelta(days=1)).to_numpy())
    w["storm"] = w.ap_max >= 40
    rows = []
    def agg(g, **tag):
        r = dict(task="T4", n=len(g), **tag)
        for c in ("3d", "r", "t", "n"):
            x = np.abs(g[f"err_{c}"].values) if c != "3d" else g.err_3d.values
            for q in (50, 90, 95): r[f"{c}_p{q}_km"] = float(np.percentile(x, q))
        return r
    for (m, h), g in w.groupby(["method", "h"]): rows.append(agg(g, method=m, h=h, split="all"))
    for (m, h, s), g in w.groupby(["method", "h", "sat"]): rows.append(agg(g, method=m, h=h, split=s))
    for (m, h, mf), g in w.groupby(["method", "h", "man_in_window"]): rows.append(agg(g, method=m, h=h, split="maneuver" if mf else "quiet"))
    for (m, h, mf), g in w.groupby(["method", "h", "man_recent"]): rows.append(agg(g, method=m, h=h, split="post-maneuver" if mf else "no-recent-maneuver"))
    for (m, h, y), g in w.groupby(["method", "h", pd.to_datetime(w.t0).dt.year]): rows.append(agg(g, method=m, h=h, split=f"year{y}"))
    for (m, h, st), g in w.groupby(["method", "h", "storm"]): rows.append(agg(g, method=m, h=h, split="storm(Ap>=40)" if st else "geomag-quiet"))
    return pd.DataFrame(rows)

def import_reported() -> pd.DataFrame:
    try:
        r = pd.read_csv(C.REPORTED["forecast_8models"]); r = r[r.Metric == "Position (m)"]
        out = []
        for _, row in r.iterrows():
            for col in r.columns[2:]:
                h, reg = col.split(" / "); v = float(str(row[col]).replace(",", ""))
                out.append(dict(task="T4-reported", method=row.Model, h=float(h.replace(" h", "")), regime=reg, pos_mean_km=v / 1000.0,
                                note="deep models on pseudo-truth (MTF/nearest trajectories), 2025-H2 test, 120 sats/regime"))
        return pd.DataFrame(out)
    except Exception as ex:
        log.warning(f"reported import failed: {ex}"); return pd.DataFrame()

def main(force=False):
    if stage_done("S4_t4_forecast") and not force: log.info("done"); return
    set_status("S4_t4_forecast", "running")
    try:
        with Timer() as T:
            origins = pd.date_range(to_utc(TRAIN_START), to_utc(C.WINDOW_END) - pd.Timedelta(hours=24), freq=f"{ORIGIN_STEP_H}h", tz="UTC")
            # chained inputs: T2 sentinel event detections (best unsupervised: SGP4-residual-T) and T1 phase labels
            t2 = {}; t1 = {}
            for name, (norad, _, _) in C.SENTINELS.items():
                p = C.RESULTS / f"t2_maneuver_events_{name}.csv"
                if p.exists():
                    ev = pd.read_csv(p); ev = ev[ev.method == "SGP4-residual-T"]; ev["epoch"] = parse_utc_series(ev.epoch)
                    t2[norad] = ev[["epoch", "pred"]].rename(columns={"pred": "flag"})
            pf = C.RESULTS / "t2_cohort_flags.parquet"
            ph = C.RESULTS / "t1_phase_labels.parquet"
            if ph.exists():
                lab = pd.read_parquet(ph); lab["t_start"] = pd.to_datetime(lab.t_start, utc=True); lab["t_end"] = pd.to_datetime(lab.t_end, utc=True)
                for norad, g in lab.groupby("norad"): t1[int(norad)] = g[["t_start", "t_end", "pred_all"]].rename(columns={"pred_all": "label"})
            # sentinels are not in the cohort phase labels: compute their phases quickly with the same rules
            from .t1_phase import fragments_one
            for name, (norad, _, _) in C.SENTINELS.items():
                fr = fragments_one(norad)
                if fr is not None: t1[norad] = fr[["t_start", "t_end", "silver"]].rename(columns={"silver": "label"})
            inputs = {"t2_flags": t2, "t1_phase": t1}
            tasks = [(name, norad, origins, inputs) for name, (norad, _, _) in C.SENTINELS.items()]
            parts = []
            with ProcessPoolExecutor(min(C.N_WORKERS, len(tasks))) as ex:
                for fu in as_completed([ex.submit(_one_sat, t) for t in tasks]):
                    df = fu.result(); parts.append(df); log.info(f"{df.sat.iloc[0] if len(df) else '?'}: {len(df)} case rows")
            cases = pd.concat(parts, ignore_index=True)
            cases = pd.concat([cases, ridge_correctors(cases)], ignore_index=True)
            uq_cases, uq_met = uq_models(cases)
            cases.to_parquet(C.RESULTS / "t4_forecast_cases.parquet", index=False); uq_cases.to_parquet(C.RESULTS / "t4_uq_cases.parquet", index=False)
            met = metrics(cases); save_df(met, "t4_forecast_metrics.csv"); save_df(uq_met, "t4_uq_metrics.csv")
            rep = import_reported(); save_df(rep, "t4_forecast_reported.csv")
            log.info("in-window 3D P50 km (all sats):\n" + met[met.split == "all"].pivot_table(index="method", columns="h", values="3d_p50_km").round(3).to_string())
            log.info("UQ:\n" + uq_met[uq_met.component != "3D"].pivot_table(index="model", columns="component", values=["cov68", "cov95", "nees"]).round(3).to_string())
        set_status("S4_t4_forecast", "done", elapsed_s=round(T.elapsed), n_cases=int(len(cases)))
    except Exception as ex:
        log.exception("T4 failed"); set_status("S4_t4_forecast", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
