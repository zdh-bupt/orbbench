"""Challenge evidence experiments on the whole cohort (truth-free, leave-one-TLE-out reference).

E5  Thinning (Challenge 2.2 "the effect of cleaning is complex"): randomly remove 0/10/30/50/70 % of an object's TLEs
    (5 seeds) and reconstruct each held-out TLE from the remaining ones with nearest and MTF. Report the LOO residual
    per removal fraction and per type×regime, and how often the best method flips between fractions.
E6  Propagation sweep (Challenge 1.3 "no unified propagation method"): MTF with window {6,12,24} h × τ {1,2,4} h ×
    k {1,2,3,5} on every stratified object; report the best (window, τ, k) per type×regime and the loss incurred by
    forcing one global setting.
E7  Error census (Challenge 2.1 "errors are numerous"): per type×regime fraction of TLEs hit by each PAF-lite rule
    (same-epoch duplicate, half-period revision, B*<0, isolated n jump) on the full cohort archive 2006→2025,
    plus the LOO residual of the removed vs kept TLEs (from e3_loo_samples).
E8  Truth availability (Challenge 2.3): fraction of cohort objects and TLE epochs with any independent reference
    (POD or operator ephemeris) inside the window.
Objects: stratified sample, up to N_PER_CELL per type×regime cell, holdouts per object = N_HOLD, window = W ± 2 d.
Outputs: results/e5_thinning.csv, e5_thinning_flips.csv, e6_sweep.csv, e6_sweep_best.csv, e7_error_census.csv,
         e7_removed_vs_kept.csv, e8_truth_availability.csv
"""
from __future__ import annotations
import sys, math
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

from . import config as C
from .util import (get_logger, set_status, stage_done, Timer, save_df, to_utc, read_tles, sgp4_states, to_rtn, quantiles, mad)

log = get_logger("challenge_evidence")
N_PER_CELL = 40; N_HOLD = 16
FRACS = [0.0, 0.1, 0.3, 0.5, 0.7]; SEEDS = [0, 1, 2, 3, 4]
WINDOWS = [6.0, 12.0, 24.0]; TAUS = [1.0, 2.0, 4.0]; KS = [1, 2, 3, 5]

def _mtf(others: pd.DataFrame, epoch, k, tau, win):
    if len(others) == 0: return None, np.nan
    dt = (others.epoch - epoch).dt.total_seconds().abs().values / 3600.0
    sel = np.where(dt <= win)[0]
    if len(sel) == 0: sel = np.array([int(np.argmin(dt))])
    sel = sel[np.argsort(dt[sel])[:k]]
    S, W = [], []
    for j in sel:
        r = others.iloc[int(j)]; st = sgp4_states(r.line1, r.line2, [epoch])[0]
        if np.isfinite(st).all(): S.append(st); W.append(np.exp(-dt[j] / tau))
    if not S: return None, np.nan
    S = np.array(S); W = np.array(W) / np.sum(W)
    return (W[:, None] * S).sum(0), float(np.sum(W * dt[sel][: len(S)]))

def _residual(st, ref):
    return float(np.linalg.norm(to_rtn((st[:3] - ref[:3])[None], ref[:3][None], ref[3:][None])[0]))

def _one(args):
    norad, t0, t1 = args
    try:
        raw = read_tles(norad, t0 - pd.Timedelta(days=2), t1 + pd.Timedelta(days=2))
        if len(raw) < 8: return norad, None, None
        inwin = raw[(raw.epoch >= t0) & (raw.epoch < t1)]
        if len(inwin) < 4: return norad, None, None
        idx = np.unique(np.linspace(0, len(inwin) - 1, min(N_HOLD, len(inwin))).astype(int))
        ns_all = raw.epoch.values.astype("datetime64[ns]").astype("int64")
        refs = {}
        for i in idx:
            k = inwin.iloc[i]; st = sgp4_states(k.line1, k.line2, [k.epoch])[0]
            if np.isfinite(st).all(): refs[int(np.int64(pd.Timestamp(k.epoch).value))] = (k.epoch, st)
        e5, e6 = [], []
        # E5 thinning
        for frac in FRACS:
            for seed in (SEEDS if frac > 0 else [0]):
                rng = np.random.default_rng(1000 * seed + norad % 1000)
                keep = np.ones(len(raw), bool)
                if frac > 0: keep[rng.random(len(raw)) < frac] = False
                for ek_ns, (ek, ref) in refs.items():
                    m = keep & (ns_all != ek_ns); others = raw[m]
                    if len(others) == 0: continue
                    sn, an = _mtf(others, ek, 1, 2.0, 1e9); sm, am = _mtf(others, ek, 3, 2.0, 12.0)
                    if sn is None or sm is None: continue
                    e5.append(dict(norad=norad, frac=frac, seed=seed, epoch=ek, nearest=_residual(sn, ref), mtf=_residual(sm, ref), age_nearest_h=an, n_left=int(m.sum())))
        # E6 sweep (no thinning)
        for ek_ns, (ek, ref) in refs.items():
            others = raw[ns_all != ek_ns]
            for win in WINDOWS:
                for tau in TAUS:
                    for k in KS:
                        s, a = _mtf(others, ek, k, tau, win)
                        if s is not None: e6.append(dict(norad=norad, epoch=ek, window_h=win, tau_h=tau, k=k, res=_residual(s, ref), age_h=a))
        return norad, (pd.DataFrame(e5) if e5 else None), (pd.DataFrame(e6) if e6 else None)
    except Exception as ex:
        return norad, str(ex), None

def _census_one(norad):
    """Per-rule hit counts on the full archive (2006+) for one object."""
    try:
        t = read_tles(norad, "2006-01-01", None)
        n = len(t)
        if n < 3: return norad, None
        ns = t.epoch.values.astype("datetime64[ns]").astype("int64")
        dt = np.diff(ns) / 1e9; T = 86400.0 / t.n.values
        half = np.zeros(n, bool); half[:-1] = dt < 0.5 * T[1:]
        bneg = t.bstar.values < 0
        nn = t.n.values; dn = np.abs(np.diff(nn)); med = np.median(dn) if len(dn) else 0
        iso = np.zeros(n, bool)
        for i in range(1, n - 1):
            if abs(nn[i] - nn[i - 1]) > 20 * med + 1e-6 and abs(nn[i] - nn[i + 1]) > 20 * med + 1e-6 and abs(nn[i + 1] - nn[i - 1]) < 5 * med + 1e-6: iso[i] = True
        a = (C.MU_KM3_S2 / ((t.n.values * 2 * math.pi / 86400) ** 2)) ** (1 / 3); hp = a * (1 - t.e.values) - C.RE_KM
        return norad, dict(norad=norad, n_tle=n, half_period=int(half.sum()), bstar_neg=int(bneg.sum()), isolated_n=int(iso.sum()), perigee_lt120=int((hp < 120).sum()),
                           any_rule=int((half | bneg | iso).sum()), first_epoch=str(t.epoch.iloc[0].date()), last_epoch=str(t.epoch.iloc[-1].date()),
                           median_gap_h=float(np.median(dt) / 3600) if len(dt) else np.nan)
    except Exception as ex:
        return norad, None

def stratified_sample(obj: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(C.SEED); parts = []
    for (ty, rg), g in obj.groupby(["type", "regime"]):
        parts.append(g.sample(min(N_PER_CELL, len(g)), random_state=int(rng.integers(1e9))))
    return pd.concat(parts)

def main(force=False):
    if stage_done("S8c_challenge_evidence") and not force: log.info("done"); return
    set_status("S8c_challenge_evidence", "running")
    try:
        with Timer() as T:
            obj = pd.read_csv(C.DATA / "objects.csv"); t0, t1 = to_utc(C.WINDOW_START), to_utc(C.WINDOW_END)
            samp = stratified_sample(obj); log.info(f"stratified sample: {len(samp)} objects over {samp.groupby(['type','regime']).ngroups} cells")
            e5, e6 = [], []
            with ProcessPoolExecutor(C.N_WORKERS) as ex:
                futs = [ex.submit(_one, (int(n), t0, t1)) for n in samp.norad]
                for i, fu in enumerate(as_completed(futs)):
                    n, a, b = fu.result()
                    if isinstance(a, pd.DataFrame): e5.append(a)
                    if isinstance(b, pd.DataFrame): e6.append(b)
                    if (i + 1) % 50 == 0: log.info(f"E5/E6 {i+1}/{len(futs)}")
            E5 = pd.concat(e5, ignore_index=True).merge(obj[["norad", "type", "regime", "source"]], on="norad")
            E6 = pd.concat(e6, ignore_index=True).merge(obj[["norad", "type", "regime", "source"]], on="norad")
            E5.to_parquet(C.RESULTS / "e5_thinning_samples.parquet", index=False); E6.to_parquet(C.RESULTS / "e6_sweep_samples.parquet", index=False)
            # ---- E5 summaries
            rows = []
            for (frac, ty, rg), g in E5.groupby(["frac", "type", "regime"]):
                if len(g) < 50: continue
                rows.append(dict(frac=frac, type=ty, regime=rg, n=len(g), n_obj=g.norad.nunique(), nearest_p50=float(g.nearest.median()), mtf_p50=float(g.mtf.median()),
                                 nearest_p90=float(g.nearest.quantile(.9)), mtf_p90=float(g.mtf.quantile(.9)), mtf_win_rate=float((g.mtf < g.nearest).mean()),
                                 age_nearest_med_h=float(g.age_nearest_h.median()), best=("MTF" if g.mtf.median() < g.nearest.median() else "nearest")))
            for frac, g in E5.groupby("frac"):
                rows.append(dict(frac=frac, type="ALL", regime="ALL", n=len(g), n_obj=g.norad.nunique(), nearest_p50=float(g.nearest.median()), mtf_p50=float(g.mtf.median()),
                                 nearest_p90=float(g.nearest.quantile(.9)), mtf_p90=float(g.mtf.quantile(.9)), mtf_win_rate=float((g.mtf < g.nearest).mean()),
                                 age_nearest_med_h=float(g.age_nearest_h.median()), best=("MTF" if g.mtf.median() < g.nearest.median() else "nearest")))
            th = pd.DataFrame(rows); save_df(th, "e5_thinning.csv")
            # per-object flips: does the better method change with the removal fraction?
            per = E5.groupby(["norad", "frac"])[["nearest", "mtf"]].median(); per["best"] = np.where(per.mtf < per.nearest, "MTF", "nearest")
            fl = per.best.unstack("frac").dropna(); flips = (fl.nunique(axis=1) > 1)
            flipdf = pd.DataFrame([dict(n_obj=len(fl), objects_with_flip=int(flips.sum()), flip_rate=float(flips.mean()),
                                        best_at_0=fl[0.0].value_counts().to_dict(), best_at_0p5=fl[0.5].value_counts().to_dict() if 0.5 in fl else None)])
            flipdf = flipdf.join(pd.DataFrame([{f"flip_rate_{ty}|{rg}": float((fl.loc[fl.index.isin(g.norad)].nunique(axis=1) > 1).mean()) for (ty, rg), g in obj.groupby(["type", "regime"]) if fl.index.isin(g.norad).sum() >= 10}]))
            save_df(flipdf, "e5_thinning_flips.csv")
            # ---- E6 summaries
            sw = E6.groupby(["type", "regime", "window_h", "tau_h", "k"]).agg(n=("res", "size"), n_obj=("norad", "nunique"), res_p50=("res", "median"), res_p90=("res", lambda x: x.quantile(.9))).reset_index()
            sw = sw[sw.n >= 50]; save_df(sw, "e6_sweep.csv")
            best = sw.loc[sw.groupby(["type", "regime"]).res_p50.idxmin()].copy()
            glob = E6.groupby(["window_h", "tau_h", "k"]).res.median(); gbest = glob.idxmin()
            best["global_best_setting"] = str(gbest)
            best["res_p50_at_global_setting"] = [float(sw[(sw.type == r.type) & (sw.regime == r.regime) & (sw.window_h == gbest[0]) & (sw.tau_h == gbest[1]) & (sw.k == gbest[2])].res_p50.iloc[0]) if len(sw[(sw.type == r.type) & (sw.regime == r.regime) & (sw.window_h == gbest[0]) & (sw.tau_h == gbest[1]) & (sw.k == gbest[2])]) else np.nan for r in best.itertuples()]
            best["loss_of_global_pct"] = 100 * (best.res_p50_at_global_setting - best.res_p50) / best.res_p50
            save_df(best, "e6_sweep_best.csv")
            # ---- E7 census on the FULL cohort archive
            cen = []
            with ProcessPoolExecutor(C.N_WORKERS) as ex:
                futs = [ex.submit(_census_one, int(n)) for n in obj.norad]
                for i, fu in enumerate(as_completed(futs)):
                    n, r = fu.result()
                    if r: cen.append(r)
                    if (i + 1) % 600 == 0: log.info(f"E7 census {i+1}/{len(futs)}")
            CE = pd.DataFrame(cen).merge(obj[["norad", "type", "regime", "source"]], on="norad")
            CE.to_parquet(C.RESULTS / "e7_census_per_object.parquet", index=False)
            rows = []
            for scope, g in [("ALL", CE)] + [(f"{t}|{r}", g) for (t, r), g in CE.groupby(["type", "regime"])]:
                N = g.n_tle.sum(); rows.append(dict(scope=scope, n_obj=len(g), n_tle=int(N), half_period_pct=100 * g.half_period.sum() / N, bstar_neg_pct=100 * g.bstar_neg.sum() / N,
                                                    isolated_n_pct=100 * g.isolated_n.sum() / N, perigee_lt120_pct=100 * g.perigee_lt120.sum() / N, any_rule_pct=100 * g.any_rule.sum() / N,
                                                    objects_with_bstar_neg_pct=100 * (g.bstar_neg > 0).mean(), objects_with_isolated_pct=100 * (g.isolated_n > 0).mean(), median_gap_h=float(g.median_gap_h.median())))
            save_df(pd.DataFrame(rows), "e7_error_census.csv")
            # removed vs kept from LOO samples (if available)
            p = C.RESULTS / "e3_loo_samples.parquet"
            if p.exists():
                S = pd.read_parquet(p).merge(obj[["norad", "type", "regime"]], on="norad", how="left")
                S = S[S.method == "nearest-raw"]
                rows = []
                for scope, g in [("ALL", S)] + [(f"{t}|{r}", g) for (t, r), g in S.groupby(["type", "regime"]) if len(g) >= 200]:
                    a, b = g[g.k_removed_by_paf], g[~g.k_removed_by_paf]
                    if len(a) < 5: continue
                    rows.append(dict(scope=scope, n_removed=len(a), n_kept=len(b), removed_res3d_p50=float(a.res_3d.median()), kept_res3d_p50=float(b.res_3d.median()),
                                     removed_res3d_p90=float(a.res_3d.quantile(.9)), kept_res3d_p90=float(b.res_3d.quantile(.9)), ratio_p50=float(a.res_3d.median() / max(b.res_3d.median(), 1e-9))))
                save_df(pd.DataFrame(rows), "e7_removed_vs_kept.csv")
            # ---- E8 truth availability
            n_obj = len(obj); n_truth_obj = sum(1 for name in C.SENTINELS if (C.DATA / f"sentinel_pod_{name}.parquet").exists())
            tle_total = int(CE.n_tle.sum())
            e8 = pd.DataFrame([dict(cohort_objects=n_obj, objects_with_pod_truth=n_truth_obj, pct_objects_with_truth=100 * n_truth_obj / (n_obj + n_truth_obj),
                                    starlink_ephemeris_objects=10653, starlink_ephemeris_overlaps_tle_archive=False, tle_epochs_cohort_2006plus=tle_total,
                                    note="POD truth exists only for the Sentinel satellites; Starlink ephemerides start 2026-06, after the TLE archive ends (2025-12-31).")])
            save_df(e8, "e8_truth_availability.csv")
            log.info("E5 ALL:\n" + th[th.type == "ALL"].round(4).to_string(index=False))
            log.info("E5 flips:\n" + flipdf.T.to_string())
            log.info("E6 best per cell:\n" + best[["type", "regime", "window_h", "tau_h", "k", "res_p50", "res_p50_at_global_setting", "loss_of_global_pct"]].round(4).to_string(index=False))
        set_status("S8c_challenge_evidence", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("challenge evidence failed"); set_status("S8c_challenge_evidence", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
