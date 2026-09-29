"""S8: assemble the paper experiments from task outputs.

E0  Benchmark table: best/all methods per task with the headline metric.
E1  Task chaining (Independent vs Chained): T4 methods with/without T1/T2/T3 inputs; T6 with/without T5 and UQ.
E2  Data ablation (mapped onto existing runs): no-maneuver (T4 without T2 flags), no-weather (Ridge-orbit vs Ridge-weather),
    no-precise (σ-fixed vs POD-calibrated σ-age; Ridge correctors need POD), TLE-only (SGP4-prev only).
E3  Trajectory quality vs POD: C0 raw-nearest / C1 PAF-lite nearest / C2 PAF-lite + MTF (non-causal reconstruction on the
    Sentinel truth cohort, 1-min over the window), RTN P50/P90/P95 and coverage loss.
E4  Uncertainty: σ models coverage/NEES/Spearman (from T4) and their effect on T6 risk ranking and T7 avoidance.
Outputs: results/E*.csv and results/experiments_summary.json
"""
from __future__ import annotations
import sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

from . import config as C
from .util import (get_logger, set_status, stage_done, Timer, save_df, save_json, to_utc, read_tles, propagate_nearest, propagate_mtf,
                   to_rtn, load_maneuvers, elements_from_ne, quantiles)

log = get_logger("experiments")
R = C.RESULTS

def _read(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None

def paf_lite(t: pd.DataFrame) -> pd.DataFrame:
    """Light re-implementation of PAF record rules for one object: half-period dedup, B*<0 removal, isolated n jump."""
    if len(t) < 3: return t
    t = t.sort_values("epoch").reset_index(drop=True)
    keep = np.ones(len(t), bool)
    T = 86400.0 / t.n.values; dt = np.diff(t.epoch.values.astype("datetime64[ns]").astype("int64")) / 1e9
    keep[:-1] &= ~(dt < 0.5 * T[1:])                       # drop older of two records closer than half a period
    keep &= t.bstar.values >= 0
    n = t.n.values; dn = np.abs(np.diff(n)); med = np.median(dn) if len(dn) else 0
    for i in range(1, len(n) - 1):
        if abs(n[i] - n[i - 1]) > 20 * med + 1e-6 and abs(n[i] - n[i + 1]) > 20 * med + 1e-6 and abs(n[i + 1] - n[i - 1]) < 5 * med + 1e-6: keep[i] = False
    return t[keep].reset_index(drop=True)

def _quality_one(args):
    name, norad = args
    from .t4_forecast import load_pod
    pod = load_pod(name); t0, t1 = to_utc(C.WINDOW_START), to_utc(C.WINDOW_END)
    pod = pod[(pod.index >= t0) & (pod.index < t1)]
    times = pod.index[::10]                                              # 10-min sampling of the window
    truth = pod.loc[times, ["x", "y", "z", "vx", "vy", "vz"]].to_numpy()
    raw = read_tles(norad, t0 - pd.Timedelta(days=3), t1 + pd.Timedelta(days=3)); paf = paf_lite(raw)
    man = load_maneuvers(name); man_mask = np.zeros(len(times), bool)
    for r in man.itertuples(): man_mask |= (times >= r.start - pd.Timedelta(hours=6)) & (times <= r.stop + pd.Timedelta(hours=6))
    out = []
    for cfg, tl, fn in (("C0 raw + nearest", raw, "nearest"), ("C1 PAF-lite + nearest", paf, "nearest"), ("C2 PAF-lite + MTF", paf, "mtf")):
        if fn == "nearest": st, age = propagate_nearest(tl, times)
        else: st, age, _ = propagate_mtf(tl, times)
        ok = np.isfinite(st).all(1)
        err = np.full((len(times), 3), np.nan); err[ok] = to_rtn(st[ok, :3] - truth[ok, :3], truth[ok, :3], truth[ok, 3:])
        e3 = np.linalg.norm(err, axis=1)
        for split, m in (("all", np.ones(len(times), bool)), ("quiet", ~man_mask), ("maneuver", man_mask)):
            mm = m & ok
            if mm.sum() < 10: continue
            q = quantiles(e3[mm]); out.append(dict(config=cfg, sat=name, split=split, n=int(mm.sum()), n_tle=len(tl), coverage=float(ok[m].mean()),
                                                 age_med_h=float(np.nanmedian(age[mm])), **{f"e3d_{k}_km": v for k, v in q.items()},
                                                 **{f"abs_{c}_p50_km": float(np.nanmedian(np.abs(err[mm, j]))) for j, c in enumerate("rtn")}))
    return pd.DataFrame(out)

def e3_quality() -> pd.DataFrame:
    tasks = [(name, norad) for name, (norad, _, _) in C.SENTINELS.items() if (C.DATA / f"sentinel_pod_{name}.parquet").exists()]
    parts = []
    with ProcessPoolExecutor(min(C.N_WORKERS, len(tasks))) as ex:
        for fu in as_completed([ex.submit(_quality_one, t) for t in tasks]): parts.append(fu.result())
    df = pd.concat(parts, ignore_index=True)
    pooled = df[df.split == "all"].groupby("config").agg(n=("n", "sum"), coverage=("coverage", "mean"), e3d_p50_km=("e3d_p50_km", "median"), e3d_p90_km=("e3d_p90_km", "median"), e3d_p95_km=("e3d_p95_km", "median")).reset_index()
    pooled["sat"] = "ALL"; pooled["split"] = "all"
    return pd.concat([df, pooled], ignore_index=True)

def e1_chaining() -> pd.DataFrame:
    rows = []
    t4 = _read("t4_forecast_metrics.csv")
    if t4 is not None:
        a = t4[t4.split == "all"]
        def get(m, h, col="3d_p50_km"):
            v = a[(a.method == m) & (a.h == h)][col]; return float(v.iloc[0]) if len(v) else np.nan
        for h in (1, 4, 24):
            rows.append(dict(exp="E1", edge="maneuver→forecast (T2→T4)", h=h, independent=get("SGP4-MTF", h), chained=get("SGP4-MTF+T2", h), metric="3D P50 km"))
            rows.append(dict(exp="E1", edge="phase+maneuver→forecast (T1,T2→T4)", h=h, independent=get("SGP4-MTF", h), chained=get("SGP4-MTF+T1+T2", h), metric="3D P50 km"))
            rows.append(dict(exp="E1", edge="weather→forecast (T3→T4)", h=h, independent=get("Ridge-weather", h), chained=get("Ridge-weather+T3", h), metric="3D P50 km"))
            rows.append(dict(exp="E1", edge="maneuver→forecast, P95", h=h, independent=get("SGP4-MTF", h, "3d_p95_km"), chained=get("SGP4-MTF+T2", h, "3d_p95_km"), metric="3D P95 km"))
        b = t4[t4.split.isin(["maneuver", "quiet"])]
        for h in (1, 24):
            for sp in ("maneuver", "quiet"):
                v0 = b[(b.method == "SGP4-MTF") & (b.h == h) & (b.split == sp)]["3d_p50_km"]; v1 = b[(b.method == "SGP4-MTF+T2") & (b.h == h) & (b.split == sp)]["3d_p50_km"]
                rows.append(dict(exp="E1", edge=f"maneuver→forecast [{sp} windows]", h=h, independent=float(v0.iloc[0]) if len(v0) else np.nan, chained=float(v1.iloc[0]) if len(v1) else np.nan, metric="3D P50 km"))
    t6 = _read("t6_warning_metrics.csv")
    if t6 is not None:
        for L in sorted(t6.lead_h.dropna().unique()):
            i = t6[(t6.mode == "Independent") & (t6.lead_h == L)]; c = t6[(t6.mode == "Chained-T5") & (t6.lead_h == L)]
            if len(i) and len(c):
                rows.append(dict(exp="E1", edge="evolution→warning (T5→T6)", h=L, independent=float(i.recall.iloc[0]), chained=float(c.recall.iloc[0]), metric="event recall"))
                rows.append(dict(exp="E1", edge="evolution→warning, pairs screened", h=L, independent=float(i.n_pairs_screened.iloc[0]), chained=float(c.n_pairs_screened.iloc[0]), metric="candidate pairs"))
            u = t6[t6.task == "T6-UQ"]
            for m in ("point", "fixed", "age"):
                v = u[(u.mode == f"UQ-{m}") & (u.lead_h == L)]
                if len(v): rows.append(dict(exp="E1", edge=f"UQ→warning ranking [{m}]", h=L, independent=np.nan, chained=float(v.recall_at_k.iloc[0]), metric="Recall@K dangerous"))
    df = pd.DataFrame(rows)
    if len(df): df["rel_change_pct"] = 100 * (df.chained - df.independent) / df.independent
    return df

def e2_ablation() -> pd.DataFrame:
    t4 = _read("t4_forecast_metrics.csv"); rows = []
    if t4 is None: return pd.DataFrame()
    a = t4[t4.split == "all"]
    def get(m, h, col="3d_p50_km"):
        v = a[(a.method == m) & (a.h == h)][col]; return float(v.iloc[0]) if len(v) else np.nan
    for h in (1, 4, 24):
        rows.append(dict(exp="E2", config="B0 Full (MTF+T2, Ridge-weather+T3, σ-age)", h=h, forecast_p50_km=get("SGP4-MTF+T2", h), corrected_p50_km=get("Ridge-weather+T3", h)))
        rows.append(dict(exp="E2", config="B1 No-maneuver labels", h=h, forecast_p50_km=get("SGP4-MTF", h), corrected_p50_km=get("Ridge-weather+T3", h)))
        rows.append(dict(exp="E2", config="B2 No-weather", h=h, forecast_p50_km=get("SGP4-MTF+T2", h), corrected_p50_km=get("Ridge-orbit", h)))
        rows.append(dict(exp="E2", config="B4 No-precise (no POD-trained corrector)", h=h, forecast_p50_km=get("SGP4-MTF+T2", h), corrected_p50_km=np.nan))
        rows.append(dict(exp="E2", config="B5 TLE-only (SGP4-prev)", h=h, forecast_p50_km=get("SGP4-prev", h), corrected_p50_km=np.nan))
    return pd.DataFrame(rows)

def e4_uq() -> pd.DataFrame:
    uq = _read("t4_uq_metrics.csv"); t6 = _read("t6_warning_metrics.csv"); t7 = _read("t7_avoidance_metrics.csv"); rows = []
    if uq is not None:
        for _, r in uq[uq.component != "3D"].iterrows():
            rows.append(dict(exp="E4", stage="T4 σ calibration", model=r.model, component=r.component, cov68=r.cov68, cov95=r.cov95, nees=r.nees, spearman=r.spearman_sigma_abserr))
    if t6 is not None:
        for _, r in t6[t6.task == "T6-UQ"].iterrows(): rows.append(dict(exp="E4", stage="T6 risk ranking", model=r["mode"], lead_h=r.lead_h, recall_at_k=r.recall_at_k, recall_at_2k=r.recall_at_2k))
    if t7 is not None:
        for _, r in t7[t7.source == "ALL"].iterrows(): rows.append(dict(exp="E4", stage="T7 avoidance", model=r.strategy, safety_rate=r.safety_rate, dv_mean_m_s=r.dv_mean_m_s, pc_post_median=r.pc_post_median))
    return pd.DataFrame(rows)

def e0_benchmark() -> pd.DataFrame:
    rows = []
    t3 = _read("t3_weather_metrics.csv")
    if t3 is not None:
        for _, r in t3[t3.horizon_d == 3].iterrows(): rows.append(dict(task="T3 space weather", subtask=f"{r.variable} +3d", method=r.method, metric="MAE", value=r.mae))
    t1 = _read("t1_phase_metrics.csv")
    if t1 is not None:
        for m, g in t1[t1.stratum.isna()].groupby("method"): rows.append(dict(task="T1 phase", subtask="3-class", method=m, metric="Macro-F1", value=g.macro_f1.mean()))
    t2 = _read("t2_maneuver_metrics.csv")
    if t2 is not None:
        for _, r in t2[t2.satellite == "ALL"].iterrows(): rows.append(dict(task="T2 maneuver", subtask="pooled Sentinels", method=r.method, metric="F2", value=r.f2));
        for _, r in t2[t2.satellite == "ALL"].iterrows(): rows.append(dict(task="T2 maneuver", subtask="pooled Sentinels", method=r.method, metric="Recall", value=r.recall))
    t4 = _read("t4_forecast_metrics.csv")
    if t4 is not None:
        a = t4[t4.split == "all"]
        for _, r in a[a.h.isin([1, 24])].iterrows(): rows.append(dict(task="T4 forecast", subtask=f"+{r.h:g} h vs POD", method=r.method, metric="3D P50 km", value=r["3d_p50_km"]))
    t5 = _read("t5_evolution_metrics.csv")
    if t5 is not None:
        for (m, h), g in t5.groupby(["method", "h"]):
            if h in (1, 24): rows.append(dict(task="T5 evolution", subtask=f"+{h} h density", method=m, metric="overlap %", value=g.overlap_pct.mean()))
    t6 = _read("t6_warning_metrics.csv"); c6 = _read("t6_cdm_metrics.csv")
    if t6 is not None:
        for _, r in t6[t6.task == "T6"].iterrows(): rows.append(dict(task="T6 warning", subtask=f"lead {r.lead_h:g} h, {r['mode']}", method="SGP4-prev screening", metric="recall", value=r.recall))
    if c6 is not None:
        for _, r in c6.iterrows(): rows.append(dict(task="T6 CDM risk", subtask="final log10 Pc", method=r.method, metric="MAE", value=r.mae_log10))
    t7 = _read("t7_avoidance_metrics.csv")
    if t7 is not None:
        for _, r in t7[t7.source == "ALL"].iterrows(): rows.append(dict(task="T7 avoidance", subtask="all events", method=r.strategy, metric="safety rate", value=r.safety_rate));
        for _, r in t7[t7.source == "ALL"].iterrows(): rows.append(dict(task="T7 avoidance", subtask="all events", method=r.strategy, metric="Δv m/s", value=r.dv_mean_m_s))
    return pd.DataFrame(rows)

def main(force=False):
    if stage_done("S8_experiments") and not force: log.info("done"); return
    set_status("S8_experiments", "running")
    try:
        with Timer() as T:
            e0 = e0_benchmark(); save_df(e0, "E0_benchmark_table.csv")
            e1 = e1_chaining(); save_df(e1, "E1_chaining.csv")
            e2 = e2_ablation(); save_df(e2, "E2_data_ablation.csv")
            e3 = e3_quality(); save_df(e3, "E3_trajectory_quality.csv")
            e4 = e4_uq(); save_df(e4, "E4_uncertainty.csv")
            log.info("E1:\n" + (e1.round(3).to_string(index=False) if len(e1) else "empty"))
            log.info("E3 pooled:\n" + e3[e3.sat == "ALL"].round(3).to_string(index=False))
            save_json({"E0_rows": len(e0), "E1_rows": len(e1), "E2_rows": len(e2), "E3_rows": len(e3), "E4_rows": len(e4)}, "experiments_summary.json")
        set_status("S8_experiments", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("S8 failed"); set_status("S8_experiments", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
