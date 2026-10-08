"""Per-object / per-group breakdown for the §5.2 comparison table (independent vs unified), plus an 'All' column per block.

T2  Sentinel-1A/2A/3A: F1 and false alarms, residual detector (independent) vs threshold conditioned on T1 phase and T3 Ap (unified).
T4  24-h along-track 95% coverage:
      Sentinel-1A/2A/3A/2B/3B (POD truth, test 2024-25H1): sigma-age (independent) vs sigma-age x g(T2 score) (unified);
      Starlink 3 satellites (MEME truth): sigma-age vs sigma-age x g(T2) from s1 calibration;
      cohort 3 objects (L2 reference, e2e): fixed sigma vs sigma-age x g(T1,T2,T3) -> needs per-object rows, recomputed here.
T6/T7 cohort groups by primary type (PAY / R/B / DEB) and altitude shell (300-800 / 800-1200 / 1200-2000 km):
      approaches found at 20% budget; constructed dangerous approaches in budget / alerted / resolved; unnecessary; dv.
Output: results/detail_table.csv (long format: block, column, metric, independent, unified)
"""
from __future__ import annotations
import sys, os, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd

from . import config as C
from .util import get_logger, set_status, Timer, save_df, mad, to_utc

log = get_logger("detail_table")
RTN = ["r", "t", "n"]

def t2_block(rows):
    t = pd.read_csv(C.RESULTS / "uc_t2_sentinel.csv"); a0 = "SGP4-residual (A0)"; a1 = "SGP4-residual + T1 phase + T3 Ap (A1)"
    for sat in ("S1A", "S2A", "S3A", "ALL"):
        r0 = t[(t.satellite == sat) & (t.variant == a0)].iloc[0]; r1 = t[(t.satellite == sat) & (t.variant == a1)].iloc[0]
        for m, col in (("F1", "f1"), ("False alarms", "fp")):
            rows.append(dict(block="T2 Maneuver", column="All" if sat == "ALL" else f"Sentinel-{sat[1:]}", metric=m, independent=float(r0[col]), unified=float(r1[col])))

def t4_sentinel(rows):
    from .chain_proof import fit_sigma_age, _t2_scores, SCORE_BINS
    from .util import read_tles
    cs = pd.read_parquet(C.RESULTS / "t4_forecast_cases.parquet"); cs = cs[(cs.method == "SGP4-MTF") & (cs.h == 24)].copy(); cs["t0"] = pd.to_datetime(cs.t0, utc=True)
    g = json.loads((C.RESULTS / "c1_g_factors.json").read_text()); t0w = to_utc(C.SENTINEL_TEST_START)
    # T2 scores per satellite (48-h max) for the g(T2) inflation
    score = np.zeros(len(cs))
    for name, (norad, _, _) in C.SENTINELS.items():
        m = (cs.sat == name).values
        if not m.any(): continue
        tl = read_tles(norad, to_utc(C.SENTINEL_START) - pd.Timedelta(days=5), C.WINDOW_END); ep, sc = _t2_scores(tl)
        t0ns = cs.t0.values[m].astype("datetime64[ns]").astype("int64"); out = np.zeros(m.sum())
        for k, t in enumerate(t0ns):
            w = (ep <= t) & (ep >= t - 48 * 3.6e12); out[k] = np.nanmax(sc[w]) if w.any() and np.isfinite(sc[w]).any() else 0.0
        score[m] = out
    cs["t2_bin"] = np.digitize(score, SCORE_BINS[1:-1]); tr = cs[cs.t0 < t0w]; te = cs[cs.t0 >= t0w].copy()
    a, b = fit_sigma_age(tr, "t"); base = g["t"]["base"]; s0 = np.maximum(a + b * (te.age_h + te.h).values, 1e-3) * base
    te["cov_ind"] = np.abs(te.err_t.values) <= 1.96 * s0; te["cov_uni"] = np.abs(te.err_t.values) <= 1.96 * s0 * np.array([g["t"]["bins"][str(int(k))] for k in te.t2_bin.values])
    for sat, d in list(te.groupby("sat")) + [("All", te)]:
        rows.append(dict(block="T4 Forecast", column=("Sentinel-" + sat[1:]) if sat != "All" else "All (Sentinel)", metric="95% coverage, 24 h", independent=float(d.cov_ind.mean()), unified=float(d.cov_uni.mean()), n=len(d)))

def t4_starlink(rows, pick=(44752, 44723, 44751)):
    s1 = pd.read_parquet(C.RESULTS / "s1_starlink_cases.parquet"); d = s1[(s1.method == "A0 raw prev") & (s1.h == 24) & s1.is_test].copy()
    cal = json.loads((C.RESULTS / "s1_starlink_calibration.json").read_text())["A0 raw prev|t"]
    sig = np.maximum(cal["age_a"] + cal["age_b"] * (d.age_h.values + 24.0), 1e-3)
    keys = list(cal["g_t2"].keys()); edges = [float(k.split("-")[0]) for k in keys] + [1e12]
    gt = np.array([cal["g_t2"][keys[max(0, np.searchsorted(edges, s, side="right") - 1)]] for s in np.nan_to_num(d.t2_score.values, nan=0.0)])
    d["cov_ind"] = np.abs(d.err_t.values) <= 1.96 * sig; d["cov_uni"] = np.abs(d.err_t.values) <= 1.96 * sig * gt
    for n in pick:
        dd = d[d.norad == n]; rows.append(dict(block="T4 Forecast", column=f"Starlink {n}", metric="95% coverage, 24 h", independent=float(dd.cov_ind.mean()), unified=float(dd.cov_uni.mean()), n=len(dd)))
    rows.append(dict(block="T4 Forecast", column="All (Starlink)", metric="95% coverage, 24 h", independent=float(d.cov_ind.mean()), unified=float(d.cov_uni.mean()), n=len(d)))

def cohort_blocks(rows):
    """Per-object T4 coverage (3 representative objects + all) and T6/T7 groups from the e2e run."""
    from . import e2e as E
    from . import t6_warning as W
    traj, ids, grid = E.load_cohort(); tles = E.load_tles(ids, grid[0] - pd.Timedelta(days=10), grid[-1]); inp = E.task_inputs(ids, grid)
    obj = pd.read_csv(C.DATA / "objects.csv").set_index("norad"); N = len(ids)
    first = np.argmax(np.isfinite(traj).all(2), axis=1); s0 = traj[np.arange(N), first, :]; alt0 = np.linalg.norm(s0[:, :3], axis=1) - C.RE_KM
    typ = obj.reindex(ids).type.fillna("UNK").to_numpy(); shellg = pd.cut(alt0, [300, 800, 1200, 2000], labels=["300-800 km", "800-1200 km", "1200-2000 km"]).astype(str)
    origins = pd.date_range(grid[0], grid[-1] - pd.Timedelta(hours=E.HZ_H), freq=f"{E.ORIGIN_STEP_H}h", tz="UTC")
    # ---- T4 per-object coverage at 24 h, independent (fixed sigma) vs unified (sigma-age x g)
    cov = {"ind": np.zeros((N, len(origins)), bool), "uni": np.zeros((N, len(origins)), bool), "ok": np.zeros((N, len(origins)), bool)}
    for oi, t0 in enumerate(origins):
        i0 = int(np.searchsorted(grid.values, np.datetime64(t0.tz_convert(None)))); i1 = min(i0 + int(E.HZ_H * 60) + 1, len(grid)); targets = grid[i0:i1]; h_arr = (targets - t0).total_seconds().values / 3600; j = len(targets) - 1
        for k, n in enumerate(ids):
            if int(n) not in tles: continue
            R = traj[k, i0 + j]
            if not np.isfinite(R).all(): continue
            for mode, cfg in (("ind", E.CONFIGS[1]), ("uni", E.CONFIGS[2])):
                res = E.forecast_one(tles[int(n)], t0, targets, cfg["chained"], inp, int(n), cfg)
                if res is None: continue
                st, age, infl, ph, mn, nm = res; sig = E.sigma_for(inp, age, h_arr, infl, fixed=not cfg["chained"], storm=1.0)[j]
                from .util import to_rtn
                e = to_rtn((st[j, :3] - R[:3])[None], R[None, :3], R[None, 3:])[0]; cov[mode][k, oi] = abs(e[1]) <= 1.96 * sig[1]; cov["ok"][k, oi] = True
        log.info(f"T4 per-object coverage: origin {oi+1}/{len(origins)}")
    ok = cov["ok"]; ci = np.where(ok, cov["ind"], np.nan); cu = np.where(ok, cov["uni"], np.nan)
    per = pd.DataFrame(dict(norad=ids, type=typ, shell=shellg, cov_ind=np.nanmean(ci, 1), cov_uni=np.nanmean(cu, 1), n=ok.sum(1)))
    per.to_csv(C.RESULTS / "detail_t4_cohort_per_object.csv", index=False)
    # representative objects: one PAY, one R/B, one DEB with full origins, median-ish coverage difference
    for ty, label in (("PAY", "payload"), ("R/B", "rocket body"), ("DEB", "debris")):
        cand = per[(per.type == ty) & (per.n == len(origins))].copy()
        if len(cand) == 0: continue
        cand["d"] = (cand.cov_uni - cand.cov_ind); pickrow = cand.iloc[(cand.d - cand.d.median()).abs().argsort().iloc[0]]
        rows.append(dict(block="T4 Forecast", column=f"{label} {int(pickrow.norad)}", metric="95% coverage, 24 h", independent=float(pickrow.cov_ind), unified=float(pickrow.cov_uni), n=int(pickrow.n)))
    rows.append(dict(block="T4 Forecast", column="All (cohort)", metric="95% coverage, 24 h", independent=float(np.nanmean(ci)), unified=float(np.nanmean(cu)), n=int(ok.sum())))
    # ---- T6/T7 groups from e2e events and injected events
    ev = pd.read_parquet(C.RESULTS / "e2e_events.parquet"); inj = pd.read_parquet(C.RESULTS / "e2e_injected.parquet")
    pos = {int(n): k for k, n in enumerate(ids)}; A0 = "A0 independent (random 20% budget)"; A1 = "A1 chained"
    # reference approaches with primary group
    pairs = W.apo_peri_pairs(s0); ref = W.screen(traj, ids, grid, pairs, W.COARSE_KM, E.CLOSE_KM)
    ref["type_i"] = [typ[pos[int(i)]] for i in ref.i]; ref["shell_i"] = [shellg[pos[int(i)]] for i in ref.i]
    def found_share(cfg, mask):
        r = ref[mask]; found = ev[(ev.config == cfg) & ev.ref_close]; keys = set(zip(found.i, found.j)) | set(zip(found.j, found.i))
        return float(np.mean([(i, j) in keys for i, j in zip(r.i, r.j)])) if len(r) else np.nan
    inj["type_i"] = [typ[pos[int(i)]] for i in inj.i]; inj["shell_i"] = [shellg[pos[int(i)]] for i in inj.i]
    groups = [("payload", ref.type_i == "PAY", inj.type_i == "PAY"), ("rocket body", ref.type_i == "R/B", inj.type_i == "R/B"), ("debris", ref.type_i == "DEB", inj.type_i == "DEB")] + \
             [(s, ref.shell_i == s, inj.shell_i == s) for s in ("300-800 km", "800-1200 km", "1200-2000 km")] + [("All", np.ones(len(ref), bool), np.ones(len(inj), bool))]
    for name, mr, mi in groups:
        rows.append(dict(block="T6 Warning", column=name, metric="Found (%)", independent=100 * found_share(A0, mr), unified=100 * found_share(A1, mr), n=int(mr.sum())))
        for cfg, key in ((A0, "independent"), (A1, "unified")):
            d = inj[(inj.config == cfg) & mi & inj.dangerous]; nd = inj[(inj.config == cfg) & mi & ~inj.dangerous]
            vals = dict(**{"In budget (%)": 100 * float(d.in_budget.mean()) if len(d) else np.nan, "Alerted": int(d.alert.sum()), "Resolved": int(d.safe_after.sum()), "Unnecessary": int(nd.alert.sum()) + 0, "dv": float(inj[(inj.config == cfg) & mi].dv_m_s.sum())})
            for m, v in vals.items():
                blk = "T6 Warning" if m in ("In budget (%)", "Alerted") else "T7 Avoidance"
                ex = [r for r in rows if r["block"] == blk and r["column"] == name and r["metric"] == m]
                if ex: ex[0][key] = v
                else: rows.append(dict(block=blk, column=name, metric=m, **{key: v}, n=int(d.shape[0])))

def main():
    set_status("S8l_detail_table", "running")
    try:
        with Timer() as T:
            rows = []; t2_block(rows); log.info("T2 done"); t4_sentinel(rows); log.info("T4 Sentinel done"); t4_starlink(rows); log.info("T4 Starlink done"); cohort_blocks(rows); log.info("cohort done")
            out = pd.DataFrame(rows); save_df(out, "detail_table.csv"); log.info("\n" + out.round(3).to_string(index=False))
        set_status("S8l_detail_table", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("detail_table failed"); set_status("S8l_detail_table", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main()
