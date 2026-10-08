"""Chaining proof: each upstream->downstream edge tested where the physics says it should matter, plus an end-to-end
A0 (independent) vs A1 (chained) table.

C1  T2 -> UQ  (maneuver-aware sigma). Sentinel long-window T4 cases (SGP4-MTF). sigma-age fitted 2019-23; inflation
    factor g_man = robust scale of standardised residual in origins with a T2-detected maneuver in the previous 48 h
    vs without (fitted on train). Test 2024-25H1: coverage/NEES overall and in post-maneuver windows for
    sigma-age, sigma-age x g(T2 flag), sigma-age x g(oracle flag).
C2  UQ -> T6  (risk ranking). Cached 700-1000 km shell subset (450 objects, 3 d): reference approaches < 10 km,
    predicted approaches < 20 km from the latest TLE at lead 6/24 h; rank predictions by Pc with
    sigma-age (+T2 inflation) vs fixed sigma vs point miss; recall@K / precision@K for matched (true) approaches and
    Spearman(score, -true miss) among matched.
C3  UQ -> T7  (avoidance efficiency). 117 CDM events with miss < 2 km; decision sigma in {calibrated (CDM), OrbBench
    sigma-age(lead 24 h), fixed sigma, conservative 2 km}; safety judged by TRUE Pc (CDM sigma) < 1e-4 and by 2 km
    distance; delta-v per policy.
C4  T5 -> T6  (screening budget across shells). 600 LEO objects 300-2000 km over 3 d: recall of reference approaches
    vs fraction of candidate pairs screened, ordering pairs by T5 predicted shell density vs random vs oracle.
C5  A0 vs A1 summary table assembled from C1-C4 and P5.
Outputs: results/c1_maneuver_sigma.csv, c2_uq_ranking.csv, c3_avoidance_sigma.csv, c4_budget_curve.csv,
         c4_budget_summary.csv, chain_a0_a1.csv ; figures/fig_c4_budget.pdf, fig_c1_coverage.pdf
"""
from __future__ import annotations
import sys, os, json
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".cache", "orbbench-mpl"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, to_utc, read_tles, sgp4_states, mad
from .proof import t2_flags
from . import t6_warning as W
from .t7_avoidance import simulate, pol_none, pol_max, AMAX, SAFE_M, HBR_M

log = get_logger("chain_proof")
RTN = ["r", "t", "n"]

# ------------------------------------------------------------------ C1
def fit_sigma_age(tr: pd.DataFrame, c: str):
    ah = (tr.age_h + tr.h).values; e = tr[f"err_{c}"].values
    bins = np.quantile(ah, np.linspace(0, 1, 9)); ctr, sc = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (ah >= lo) & (ah <= hi)
        if m.sum() > 20: ctr.append(ah[m].mean()); sc.append(1.4826 * mad(e[m]))
    b, a = np.polyfit(ctr, sc, 1); return a, b

def _t2_scores(tles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-TLE SGP4 consistency score |res_t|/MAD (NaN for first TLE)."""
    if len(tles) < 4: return tles.epoch.values.astype("datetime64[ns]").astype("int64"), np.full(len(tles), np.nan)
    own = np.vstack([sgp4_states(r.line1, r.line2, [r.epoch]) for r in tles.itertuples()])
    from .util import to_rtn
    res = np.full(len(tles), np.nan)
    for i in range(1, len(tles)):
        pr = tles.iloc[i - 1]; st = sgp4_states(pr.line1, pr.line2, [tles.epoch.iloc[i]])[0]
        if np.isfinite(st).all() and np.isfinite(own[i]).all():
            res[i] = to_rtn((st[:3] - own[i, :3])[None], own[i, :3][None], own[i, 3:][None])[0][1]
    return tles.epoch.values.astype("datetime64[ns]").astype("int64"), np.abs(res) / max(mad(res), 1e-9)

SCORE_BINS = [0, 3, 6, 12, 1e9]
def c1_maneuver_sigma() -> pd.DataFrame:
    cs = pd.read_parquet(C.RESULTS / "t4_forecast_cases.parquet"); cs = cs[cs.method == "SGP4-MTF"].copy(); cs["t0"] = pd.to_datetime(cs.t0, utc=True)
    logged = [n for n, v in C.SENTINELS.items() if v[2]]
    cs = cs[cs.sat.isin(logged)].copy()                       # only satellites with maneuver logs -> oracle is meaningful
    score = np.zeros(len(cs))
    for name in logged:
        m = (cs.sat == name).values
        if not m.any(): continue
        norad = C.SENTINELS[name][0]
        tl = read_tles(norad, to_utc(C.SENTINEL_START) - pd.Timedelta(days=5), C.WINDOW_END)
        ep, sc = _t2_scores(tl)
        t0ns = cs.t0.values[m].astype("datetime64[ns]").astype("int64")
        # max T2 score over TLEs in [t0-48h, t0]
        out = np.zeros(m.sum())
        for k, t in enumerate(t0ns):
            w = (ep <= t) & (ep >= t - 48 * 3.6e12)
            out[k] = np.nanmax(sc[w]) if w.any() and np.isfinite(sc[w]).any() else 0.0
        score[m] = out
        log.info(f"C1 {name}: {len(tl)} TLEs; origins with T2 score>=6 in prev 48h: {(out>=6).mean()*100:.1f}% (logged maneuver in prev 48h: {cs.man_recent.values[m].mean()*100:.1f}%)")
    cs["t2_score"] = score; cs["t2_bin"] = np.digitize(score, SCORE_BINS[1:-1]); cs["flag_oracle"] = cs.man_recent.astype(bool)
    t0w = to_utc(C.SENTINEL_TEST_START); tr = cs[cs.t0 < t0w]; te = cs[cs.t0 >= t0w].copy()
    rows = []; gtab = {}
    for c in RTN:
        a, b = fit_sigma_age(tr, c)
        z_tr = tr[f"err_{c}"].values / np.maximum(a + b * (tr.age_h + tr.h).values, 1e-3)
        base = 1.4826 * mad(z_tr[(~tr.flag_oracle.values) & (tr.t2_bin.values == 0)]) or 1.0
        g_bin = {int(k): (1.4826 * mad(z_tr[tr.t2_bin.values == k]) / base if (tr.t2_bin.values == k).sum() > 50 else 1.0) for k in range(len(SCORE_BINS) - 1)}
        g_or = (1.4826 * mad(z_tr[tr.flag_oracle.values]) / base) if tr.flag_oracle.values.sum() > 50 else 1.0
        gtab[c] = dict(bins=g_bin, oracle=g_or, base=base)
        s0 = np.maximum(a + b * (te.age_h + te.h).values, 1e-3) * base; e_te = te[f"err_{c}"].values
        variants = {"sigma-age": s0,
                    "sigma-age x g(T2 score)": s0 * np.array([g_bin[int(k)] for k in te.t2_bin.values]),
                    "sigma-age x g(logged maneuver)": s0 * np.where(te.flag_oracle.values, g_or, 1.0),
                    "sigma-age x g(both)": s0 * np.maximum(np.array([g_bin[int(k)] for k in te.t2_bin.values]), np.where(te.flag_oracle.values, g_or, 1.0))}
        for model, sg in variants.items():
            for split, m in (("all", np.ones(len(te), bool)), ("post-maneuver (logged)", te.flag_oracle.values), ("no recent maneuver", ~te.flag_oracle.values),
                             ("T2 score>=6", te.t2_score.values >= 6), ("T2 score<3", te.t2_score.values < 3)):
                for h in [None, 1.0, 4.0, 24.0]:
                    mm = m & ((te.h.values == h) if h is not None else True)
                    if mm.sum() < 30: continue
                    zz = e_te[mm] / sg[mm]
                    rows.append(dict(component=c, model=model, split=split, h=("all" if h is None else h), n=int(mm.sum()), cov68=float(np.mean(np.abs(zz) <= 1)),
                                     cov95=float(np.mean(np.abs(zz) <= 1.96)), nees=float(np.mean(zz ** 2)), err_p50_km=float(np.median(np.abs(e_te[mm]))), sigma_p50_km=float(np.median(sg[mm]))))
    out = pd.DataFrame(rows); out["g_T2_bins_T"] = json.dumps({str(k): round(v, 2) for k, v in gtab["t"]["bins"].items()}); out["g_oracle_T"] = round(gtab["t"]["oracle"], 3)
    (C.RESULTS / "c1_g_factors.json").write_text(json.dumps(gtab, indent=1))
    return out

# ------------------------------------------------------------------ C2
def c2_uq_ranking() -> pd.DataFrame:
    traj, ids, grid, obj = W.load_shell_subset()
    first = np.argmax(np.isfinite(traj).all(2), axis=1); s0 = traj[np.arange(len(ids)), first, :].astype(float)
    pairs = W.apo_peri_pairs(s0)
    truth = W.screen(traj, ids, grid, pairs, W.COARSE_KM, 10.0); truth["tca"] = pd.to_datetime(truth.tca, utc=True)
    log.info(f"C2: {len(ids)} objects, {len(pairs)} pairs, {len(truth)} reference approaches < 10 km")
    calib = json.loads((C.RESULTS / "t4_uq_calibration.json").read_text())
    g = json.loads((C.RESULTS / "c1_g_factors.json").read_text()) if (C.RESULTS / "c1_g_factors.json").exists() else None
    sig_age = lambda ah: np.sqrt(np.mean([np.maximum(calib[c]["age_a"] + calib[c]["age_b"] * ah, 1e-3) ** 2 for c in RTN]))
    sig_fixed = float(np.sqrt(np.mean([calib[c]["fixed"] ** 2 for c in RTN])))
    flags = pd.read_parquet(C.RESULTS / "t2_cohort_flags.parquet") if (C.RESULTS / "t2_cohort_flags.parquet").exists() else None
    tles = {int(n): read_tles(int(n), grid[0] - pd.Timedelta(days=8), grid[-1]) for n in ids}
    # object-level quality scales (km): LOO median residual of nearest-raw (truth-free), and delivered OAD proxy
    try:
        S = pd.read_parquet(C.RESULTS / "e3_loo_samples.parquet"); S = S[(S.method == "nearest-raw") & (~S.k_removed_by_paf)]
        loo_obj = S.groupby("norad").res_3d.median(); loo_p90 = S.groupby("norad").res_3d.quantile(.9)
    except Exception:
        loo_obj = pd.Series(dtype=float); loo_p90 = pd.Series(dtype=float)
    try:
        px = pd.read_csv(C.MNT / "processed_data/multi_proxy_step6_deliverable/results/per_sat_ok_531.csv").set_index("norad")
        oad_obj = px.overlap_median
    except Exception:
        oad_obj = pd.Series(dtype=float)
    def q_loo(n): v = loo_obj.get(int(n), np.nan); return v if np.isfinite(v) else float(loo_obj.median())
    def q_loo90(n): v = loo_p90.get(int(n), np.nan); return v if np.isfinite(v) else float(loo_p90.median())
    rows = []; allev = []
    for L in (6, 24):
        t_issue = grid[0] - pd.Timedelta(hours=L); pred = np.full_like(traj, np.nan); ages = np.full(len(ids), np.nan); manflag = np.zeros(len(ids), bool)
        for k, n in enumerate(ids):
            tl = tles[int(n)]; prev = tl[tl.epoch <= t_issue]
            if len(prev): pred[k] = sgp4_states(prev.line1.iloc[-1], prev.line2.iloc[-1], grid); ages[k] = (grid[0] - prev.epoch.iloc[-1]).total_seconds() / 3600
            if flags is not None:
                f = flags[(flags.norad == int(n)) & (flags.flag)]
                if len(f):
                    fe = pd.to_datetime(f.epoch, utc=True); manflag[k] = ((fe <= t_issue) & (fe >= t_issue - pd.Timedelta(hours=48))).any()
        det = W.screen(pred, ids, grid, pairs, W.COARSE_KM, 20.0)
        if len(det) == 0: continue
        det["tca"] = pd.to_datetime(det.tca, utc=True); det["truth_miss"] = np.nan
        for di, d in det.iterrows():
            m = truth[(truth.i == d.i) & (truth.j == d.j)]
            if len(m):
                dtm = (m.tca - d.tca).dt.total_seconds().abs() / 60
                if (dtm <= W.TCA_TOL_MIN).any(): det.loc[di, "truth_miss"] = m.loc[dtm.idxmin(), "miss_km"]
        pos = {int(n): k for k, n in enumerate(ids)}
        det["hbr_km"] = [(obj.loc[i, "hbr_m"] + obj.loc[j, "hbr_m"]) / 1000.0 for i, j in zip(det.i, det.j)]
        det["age_i"] = [ages[pos[i]] + (t - grid[0]).total_seconds() / 3600 for i, t in zip(det.i, det.tca)]
        det["age_j"] = [ages[pos[j]] + (t - grid[0]).total_seconds() / 3600 for j, t in zip(det.j, det.tca)]
        det["man_i"] = [manflag[pos[i]] for i in det.i]; det["man_j"] = [manflag[pos[j]] for j in det.j]
        gT = float(np.mean([g[c]["oracle"] for c in RTN])) if g else 1.0     # maneuver inflation from C1 (logged-maneuver factor)
        det["sigma_age"] = np.sqrt(det.age_i.map(sig_age) ** 2 + det.age_j.map(sig_age) ** 2)
        det["sigma_age_man"] = np.sqrt((det.age_i.map(sig_age) * np.where(det.man_i, gT, 1)) ** 2 + (det.age_j.map(sig_age) * np.where(det.man_j, gT, 1)) ** 2)
        det["sigma_fixed"] = np.sqrt(2) * sig_fixed
        # object-level: LOO residual is a consistency scale at age ~ neighbour gap; scale linearly with TLE age (h) relative to 12 h
        det["sigma_obj_loo"] = np.sqrt((det.i.map(q_loo) * np.maximum(det.age_i, 1) / 12.0) ** 2 + (det.j.map(q_loo) * np.maximum(det.age_j, 1) / 12.0) ** 2)
        det["sigma_obj_loo90"] = np.sqrt((det.i.map(q_loo90) * np.maximum(det.age_i, 1) / 12.0) ** 2 + (det.j.map(q_loo90) * np.maximum(det.age_j, 1) / 12.0) ** 2)
        det["score_point"] = -det.miss_km
        det["score_point_over_sigma_obj"] = -det.miss_km / det.sigma_obj_loo          # Mahalanobis-like miss
        for s in ("sigma_age", "sigma_age_man", "sigma_fixed", "sigma_obj_loo", "sigma_obj_loo90"):
            det[f"pc_{s}"] = [W.pc_2d(m, h, sg) for m, h, sg in zip(det.miss_km, det.hbr_km, det[s])]
        det["lead_h"] = L; allev.append(det)
        matched = det.truth_miss.notna().values; danger = (det.truth_miss < 5.0).values
        for name, sc in (("point miss", det.score_point.values), ("Pc sigma-fixed", det.pc_sigma_fixed.values), ("Pc sigma-age", det.pc_sigma_age.values), ("Pc sigma-age x T2", det.pc_sigma_age_man.values),
                         ("Pc sigma-object (LOO p50)", det.pc_sigma_obj_loo.values), ("Pc sigma-object (LOO p90)", det.pc_sigma_obj_loo90.values), ("miss / sigma-object", det.score_point_over_sigma_obj.values)):
            order = np.argsort(-sc)
            for lab, y in (("matched (true approach < 10 km)", matched), ("dangerous (true miss < 5 km)", danger)):
                K = max(int(y.sum()), 1)
                rows.append(dict(lead_h=L, ranking=name, target=lab, n_pred=len(det), n_positive=int(y.sum()), recall_at_k=float(y[order[:K]].mean()),
                                 recall_at_2k=float(y[order[:2 * K]].sum() / max(y.sum(), 1)), precision_at_10=float(y[order[:10]].mean()),
                                 spearman_score_vs_neg_truemiss=float(spearmanr(sc[matched], -det.truth_miss.values[matched]).statistic) if matched.sum() > 5 and np.std(sc[matched]) > 0 else np.nan))
    if allev: pd.concat(allev, ignore_index=True).to_csv(C.RESULTS / "c2_predicted_events.csv", index=False)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ C3
def c3_avoidance_sigma() -> pd.DataFrame:
    c = pd.read_csv(C.CDM_CSV); c = c[c.miss_distance < 2000].copy()
    sig_true = np.sqrt(np.nanmean(np.stack([c[f"t_sigma_{k}"] ** 2 + c[f"c_sigma_{k}"] ** 2 for k in ("r", "t", "n")]), axis=0))
    sig_true = np.where(np.isfinite(sig_true) & (sig_true > 0), sig_true, np.nanmedian(sig_true))
    calib = json.loads((C.RESULTS / "t4_uq_calibration.json").read_text())
    sig_orb = 1000 * np.sqrt(2) * np.sqrt(np.mean([np.maximum(calib[k]["age_a"] + calib[k]["age_b"] * 24.0, 1e-3) ** 2 for k in RTN]))     # m, lead 24 h
    sig_fix = 1000 * np.sqrt(2) * np.sqrt(np.mean([calib[k]["fixed"] ** 2 for k in RTN]))
    med_true = float(np.nanmedian(sig_true))
    variants = {"Pc-gate, calibrated per-event sigma (CDM)": lambda i: sig_true[i],
                "Pc-gate, per-event sigma x0.5 (overconfident)": lambda i: 0.5 * sig_true[i],
                "Pc-gate, per-event sigma x2 (underconfident)": lambda i: 2.0 * sig_true[i],
                "Pc-gate, per-event sigma x4": lambda i: 4.0 * sig_true[i],
                "Pc-gate, one fixed sigma (median of events)": lambda i: med_true,
                "Pc-gate, OrbBench TLE sigma-age (24 h, not event-specific)": lambda i: sig_orb,
                "Pc-gate, conservative 2 km sigma": lambda i: 2000.0}
    rows = []
    for name, sfun in [("No-maneuver", None), ("Distance-max (2 km)", None)] + list(variants.items()):
        res = []
        for i, r in enumerate(c.itertuples()):
            rel_r = np.array([r.relative_position_r, r.relative_position_t, r.relative_position_n], float); rel_v = np.array([r.relative_velocity_r, r.relative_velocity_t, r.relative_velocity_n], float)
            st = sig_true[i]
            if name == "No-maneuver": pol = pol_none
            elif name.startswith("Distance"): pol = pol_max
            else:
                sd = sfun(i)
                def pol(d, k, dir0, s, vn=None, sd=sd): return dir0 * AMAX if W.pc_2d(d / 1000, HBR_M / 1000, max(sd, 1.0) / 1000) >= 1e-4 else np.zeros(3)
            out = simulate(rel_r, rel_v, pol, st)          # pc_post computed with TRUE sigma
            res.append(dict(dv=out["dv_m_s"], safe_dist=out["safe"], safe_pc=out["pc_post"] < 1e-4, pc_post=out["pc_post"], min_d=out["min_dist_m"], pc_pre=W.pc_2d(r.miss_distance / 1000, HBR_M / 1000, max(st, 1.0) / 1000)))
        R = pd.DataFrame(res)
        rows.append(dict(policy=name, n_events=len(R), pre_pc_ge_1e4_frac=float((R.pc_pre >= 1e-4).mean()), safety_rate_pc=float(R.safe_pc.mean()), safety_rate_2km=float(R.safe_dist.mean()),
                         dv_mean_m_s=float(R.dv.mean()), dv_p90_m_s=float(R.dv.quantile(.9)), min_dist_p50_m=float(R.min_d.median()), pc_post_p90=float(R.pc_post.quantile(.9)),
                         decision_sigma_p50_m=float(np.median([sfun(i) for i in range(len(c))])) if sfun else np.nan, true_sigma_p50_m=float(np.median(sig_true))))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ C4
def load_leo_subset(max_obj=600, days=3):
    obj = pd.read_csv(C.DATA / "objects.csv").set_index("norad")
    cache = C.DATA / f"c4_leo_{max_obj}_{days}d.npz"
    if cache.exists():
        z = np.load(cache); return z["traj"], z["ids"], pd.DatetimeIndex(z["grid"]).tz_localize("UTC"), obj
    z = np.load(C.DATA / "traj_window.npz"); times = pd.DatetimeIndex(z["times"]).tz_localize("UTC"); norad = z["norad"]; X = z["X"]
    t_end = times[-1]; t_start = t_end - pd.Timedelta(days=days); i0 = int(np.searchsorted(times.values, np.datetime64(t_start.tz_convert(None))))
    s0 = X[:, i0, :].astype(float); ok = np.isfinite(s0).all(1); alt = np.linalg.norm(s0[:, :3], axis=1) - C.RE_KM
    sel = np.where(ok & (alt >= 300) & (alt <= 2000))[0]; rng = np.random.default_rng(C.SEED)
    if len(sel) > max_obj: sel = np.sort(rng.choice(sel, max_obj, replace=False))
    ids = norad[sel]; grid = pd.date_range(t_start, t_end, freq="1min", tz="UTC"); traj = np.full((len(sel), len(grid), 6), np.nan, dtype=np.float32)
    from .stage0_staging import _stage_one
    from concurrent.futures import ProcessPoolExecutor, as_completed
    pos = {int(n): k for k, n in enumerate(ids)}
    with ProcessPoolExecutor(C.N_WORKERS) as ex:
        futs = [ex.submit(_stage_one, (int(n), obj.loc[int(n)].source, obj.loc[int(n)].path, t_start, t_end, 1)) for n in ids]
        for q, fu in enumerate(as_completed(futs)):
            n, arr, _ = fu.result()
            if arr is not None: traj[pos[int(n)]] = arr
            if (q + 1) % 100 == 0: log.info(f"C4 1-min trajectories {q+1}/{len(ids)}")
    np.savez_compressed(cache, traj=traj, ids=ids, grid=grid.tz_convert(None).values.astype("datetime64[ns]"))
    return traj, ids, grid, obj

def c4_budget_curve() -> tuple[pd.DataFrame, pd.DataFrame]:
    traj, ids, grid, obj = load_leo_subset()
    first = np.argmax(np.isfinite(traj).all(2), axis=1); s0 = traj[np.arange(len(ids)), first, :].astype(float)
    pairs = W.apo_peri_pairs(s0)
    truth = W.screen(traj, ids, grid, pairs, W.COARSE_KM, 10.0)
    log.info(f"C4: {len(ids)} LEO objects, {len(pairs)} candidate pairs of {len(ids)*(len(ids)-1)//2}, {len(truth)} reference approaches < 10 km")
    alt0 = np.linalg.norm(s0[:, :3], axis=1) - C.RE_KM; shell = np.floor((alt0 - 200) / 50) * 50 + 200
    pr = pd.read_parquet(C.RESULTS / "t5_shell_priority.parquet"); pr = pr[pr.h == 24]; dens = pr.groupby("alt_lo").pred_members.mean()
    d_obj = np.nan_to_num(dens.reindex(shell).to_numpy(), nan=0.0)
    truth_pairs = set(zip(truth.i, truth.j)); pos = {int(n): k for k, n in enumerate(ids)}
    hit = np.array([(int(ids[a]), int(ids[b])) in truth_pairs for a, b in pairs])
    scores = {"T5 shell density (product)": d_obj[pairs[:, 0]] * d_obj[pairs[:, 1]], "T5 shell density (min)": np.minimum(d_obj[pairs[:, 0]], d_obj[pairs[:, 1]]),
              "same-shell first": -np.abs(shell[pairs[:, 0]] - shell[pairs[:, 1]]),
              "random": np.random.default_rng(C.SEED).random(len(pairs)), "oracle": hit.astype(float)}
    budgets = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]); rows = []
    for name, sc in scores.items():
        order = np.argsort(-sc, kind="stable"); cum = np.cumsum(hit[order]) / max(hit.sum(), 1)
        for b in budgets:
            k = max(1, int(b * len(pairs))); rows.append(dict(ordering=name, budget=b, n_pairs=k, recall=float(cum[k - 1])))
        auc = float(np.trapezoid(np.concatenate([[0], cum]), np.linspace(0, 1, len(cum) + 1)))
        rows.append(dict(ordering=name, budget="AUC", n_pairs=len(pairs), recall=auc))
    curve = pd.DataFrame(rows)
    summ = curve[curve.budget != "AUC"].pivot_table(index="ordering", columns="budget", values="recall").reset_index()
    summ["AUC"] = summ.ordering.map(curve[curve.budget == "AUC"].set_index("ordering").recall)
    summ["n_truth"] = len(truth); summ["n_pairs"] = len(pairs); summ["n_objects"] = len(ids)
    return curve, summ

# ------------------------------------------------------------------ C5 table + figures
def c5_table(c1, c2, c3, c4s) -> pd.DataFrame:
    rows = []
    try:
        p5 = pd.read_csv(C.RESULTS / "p5_continuity.csv").set_index("method")
        rows.append(dict(stage="Trajectory product (T4)", metric="kinematic jump at TLE switch, P90 (m)", A0="nearest TLE", A0_value=round(p5.loc["nearest", "jump_p90_m"], 1), A1="MTF", A1_value=round(p5.loc["MTF", "jump_p90_m"], 1)))
        rows.append(dict(stage="Trajectory product (T4)", metric="share of switches with jump > 50 m", A0="nearest TLE", A0_value=round(p5.loc["nearest", "frac_jump_gt_50m"], 3), A1="MTF", A1_value=round(p5.loc["MTF", "frac_jump_gt_50m"], 3)))
    except Exception: pass
    c1 = c1.copy(); c1["h"] = c1.h.astype(str)
    t = c1[(c1.component == "t") & (c1.h == "all")]
    for split in ("post-maneuver (logged)", "all"):
        a = t[(t.model == "sigma-age") & (t.split == split)]; b = t[(t.model == "sigma-age x g(T2 score)") & (t.split == split)]; o = t[(t.model == "sigma-age x g(both)") & (t.split == split)]
        if len(a) and len(b): rows.append(dict(stage=f"Uncertainty (T2 -> UQ), {split}", metric="along-track 95% cov / NEES", A0="sigma-age only", A0_value=f"{a.cov95.iloc[0]:.3f} / {a.nees.iloc[0]:.2f}", A1="sigma-age x g(T2 score)", A1_value=f"{b.cov95.iloc[0]:.3f} / {b.nees.iloc[0]:.2f}"))
        if len(a) and len(o): rows.append(dict(stage=f"Uncertainty (events -> UQ), {split}", metric="along-track 95% cov / NEES", A0="sigma-age only", A0_value=f"{a.cov95.iloc[0]:.3f} / {a.nees.iloc[0]:.2f}", A1="sigma-age x g(T2 + maneuver log)", A1_value=f"{o.cov95.iloc[0]:.3f} / {o.nees.iloc[0]:.2f}"))
    for L in (6, 24):
        a = c2[(c2.lead_h == L) & (c2.ranking == "point miss") & (c2.target.str.startswith("matched"))]; b = c2[(c2.lead_h == L) & (c2.ranking == "Pc sigma-age x T2") & (c2.target.str.startswith("matched"))]
        if len(a) and len(b): rows.append(dict(stage=f"Warning ranking (UQ -> T6), lead {L} h", metric="recall@K of true approaches", A0="point miss", A0_value=round(a.recall_at_k.iloc[0], 3), A1="Pc(sigma-age x T2)", A1_value=round(b.recall_at_k.iloc[0], 3)))
    if c4s is not None and len(c4s):
        s4 = c4s.set_index("ordering")
        for b in (0.2, 0.5):
            if b in s4.columns: rows.append(dict(stage=f"Screening budget (T5 -> T6), {int(b*100)}% of pairs", metric="recall of reference approaches", A0="random order", A0_value=round(s4.loc["random", b], 3), A1="T5 shell density (min)", A1_value=round(s4.loc["T5 shell density (min)", b], 3)))
        if "AUC" in s4.columns: rows.append(dict(stage="Screening budget (T5 -> T6)", metric="AUC of recall-budget curve", A0="random order", A0_value=round(s4.loc["random", "AUC"], 3), A1="T5 shell density (min)", A1_value=round(s4.loc["T5 shell density (min)", "AUC"], 3)))
    def g(name): r = c3[c3.policy.str.startswith(name)]; return r.iloc[0] if len(r) else None
    a, b, cc, f = g("Distance-max"), g("Pc-gate, OrbBench"), g("Pc-gate, calibrated"), g("Pc-gate, one fixed")
    if a is not None and cc is not None:
        rows.append(dict(stage="Avoidance (UQ -> T7), 117 CDM events", metric="mean delta-v (m/s) [safety by true Pc<1e-4]", A0="distance rule (2 km)", A0_value=f"{a.dv_mean_m_s:.2f} [{a.safety_rate_pc:.2f}]", A1="Pc-gate, per-event calibrated sigma", A1_value=f"{cc.dv_mean_m_s:.2f} [{cc.safety_rate_pc:.2f}]"))
    if f is not None and cc is not None:
        rows.append(dict(stage="Avoidance (UQ -> T7), 117 CDM events", metric="mean delta-v (m/s) [safety by true Pc<1e-4]", A0="Pc-gate, one fixed sigma", A0_value=f"{f.dv_mean_m_s:.2f} [{f.safety_rate_pc:.2f}]", A1="Pc-gate, per-event calibrated sigma", A1_value=f"{cc.dv_mean_m_s:.2f} [{cc.safety_rate_pc:.2f}]"))
    return pd.DataFrame(rows)

def figures(c1, curve):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BLUE, RED, YELLOW, GREEN, PURPLE = "#4477AA", "#EE6677", "#FFBE7A", "#1F7940", "#BEB8DC"
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    if curve is not None and len(curve):
        fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
        for name, col, ls in (("T5 shell density (product)", BLUE, "-"), ("same-shell first", GREEN, "-."), ("random", RED, "--"), ("oracle", "#999999", ":")):
            g = curve[(curve.ordering == name) & (curve.budget != "AUC")].copy(); g["budget"] = g.budget.astype(float)
            ax.plot(g.budget, g.recall, marker="o", ms=2.5, lw=0.9, ls=ls, color=col, label=name)
        ax.set_xlabel("Fraction of candidate pairs screened"); ax.set_ylabel("Recall of reference approaches"); ax.grid(color="#DDDDDD", lw=0.4, alpha=0.6); ax.legend(frameon=False)
        fig.savefig(C.FIGURES / "fig_c4_budget.pdf"); fig.savefig(C.FIGURES / "fig_c4_budget.png", dpi=300); plt.close(fig)
    c1 = c1.copy(); c1["h"] = c1.h.astype(str)
    t = c1[(c1.component == "t") & (c1.h != "all") & (c1.split.isin(["post-maneuver (logged)", "no recent maneuver"]))]
    if len(t):
        fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9), constrained_layout=True, sharey=True)
        for ax, split in zip(axes, ("no recent maneuver", "post-maneuver (logged)")):
            for model, col in (("sigma-age", RED), ("sigma-age x g(T2 score)", BLUE), ("sigma-age x g(both)", GREEN)):
                g = t[(t.split == split) & (t.model == model)].copy(); g["h"] = g.h.astype(float); g = g.sort_values("h")
                ax.plot(g.h, g.cov95, marker="o", ms=2.5, lw=0.9, color=col, label=model)
            ax.axhline(0.95, color="#555555", lw=0.6, ls="--"); ax.set_xscale("log"); ax.set_xticks([1, 4, 24]); ax.set_xticklabels(["1", "4", "24"]); ax.set_title(split, fontsize=7); ax.set_xlabel("Horizon (h)")
        axes[0].set_ylabel("95% coverage (T)"); axes[0].legend(frameon=False); fig.savefig(C.FIGURES / "fig_c1_coverage.pdf"); fig.savefig(C.FIGURES / "fig_c1_coverage.png", dpi=300); plt.close(fig)

def main(parts="12345", force=False):
    if stage_done("S8e_chain_proof") and not force: log.info("done"); return
    set_status("S8e_chain_proof", "running")
    try:
        with Timer() as T:
            c1 = c2 = c3 = c4s = curve = None
            if "1" in parts:
                c1 = c1_maneuver_sigma(); save_df(c1, "c1_maneuver_sigma.csv")
                log.info("C1 (T component, h=all):\n" + c1[(c1.component == "t") & (c1.h == "all")][["model", "split", "n", "cov68", "cov95", "nees", "err_p50_km", "sigma_p50_km"]].round(3).to_string(index=False))
                log.info(f"C1 g factors: T2 {c1.g_T2.iloc[0]}  oracle {c1.g_oracle.iloc[0]}")
            if "2" in parts:
                c2 = c2_uq_ranking(); save_df(c2, "c2_uq_ranking.csv"); log.info("C2:\n" + c2.round(3).to_string(index=False))
            if "3" in parts:
                c3 = c3_avoidance_sigma(); save_df(c3, "c3_avoidance_sigma.csv"); log.info("C3:\n" + c3.round(3).to_string(index=False))
            if "4" in parts:
                curve, c4s = c4_budget_curve(); save_df(curve, "c4_budget_curve.csv"); save_df(c4s, "c4_budget_summary.csv"); log.info("C4:\n" + c4s.round(3).to_string(index=False))
            if "5" in parts:
                c1 = c1 if c1 is not None else pd.read_csv(C.RESULTS / "c1_maneuver_sigma.csv"); c2 = c2 if c2 is not None else pd.read_csv(C.RESULTS / "c2_uq_ranking.csv")
                c3 = c3 if c3 is not None else pd.read_csv(C.RESULTS / "c3_avoidance_sigma.csv"); c4s = c4s if c4s is not None else (pd.read_csv(C.RESULTS / "c4_budget_summary.csv") if (C.RESULTS / "c4_budget_summary.csv").exists() else None)
                if c4s is not None: c4s.columns = [float(c) if str(c).replace(".", "", 1).isdigit() else c for c in c4s.columns]
                tab = c5_table(c1, c2, c3, c4s); save_df(tab, "chain_a0_a1.csv"); log.info("A0 vs A1:\n" + tab.to_string(index=False))
                try: figures(c1, curve if curve is not None else pd.read_csv(C.RESULTS / "c4_budget_curve.csv"))
                except Exception as ex: log.warning(f"figures skipped: {ex}")
        set_status("S8e_chain_proof", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("chain proof failed"); set_status("S8e_chain_proof", "failed", message=str(ex)); raise

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(parts=args[0] if args else "12345", force="--force" in sys.argv)
