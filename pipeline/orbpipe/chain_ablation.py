"""Task-level chain ablations: full chain vs the same chain with one upstream task's output removed,
judged on the downstream task's metric with truth.

X1  Warning -> avoidance chain on 199 ESA CDM events (truth = operator covariances in the CDM):
      T4-UQ predicts the chaser covariance from OD age (+ T5 object/orbit attributes, + T3 space weather),
      T6 computes an encounter-plane Pc from it and ranks events, T7 fires only when predicted Pc >= PC_GATE and thrusts until predicted Pc < PC_TARGET.
    Ablations: -T3, -T5, -T4 (fixed sigma), -T6 (distance rule), none (A0 = fixed sigma + distance rule).
    Downstream: T7 safety (true Pc_post < PC_GATE on all events), delta-v, unnecessary maneuvers; T6 ranking quality.
X2  Phase/maneuver -> forecast-reliability -> screening triage on 17 Starlinks (truth = MEME):
      T4 flags forecasts expected to miss by > thr km so that T6 does not screen on them; inputs age (+T1 phase and
      SMA rate, +T2 consistency score, +T3 Ap/F10.7). Ablations: drop each. Downstream: AUC, recall at 30 % flag budget,
      Spearman with the true error.
X3  Evolution -> warning -> avoidance on 600 LEO cohort objects (139 reference approaches < 10 km, 3 d):
      T5 orders candidate pairs, T6 screens the top BUDGET fraction and computes Pc with sigma-age, T7 Pc-gates.
    Ablations: -T5 (random order), -T6 sigma (distance rule), both. Downstream: approaches handled within budget, delta-v.
Outputs: results/x1_cdm_chain.csv, x1_cdm_ranking.csv, x2_starlink_triage.csv, x3_leo_chain.csv, chain_ablation_summary.csv
"""
from __future__ import annotations
import sys, os, json
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".cache", "orbbench-mpl"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df
from .starlink_chain import pc_encounter_plane
from .util import read_tles, to_utc, mad

log = get_logger("chain_ablation")
PC_GATE = 1e-4; PC_TARGET = 1e-5; HBR_KM = 0.010; AMAX = 0.3; DT = 1.0; LEAD_S = 150.0; TMAX = 150

# ------------------------------------------------------------------ shared T7 with covariance gate
def run_avoidance(rel_r_m, rel_v_m, P_pred_km2, P_true_km2, mode: str):
    """mode: 'none' | 'distance' (thrust while predicted miss < 2 km) | 'pc' (thrust while Pc(P_pred) >= PC_GATE).
    Relative motion linear over [TCA-150 s, TCA]; thrust along the encounter-plane miss direction.
    Returns dv (m/s), true Pc after (with P_true) and final predicted miss (m)."""
    r = np.array(rel_r_m, float); v = np.array(rel_v_m, float); vn = np.linalg.norm(v)
    vhat = v / vn if vn > 0 else np.array([1.0, 0, 0]); r_perp = r - np.dot(r, vhat) * vhat
    if np.linalg.norm(r_perp) < 1e-3:
        r_perp = np.cross(vhat, [0, 0, 1.0]); r_perp = r_perp if np.linalg.norm(r_perp) > 1e-6 else np.cross(vhat, [0, 1.0, 0])
    dir0 = r_perp / np.linalg.norm(r_perp); r = r - v * LEAD_S; dv = 0.0
    armed = mode == "pc" and pc_encounter_plane(r / 1000 + v / 1000 * max(-np.dot(r, v) / np.dot(v, v), 0.0), v / 1000, P_pred_km2, HBR_KM) >= PC_GATE
    if mode == "none" or (mode == "pc" and not armed): return 0.0, pc_encounter_plane((r + v * max(-np.dot(r, v) / np.dot(v, v), 0.0)) / 1000, v / 1000, P_true_km2, HBR_KM), float(np.linalg.norm(r + v * max(-np.dot(r, v) / np.dot(v, v), 0.0)))
    for k in range(TMAX):
        vn2 = np.dot(v, v); tca = -np.dot(r, v) / vn2 if vn2 > 0 else 0.0; r_tca = r + v * max(tca, 0.0)
        if mode == "distance": a = AMAX if np.linalg.norm(r_tca) < 2000.0 else 0.0
        else: a = AMAX if pc_encounter_plane(r_tca / 1000, v / 1000, P_pred_km2, HBR_KM) >= PC_TARGET else 0.0   # alerted at PC_GATE, maneuver down to PC_TARGET
        v = v + dir0 * a * DT; r = r + v * DT; dv += a * DT
    vn2 = np.dot(v, v); tca = -np.dot(r, v) / vn2 if vn2 > 0 else 0.0; r_tca = r + v * max(tca, 0.0)
    return dv, pc_encounter_plane(r_tca / 1000, v / 1000, P_true_km2, HBR_KM), float(np.linalg.norm(r_tca))

# ------------------------------------------------------------------ X1
def x1_cdm():
    c = pd.read_csv(C.CDM_CSV).reset_index(drop=True)
    c["type"] = c.c_object_type.astype("category").cat.codes; c["age_d"] = c.c_time_lastob_start
    F = {"T4 age/OD": ["age_d", "c_obs_used", "c_actual_od_span"], "T5 object/orbit": ["type", "c_h_per", "c_h_apo", "c_cd_area_over_mass", "c_rcs_estimate"], "T3 weather": ["AP", "F10", "F3M"]}
    def sigma_model(cols):
        """5-fold CV prediction of the chaser RTN sigmas (m) from task outputs; returns [n,3]."""
        X = c[cols].fillna(c[cols].median()); out = np.zeros((len(c), 3))
        for j, comp in enumerate(("c_sigma_r", "c_sigma_t", "c_sigma_n")):
            y = np.log(c[comp].values); pred = np.zeros(len(y))
            for tr, te in KFold(5, shuffle=True, random_state=C.SEED).split(X):
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=1.0, random_state=C.SEED).fit(X.iloc[tr], y[tr]); pred[te] = m.predict(X.iloc[te])
            out[:, j] = np.exp(pred)
        return out
    true_c = c[["c_sigma_r", "c_sigma_t", "c_sigma_n"]].to_numpy(float); fixed = np.tile(np.median(true_c, axis=0), (len(c), 1))
    full = sigma_model(F["T4 age/OD"] + F["T5 object/orbit"] + F["T3 weather"])
    variants = {
        "A1 full (T3+T5+T4 -> T6 Pc -> T7 Pc-gate)": (full, "pc", 1.0),
        "  -T3 (no space weather)": (sigma_model(F["T4 age/OD"] + F["T5 object/orbit"]), "pc", 1.0),
        "  -T5 (no object/orbit attributes)": (sigma_model(F["T4 age/OD"] + F["T3 weather"]), "pc", 1.0),
        "  -T3 -T5 (age/OD only)": (sigma_model(F["T4 age/OD"]), "pc", 1.0),
        "  -T4 (fixed sigma, no uncertainty task)": (fixed, "pc", 1.0),
        "  -T6 (distance rule, ignores Pc)": (full, "distance", 1.0),
        "A0 none (fixed sigma + distance rule)": (fixed, "distance", 1.0),
        "oracle (operator per-event covariance)": (true_c, "pc", 1.0),
        "no maneuver": (fixed, "none", 1.0),
    }
    for k in (0.7, 0.5, 0.35, 0.25):
        variants[f"A1 full, safety factor sigma x{k}"] = (full, "pc", k); variants[f"  -T4 fixed sigma x{k}"] = (fixed, "pc", k)
    rows = []; rank_rows = []; pc_true_all = None; cases = []
    for name, (sig_pred, mode, kfac) in variants.items():
        res = []
        for i, r in c.iterrows():
            rel_r = np.array([r.relative_position_r, r.relative_position_t, r.relative_position_n]); rel_v = np.array([r.relative_velocity_r, r.relative_velocity_t, r.relative_velocity_n])
            Pt = np.diag(np.array([r.t_sigma_r, r.t_sigma_t, r.t_sigma_n]) ** 2) / 1e6
            P_true = Pt + np.diag(true_c[i] ** 2) / 1e6; P_pred = Pt + np.diag((kfac * sig_pred[i]) ** 2) / 1e6
            pc_true = pc_encounter_plane(rel_r / 1000, rel_v / 1000, P_true, HBR_KM); pc_pred = pc_encounter_plane(rel_r / 1000, rel_v / 1000, P_pred, HBR_KM)
            dv, pc_post, miss_post = run_avoidance(rel_r, rel_v, P_pred, P_true, mode)
            res.append(dict(pc_true=pc_true, pc_pred=pc_pred, dv=dv, pc_post=pc_post, miss_post=miss_post, maneuvered=dv > 0, dangerous=pc_true >= PC_GATE))
        R = pd.DataFrame(res); dang = R.dangerous; R["chain"] = name; R["event"] = np.arange(len(R)); cases.append(R.copy())
        if pc_true_all is None: pc_true_all = R.pc_true.values
        rows.append(dict(chain=name, n_events=len(R), n_dangerous=int(dang.sum()), safety_all=float((R.pc_post < PC_GATE).mean()), safety_dangerous=float((R[dang].pc_post < PC_GATE).mean()) if dang.any() else np.nan,
                         missed_dangerous=int(((~R.maneuvered) & dang).sum()), unnecessary_maneuvers=int((R.maneuvered & ~dang).sum()), dv_total_m_s=float(R.dv.sum()), dv_mean_m_s=float(R.dv.mean()),
                         sigma_t_within_x2=float(np.mean(np.abs(np.log(kfac * sig_pred[:, 1] / true_c[:, 1])) < np.log(2)))))
        if mode == "pc" and kfac == 1.0:
            y = dang.values.astype(int); sc = R.pc_pred.values; K = max(int(y.sum()), 1); o = np.argsort(-sc)
            rank_rows.append(dict(chain=name, spearman_pcpred_vs_pctrue=float(spearmanr(sc, R.pc_true).statistic), recall_at_k=float(y[o[:K]].mean()), recall_at_2k=float(y[o[:2 * K]].sum() / max(y.sum(), 1)),
                                  auc_dangerous=float(roc_auc_score(y, sc)) if 0 < y.sum() < len(y) else np.nan))
    pd.concat(cases, ignore_index=True).to_parquet(C.RESULTS / "x1_cdm_cases.parquet", index=False)
    o = pd.DataFrame(rows); rk = pd.DataFrame(rank_rows)
    y = (pc_true_all >= PC_GATE).astype(int); sc = -c.miss_distance.values; K = max(int(y.sum()), 1); oo = np.argsort(-sc)
    rk.loc[len(rk)] = dict(chain="point miss (no T4/T6 uncertainty)", spearman_pcpred_vs_pctrue=float(spearmanr(sc, pc_true_all).statistic), recall_at_k=float(y[oo[:K]].mean()), recall_at_2k=float(y[oo[:2 * K]].sum() / max(y.sum(), 1)),
                           auc_dangerous=float(roc_auc_score(y, sc)) if 0 < y.sum() < len(y) else np.nan)
    return o, rk

# ------------------------------------------------------------------ X2
def x2_starlink_triage():
    cs = pd.read_parquet(C.RESULTS / "s1_starlink_cases.parquet"); cs = cs[cs.method == "A0 raw prev"].copy(); cs["t0"] = pd.to_datetime(cs.t0, utc=True)
    rows = []
    for l in open(C.DATA / "Kp_ap_Ap_SN_F107_since_1932.txt"):
        if l.startswith("#") or not l.strip(): continue
        p = l.split()
        try: rows.append((int(p[0]), int(p[1]), int(p[2]), float(p[23]), float(p[25])))
        except Exception: pass
    sw = pd.DataFrame(rows, columns=["y", "m", "d", "Ap", "F107"]); sw["date"] = pd.to_datetime(sw[["y", "m", "d"]].rename(columns={"y": "year", "m": "month", "d": "day"})); sw = sw.set_index("date")
    d0 = cs.t0.dt.tz_convert(None).dt.normalize(); cs["Ap"] = sw.Ap.reindex(d0).to_numpy(); cs["Ap_lag1"] = sw.Ap.reindex(d0 - pd.Timedelta(days=1)).to_numpy(); cs["F107"] = sw.F107.reindex(d0).to_numpy()
    F = {"T4 age": ["age_h"], "T1 phase": ["phase", "sma_rate_km_d"], "T2 maneuver score": ["t2_score"], "T3 weather": ["Ap", "Ap_lag1", "F107"]}
    out = []; preds = []
    for h, thr in ((4.0, 3.0), (24.0, 10.0)):
        d = cs[cs.h == h]; tr = d[~d.is_test]; te = d[d.is_test]; ytr = (tr.err_3d > thr).astype(int); yte = (te.err_3d > thr).astype(int)
        variants = {"A1 full (T1+T2+T3 -> T4 reliability -> T6 triage)": F["T4 age"] + F["T1 phase"] + F["T2 maneuver score"] + F["T3 weather"],
                    "  -T3": F["T4 age"] + F["T1 phase"] + F["T2 maneuver score"], "  -T2": F["T4 age"] + F["T1 phase"] + F["T3 weather"], "  -T1": F["T4 age"] + F["T2 maneuver score"] + F["T3 weather"],
                    "  -T1 -T2 -T3 (A0: age only)": F["T4 age"]}
        for name, cols in variants.items():
            Xtr = tr[cols].fillna(tr[cols].median()); Xte = te[cols].fillna(tr[cols].median())
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=1.0, random_state=C.SEED).fit(Xtr, ytr); p = clf.predict_proba(Xte)[:, 1]
            k = int(0.3 * len(te)); top = np.argsort(-p)[:k]; preds.append(pd.DataFrame(dict(h=h, chain=name, norad=te.norad.values, t0=te.t0.values, y=yte.values, p=p, err_3d=te.err_3d.values)))
            reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=1.0, random_state=C.SEED).fit(Xtr, np.log(tr.err_3d.clip(lower=1e-3))); pe = reg.predict(Xte)
            out.append(dict(h=h, threshold_km=thr, chain=name, n_test=len(te), positive_rate=float(yte.mean()), auc=float(roc_auc_score(yte, p)), avg_precision=float(average_precision_score(yte, p)),
                            recall_at_30pct_flag=float(yte.values[top].sum() / max(yte.sum(), 1)), precision_at_30pct_flag=float(yte.values[top].mean()), spearman_pred_vs_true_err=float(spearmanr(pe, te.err_3d).statistic)))
    pd.concat(preds, ignore_index=True).to_parquet(C.RESULTS / "x2_starlink_cases.parquet", index=False)
    return pd.DataFrame(out)

# ------------------------------------------------------------------ X3
def x3_leo_chain():
    curve = pd.read_csv(C.RESULTS / "c4_budget_curve.csv"); curve = curve[curve.budget != "AUC"].copy(); curve["budget"] = curve.budget.astype(float)
    from . import t6_warning as W
    z = np.load(C.DATA / "c4_leo_600_3d.npz"); traj = z["traj"]; ids = z["ids"]; grid = pd.DatetimeIndex(z["grid"]).tz_localize("UTC")
    first = np.argmax(np.isfinite(traj).all(2), axis=1); s0 = traj[np.arange(len(ids)), first, :].astype(float)
    pairs = W.apo_peri_pairs(s0); truth = W.screen(traj, ids, grid, pairs, W.COARSE_KM, 10.0)
    calib = json.loads((C.RESULTS / "t4_uq_calibration.json").read_text())
    sig = np.array([max(calib[c]["age_a"] + calib[c]["age_b"] * 24.0, 1e-3) for c in ("r", "t", "n")])          # cohort sigma-age at 24 h age, km
    P_pred = np.diag(sig ** 2) * 2; P_true = P_pred                                                             # no independent covariance for cohort objects
    per = []
    for e in truth.itertuples():
        rel_r = np.array([e.rx, e.ry, e.rz]) * 1000; rel_v = np.array([e.vx, e.vy, e.vz]) * 1000
        dv_pc, pc_post, _ = run_avoidance(rel_r, rel_v, P_pred, P_true, "pc"); dv_d, _, miss_d = run_avoidance(rel_r, rel_v, P_pred, P_true, "distance")
        per.append(dict(miss_km=e.miss_km, dv_pc=dv_pc, dv_dist=dv_d, pc_pre=pc_encounter_plane(rel_r / 1000, rel_v / 1000, P_true, HBR_KM)))
    per = pd.DataFrame(per); budget = 0.2
    rec = {o: float(curve[(curve.ordering == o) & (curve.budget == budget)].recall.iloc[0]) for o in ("T5 shell density (min)", "random", "oracle")}
    rows = []
    for name, order, mode in (("A1 full (T5 order -> T6 Pc -> T7 Pc-gate)", "T5 shell density (min)", "pc"), ("  -T5 (random order)", "random", "pc"), ("  -T6 sigma (distance rule)", "T5 shell density (min)", "distance"),
                              ("A0 none (random order + distance rule)", "random", "distance"), ("oracle ordering + Pc-gate", "oracle", "pc")):
        n_h = rec[order] * len(truth); dv_ev = per.dv_pc.mean() if mode == "pc" else per.dv_dist.mean()
        rows.append(dict(chain=name, budget_fraction=budget, n_reference_approaches=len(truth), approaches_reached=round(n_h, 1), recall=rec[order], dv_per_event_m_s=round(dv_ev, 3), dv_total_m_s=round(dv_ev * n_h, 2),
                         events_with_pc_ge_gate=int((per.pc_pre >= PC_GATE).sum()), dv_per_reached_dangerous=np.nan))
    return pd.DataFrame(rows)

def main(force=False, force_all=False):
    if stage_done("S8g_chain_ablation") and not force: log.info("done"); return
    set_status("S8g_chain_ablation", "running")
    try:
        with Timer() as T:
            if (C.RESULTS / "x1_cdm_chain.csv").exists() and not force_all: x1, rk = pd.read_csv(C.RESULTS / "x1_cdm_chain.csv"), pd.read_csv(C.RESULTS / "x1_cdm_ranking.csv")
            else: x1, rk = x1_cdm()
            save_df(x1, "x1_cdm_chain.csv"); save_df(rk, "x1_cdm_ranking.csv"); log.info("X1:\n" + x1.round(3).to_string(index=False)); log.info("X1 ranking:\n" + rk.round(3).to_string(index=False))
            x2 = pd.read_csv(C.RESULTS / "x2_starlink_triage.csv") if (C.RESULTS / "x2_starlink_triage.csv").exists() and not force_all else x2_starlink_triage(); save_df(x2, "x2_starlink_triage.csv")
            x3 = pd.read_csv(C.RESULTS / "x3_leo_chain.csv") if (C.RESULTS / "x3_leo_chain.csv").exists() and not force_all else x3_leo_chain(); save_df(x3, "x3_leo_chain.csv")
            x4 = x4_sentinel_chain(); save_df(x4, "x4_sentinel_chain.csv"); log.info("X4:\n" + x4.round(3).to_string(index=False))
            summ = []
            for _, r in x1.iterrows(): summ.append(dict(example="X1 CDM warning->avoidance", chain=r.chain, downstream_metric="safety (true Pc<1e-4) | dv total m/s | missed dangerous | unnecessary", value=f"{r.safety_all:.3f} | {r.dv_total_m_s:.1f} | {r.missed_dangerous} | {r.unnecessary_maneuvers}"))
            for _, r in x2[x2.h == 24].iterrows(): summ.append(dict(example="X2 Starlink triage (24 h, >10 km)", chain=r.chain, downstream_metric="AUC | recall@30% flag", value=f"{r.auc:.3f} | {r.recall_at_30pct_flag:.3f}"))
            for _, r in x3.iterrows(): summ.append(dict(example="X3 LEO evolution->warning->avoidance (20% budget)", chain=r.chain, downstream_metric="approaches reached | dv total m/s", value=f"{r.approaches_reached} | {r.dv_total_m_s}"))
            for _, r in x4[(x4.h == 24) & (x4.split == "post-maneuver (logged, 48 h)")].iterrows(): summ.append(dict(example="X4 Sentinel T2->UQ->warning->avoidance (24 h, post-maneuver)", chain=r.chain, downstream_metric="T6 alert recall | T7 safety on dangerous | missed | unnecessary | dv", value=f"{r.t6_alert_recall:.3f} | {r.t7_safety_dangerous:.3f} | {r.t7_missed_dangerous} | {r.t7_unnecessary} | {r.dv_total_m_s:.0f}"))
            save_df(pd.DataFrame(summ), "chain_ablation_summary.csv")
        set_status("S8g_chain_ablation", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("chain ablation failed"); set_status("S8g_chain_ablation", "failed", message=str(ex)); raise


def pc_fast(rel_r, rel_v, P_rel, hbr_km):
    """Closed-form encounter-plane Pc for hbr << sigma: pi hbr^2 x N(mu; 0, P2). Error ~ (hbr/sigma)^2, < 0.3% at sigma >= 0.2 km."""
    v = rel_v / max(np.linalg.norm(rel_v), 1e-12); u = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(v, u); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1); B = np.vstack([e1, e2]); mu = B @ rel_r; P2 = B @ P_rel @ B.T
    det = P2[0, 0] * P2[1, 1] - P2[0, 1] * P2[1, 0]
    if det <= 0: return float(np.linalg.norm(mu) < hbr_km)
    Pi = np.array([[P2[1, 1], -P2[0, 1]], [-P2[1, 0], P2[0, 0]]]) / det; q = mu @ Pi @ mu
    return float(np.pi * hbr_km ** 2 * np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det)))

# ------------------------------------------------------------------ X4  Sentinel: T2 -> T4-UQ -> T6 -> T7 (synthetic chasers on POD truth)
SIG_CHASER_KM = 0.2; RHO_RANGE_KM = (0.05, 10.0); K_CHASERS = 4

def _avoid_disp(rel_r_m, rel_v_m, P_pred_km2, mode):
    """Same controller as run_avoidance; returns dv (m/s) and the displacement (m) of the TCA point produced by the thrust."""
    r = np.array(rel_r_m, float); v = np.array(rel_v_m, float); vn = np.linalg.norm(v); vhat = v / vn
    r_perp = r - np.dot(r, vhat) * vhat
    if np.linalg.norm(r_perp) < 1e-3: r_perp = np.cross(vhat, [0, 0, 1.0])
    dir0 = r_perp / np.linalg.norm(r_perp)
    vn2 = np.dot(v, v); r_tca0 = r + v * max(-np.dot(r, v) / vn2, 0.0)
    r = r - v * LEAD_S; v0 = v.copy(); dv = 0.0
    # without thrust the TCA point is constant, so a gate that is closed at the start never opens: skip the loop
    if mode == "none" or (mode == "distance" and np.linalg.norm(r_tca0) >= 2000.0) or (mode == "pc" and pc_fast(r_tca0 / 1000, v / 1000, P_pred_km2, HBR_KM) < PC_GATE):
        return 0.0, np.zeros(3), v
    for k in range(TMAX):
        tca = -np.dot(r, v) / np.dot(v, v); r_tca = r + v * max(tca, 0.0)
        if mode == "distance": a = AMAX if np.linalg.norm(r_tca) < 2000.0 else 0.0
        elif mode == "pc": a = AMAX if pc_fast(r_tca / 1000, v / 1000, P_pred_km2, HBR_KM) >= PC_TARGET else 0.0   # alerted at PC_GATE, maneuver down to PC_TARGET
        else: a = 0.0
        v = v + dir0 * a * DT; r = r + v * DT; dv += a * DT
    tca = -np.dot(r, v) / np.dot(v, v); r_tca = r + v * max(tca, 0.0)
    return dv, r_tca - r_tca0, v

def x4_sentinel_chain():
    from .chain_proof import _t2_scores, SCORE_BINS, fit_sigma_age
    cs = pd.read_parquet(C.RESULTS / "t4_forecast_cases.parquet"); cs = cs[cs.method == "SGP4-MTF"].copy(); cs["t0"] = pd.to_datetime(cs.t0, utc=True)
    logged = [n for n, v in C.SENTINELS.items() if v[2]]; cs = cs[cs.sat.isin(logged)].copy()
    score = np.zeros(len(cs))
    for name in logged:
        m = (cs.sat == name).values
        if not m.any(): continue
        tl = read_tles(C.SENTINELS[name][0], to_utc(C.SENTINEL_START) - pd.Timedelta(days=5), C.WINDOW_END); ep, sc = _t2_scores(tl)
        t0ns = cs.t0.values[m].astype("datetime64[ns]").astype("int64"); out = np.zeros(m.sum())
        for k, t in enumerate(t0ns):
            w = (ep <= t) & (ep >= t - 48 * 3.6e12); out[k] = np.nanmax(sc[w]) if w.any() and np.isfinite(sc[w]).any() else 0.0
        score[m] = out
    cs["t2_score"] = score; cs["t2_bin"] = np.digitize(score, SCORE_BINS[1:-1]); cs["flag_oracle"] = cs.man_recent.astype(bool)
    t0w = to_utc(C.SENTINEL_TEST_START); tr = cs[cs.t0 < t0w]; g = json.loads((C.RESULTS / "c1_g_factors.json").read_text())
    sig = {}
    for c in ("r", "t", "n"):
        a, b = fit_sigma_age(tr, c); base = g[c]["base"]
        te_c = cs[cs.t0 >= t0w]; s0 = np.maximum(a + b * (te_c.age_h + te_c.h).values, 1e-3) * base
        sig[f"age_{c}"] = s0; sig[f"fixed_{c}"] = np.full(len(te_c), 1.4826 * mad(tr[f"err_{c}"].values))
        sig[f"t2_{c}"] = s0 * np.array([g[c]["bins"][str(int(k))] for k in te_c.t2_bin.values]); sig[f"log_{c}"] = s0 * np.where(te_c.flag_oracle.values, g[c]["oracle"], 1.0)
    te = cs[cs.t0 >= t0w].reset_index(drop=True)
    for k, v in sig.items(): te[f"sigma_{k}"] = v
    rng = np.random.default_rng(C.SEED); V_KM_S = 7.45; Pc_c = np.eye(3) * SIG_CHASER_KM ** 2
    variants = {"A1 full (T2 detector -> T4 sigma -> T6 Pc alert -> T7 Pc-gate)": ("t2", "pc"), "A1 with perfect T2 (logged maneuvers)": ("log", "pc"),
                "  -T2 (sigma-age, maneuver-blind)": ("age", "pc"), "  -T4-UQ (fixed sigma)": ("fixed", "pc"), "  -T6 (distance rule on predicted miss)": ("t2", "distance"),
                "A0 none (fixed sigma + distance rule)": ("fixed", "distance"), "perfect T4 forecast (no error)": ("perfect", "pc"), "no maneuver": ("fixed", "none")}
    rows = []; per_case = []
    for h in (4.0, 24.0):
        d = te[te.h == h].reset_index(drop=True); n = len(d)
        theta = np.radians(rng.uniform(15, 165, (n, K_CHASERS))); rho = np.exp(rng.uniform(np.log(RHO_RANGE_KM[0]), np.log(RHO_RANGE_KM[1]), (n, K_CHASERS))); psi = rng.uniform(0, 2 * np.pi, (n, K_CHASERS))
        err = d[["err_r", "err_t", "err_n"]].to_numpy(float)
        for name, (sk, mode) in variants.items():
            res = []
            for i in range(n):
                for j in range(K_CHASERS):
                    th = theta[i, j]; relv = V_KM_S * np.array([0.0, np.cos(th) - 1, np.sin(th)]); hh = np.array([0.0, np.sin(th), 1 - np.cos(th)]); hh /= np.linalg.norm(hh)
                    m_true = rho[i, j] * (np.cos(psi[i, j]) * np.array([1.0, 0, 0]) + np.sin(psi[i, j]) * hh)
                    e = np.zeros(3) if sk == "perfect" else err[i]
                    rel_pred = m_true - e                                              # chaser minus forecast position
                    ssat = np.array([0.02, 0.02, 0.02]) if sk == "perfect" else np.array([d.loc[i, f"sigma_{sk}_r"], d.loc[i, f"sigma_{sk}_t"], d.loc[i, f"sigma_{sk}_n"]])
                    P_pred = Pc_c + np.diag(ssat ** 2)
                    pc_true = pc_fast(m_true, relv, Pc_c, HBR_KM); pc_pred = pc_fast(rel_pred, relv, P_pred, HBR_KM)
                    dv, disp_m, _ = _avoid_disp(rel_pred * 1000, relv * 1000, P_pred, mode)
                    m_post = m_true + disp_m / 1000
                    pc_post = pc_fast(m_post, relv, Pc_c, HBR_KM)
                    res.append((pc_true, pc_pred, dv, pc_post, bool(d.flag_oracle[i]), d.t2_score[i]))
            R = pd.DataFrame(res, columns=["pc_true", "pc_pred", "dv", "pc_post", "post_man", "t2_score"]); R["dangerous"] = R.pc_true >= PC_GATE; R["alert"] = R.pc_pred >= PC_GATE; R["man"] = R.dv > 0; R["h"] = h; R["chain"] = name; R["origin"] = np.repeat(np.arange(n), K_CHASERS); per_case.append(R.copy())
            for split, mm in (("all", np.ones(len(R), bool)), ("post-maneuver (logged, 48 h)", R.post_man.values), ("no recent maneuver", ~R.post_man.values)):
                S = R[mm]; dg = S.dangerous
                rows.append(dict(h=h, split=split, chain=name, n_conj=len(S), n_dangerous=int(dg.sum()), t6_alert_recall=float((S.alert & dg).sum() / max(dg.sum(), 1)), t6_false_alert_rate=float((S.alert & ~dg).sum() / max((~dg).sum(), 1)),
                                 t7_safety_dangerous=float((S[dg].pc_post < PC_GATE).mean()) if dg.any() else np.nan, t7_missed_dangerous=int(((~S.man) & dg).sum()), t7_unnecessary=int((S.man & ~dg).sum()),
                                 t7_made_worse=int(((S.pc_post >= PC_GATE) & ~dg).sum()), dv_total_m_s=float(S.dv.sum()), dv_per_conj_m_s=float(S.dv.mean())))
    pd.concat(per_case, ignore_index=True).to_parquet(C.RESULTS / "x4_sentinel_cases.parquet", index=False)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    main(force="--force" in sys.argv, force_all="--all" in sys.argv)
