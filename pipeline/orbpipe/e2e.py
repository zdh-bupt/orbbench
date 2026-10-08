"""End-to-end T1->T7 on one cohort: 600 LEO objects, 3-day window (1-min L2 reference trajectories from data/c4_leo_600_3d.npz).

At every 12-h origin t0 the seven tasks run twice on the same TLEs, methods and reference events:
  A0 independent  T4 = SGP4 of the latest TLE, fixed sigma; T6 screens candidate pairs (all, or a random 20 % budget) and alerts
                  on predicted miss < ALERT_KM; T7 thrusts while predicted miss < 2 km.
  A1 chained      T1 phase + T2 flags -> T4 (post-maneuver reset of the TLE window, sigma inflation), T3 Ap forecast -> sigma storm
                  factor, T4 state + RTN covariance -> T6 Pc, T5 shell density -> T6 screening order (20 % budget), T6 Pc >= 1e-4 ->
                  T7 Pc-gated thrust (to 1e-5), maneuver re-propagated and re-checked against the reference.
Ablations remove one task output from A1 (-T1, -T2, -T3, -T4 sigma, -T5 order, -T6 Pc).
Reference (pseudo-truth): the L2 trajectories; reference approaches = pairs with reference miss < CLOSE_KM within the horizon;
"dangerous" = reference miss < DANGER_KM. T1 truth = silver rule labels; T2 has no truth on this cohort (Sentinel F1 reported);
T3 truth = observed indices.
Outputs: results/e2e_task_table.csv, e2e_decision.csv, e2e_ablation.csv, e2e_events.parquet, e2e_timing.csv
"""
from __future__ import annotations
import sys, os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, read_tles, sgp4_states, to_rtn, rtn_basis, mad
from . import t6_warning as W
from .chain_ablation import pc_fast, _avoid_disp, PC_GATE, HBR_KM

log = get_logger("e2e")
HZ_H = 24.0; ORIGIN_STEP_H = 12; BUDGET = 0.2; CLOSE_KM = 10.0; DANGER_KM = 1.0; PRED_KM = 30.0; ALERT_KM = 5.0; MATCH_MIN = 10.0
N_OBJ = int(os.environ.get("E2E_N_OBJ", "1500")); DAYS = 3
INJ_PER_ORIGIN = 30; INJ_MISS_KM = (0.02, 1.0)   # injected conjunctions: rigid shift of the secondary so the reference miss is log-uniform in this range
ERR_H = [1.0, 4.0, 12.0, 24.0]; RTN = ["r", "t", "n"]
ALT_EDGES = np.arange(200, 2001, 50.0); LAT_EDGES = np.arange(-90, 91, 10.0)

# ------------------------------------------------------------------ inputs
def load_cohort():
    from .chain_proof import load_leo_subset
    traj, ids, grid, _ = load_leo_subset(max_obj=N_OBJ, days=DAYS); return traj.astype(float), ids.astype(int), grid

def _tles_one(args):
    norad, t0, t1 = args
    try: t = read_tles(norad, t0, t1); return norad, t[["epoch", "line1", "line2", "n", "e", "inc", "bstar"]]
    except Exception as ex: return norad, None

def load_tles(ids, t0, t1):
    cache = C.DATA / f"e2e_tles_{N_OBJ}.parquet"
    if cache.exists():
        d = pd.read_parquet(cache); d["epoch"] = pd.to_datetime(d.epoch, utc=True); return {int(n): g.reset_index(drop=True) for n, g in d.groupby("norad")}
    out = {}
    with ProcessPoolExecutor(C.N_WORKERS) as ex:
        for fu in as_completed([ex.submit(_tles_one, (int(n), t0, t1)) for n in ids]):
            n, t = fu.result()
            if t is not None and len(t): t = t.copy(); t["norad"] = n; out[n] = t
    pd.concat(out.values(), ignore_index=True).to_parquet(cache, index=False); return out

def task_inputs(ids, grid):
    """Upstream outputs already produced by the pipeline for these objects."""
    t1 = pd.read_parquet(C.RESULTS / "t1_phase_labels.parquet"); t1 = t1[t1.norad.isin(ids)].copy(); t1["t_start"] = pd.to_datetime(t1.t_start, utc=True); t1["t_end"] = pd.to_datetime(t1.t_end, utc=True)
    t2 = pd.read_parquet(C.RESULTS / "t2_cohort_flags.parquet"); t2 = t2[t2.norad.isin(ids)].copy(); t2["epoch"] = pd.to_datetime(t2.epoch, utc=True)
    t3 = pd.read_parquet(C.RESULTS / "t3_weather_forecasts.parquet"); t3["origin_date"] = pd.to_datetime(t3.origin_date); t3["target_date"] = pd.to_datetime(t3.target_date)
    t5 = pd.read_parquet(C.RESULTS / "t5_shell_priority.parquet"); t5["t0"] = pd.to_datetime(t5.t0, utc=True)
    sw = pd.read_csv(C.DATA / "space_weather_daily.csv", parse_dates=["date"]).set_index("date")
    calib = json.loads((C.RESULTS / "t4_uq_calibration.json").read_text()); g = json.loads((C.RESULTS / "c1_g_factors.json").read_text())
    return dict(t1=t1, t2=t2, t3=t3, t5=t5, sw=sw, calib=calib, g=g)

# ------------------------------------------------------------------ T4 forecasts
def forecast_one(tles, t0, targets, chained, inp, norad, flags):
    """Return states [T,6] (TEME km), age_h at t0, inflation factor per component, phase_flag, man_flag, n_members."""
    c = tles[tles.epoch <= t0]
    if len(c) == 0: return None
    if not chained:
        r = c.iloc[-1]; return sgp4_states(r.line1, r.line2, targets), (t0 - r.epoch).total_seconds() / 3600, np.ones(3), False, False, 1
    man = False; after = None; t2 = inp["t2"]; f = t2[(t2.norad == norad) & (t2.epoch <= t0) & (t2.epoch >= t0 - pd.Timedelta(hours=48))]
    score = float(np.nanmax(f.score_res.values)) if len(f) and np.isfinite(f.score_res.values).any() else 0.0
    if flags["T2"] and len(f) and f.flag.any(): after = f[f.flag].epoch.max(); man = True
    win = 12.0; phase = False; t1 = inp["t1"]; p = t1[(t1.norad == norad) & (t1.t_start <= t0) & (t1.t_end >= t0 - pd.Timedelta(hours=48))]
    if flags["T1"] and len(p) and (p.pred_all != "STABLE_ORBIT").any(): win = 6.0; phase = True
    cc = c[c.epoch >= t0 - pd.Timedelta(hours=win)]
    if after is not None and len(c[c.epoch >= after]): cc = cc[cc.epoch >= after] if len(cc[cc.epoch >= after]) else c[c.epoch >= after]
    if len(cc) == 0: cc = c.tail(1)
    cc = cc.tail(3); S = []; Wt = []
    for r in cc.itertuples():
        st = sgp4_states(r.line1, r.line2, targets)
        if np.isfinite(st).all(): S.append(st); Wt.append(np.exp(-(t0 - r.epoch).total_seconds() / 7200.0))
    if not S: return None
    Wt = np.array(Wt) / np.sum(Wt); fused = np.einsum("m,mtk->tk", Wt, np.stack(S)); age = float(np.sum(Wt * [(t0 - r.epoch).total_seconds() / 3600 for r in cc.itertuples()][-len(S):]))
    # sigma inflation from upstream tasks
    g = inp["g"]; infl = np.ones(3); bins = [0, 3, 6, 12, 1e9]; b = int(np.digitize(score, bins[1:-1]))
    for k, comp in enumerate(RTN):
        if flags["T2"]: infl[k] *= g[comp]["bins"][str(b)]
        if flags["T1"] and phase: infl[k] = max(infl[k], g[comp]["oracle"])
    return fused, age, infl, phase, man, len(S)

def sigma_for(inp, age_h, h_arr, infl, fixed, storm):
    """RTN sigma per target time [T,3] km."""
    cal = inp["calib"]; out = np.zeros((len(h_arr), 3))
    for k, comp in enumerate(RTN):
        if fixed: out[:, k] = cal[comp]["fixed"]
        else: out[:, k] = np.maximum(cal[comp]["age_a"] + cal[comp]["age_b"] * (age_h + h_arr), 1e-3) * inp["g"][comp]["base"] * infl[k] * storm
    return out

def cov_teme(state, sig):
    if not np.isfinite(state).all(): return np.diag(sig ** 2)
    R = rtn_basis(state[None, :3], state[None, 3:])[0]; return R.T @ np.diag(sig ** 2) @ R

# ------------------------------------------------------------------ per-origin run
def density(P):
    r = np.linalg.norm(P, axis=1); alt = r - C.RE_KM; lat = np.degrees(np.arcsin(np.clip(P[:, 2] / r, -1, 1)))
    H, _, _ = np.histogram2d(alt, lat, bins=[ALT_EDGES, LAT_EDGES]); return H / max(H.sum(), 1)

def run_origin(t0, traj, ids, grid, tles, inp, cfg, ref_events, pairs, timing):
    """cfg: dict(name, chained, T1,T2,T3,T4sig,T5,T6pc). Returns dict of per-task metrics and event table."""
    i0 = int(np.searchsorted(grid.values, np.datetime64(t0.tz_convert(None)))); i1 = min(i0 + int(HZ_H * 60) + 1, len(grid)); targets = grid[i0:i1]; h_arr = (targets - t0).total_seconds().values / 3600
    N = len(ids); F = np.full((N, len(targets), 6), np.nan); SIG = np.full((N, len(targets), 3), np.nan); ages = np.full(N, np.nan); nmem = np.zeros(N, int); phase_n = 0; man_n = 0
    # T3: storm factor from Ap forecast (chained) or none
    storm = 1.0; t3 = inp["t3"]; d0 = pd.Timestamp(t0.date())
    fc = t3[(t3.origin_date == d0) & (t3.method == "HistGB-AR30") & (t3.variable == "ap_avg") & (t3.horizon_d <= 2)]
    ap_fc = float(fc.pred.max()) if len(fc) else np.nan
    if cfg["chained"] and cfg["T3"] and np.isfinite(ap_fc) and ap_fc >= 40: storm = 1.3
    tf = time.perf_counter()
    for k, n in enumerate(ids):
        if int(n) not in tles: continue
        res = forecast_one(tles[int(n)], t0, targets, cfg["chained"], inp, int(n), cfg)
        if res is None: continue
        st, age, infl, ph, mn, nm = res; F[k] = st; ages[k] = age; nmem[k] = nm; phase_n += ph; man_n += mn
        SIG[k] = sigma_for(inp, age, h_arr, infl, fixed=(not cfg["chained"]) or (not cfg["T4sig"]), storm=storm)
    timing["forecast_s"] += time.perf_counter() - tf
    R = traj[:, i0:i1, :]
    ages_prev = np.full(N, np.nan)
    for k, n in enumerate(ids):
        t = tles.get(int(n))
        if t is None: continue
        c = t[t.epoch <= t0]
        if len(c): ages_prev[k] = (t0 - c.epoch.iloc[-1]).total_seconds() / 3600
    def P_true(a, b, jt, h):
        """Config-independent reference covariance: calibrated sigma-age of both objects at the reference states."""
        sa = sigma_for(inp, ages_prev[a] if np.isfinite(ages_prev[a]) else 24.0, np.array([h]), np.ones(3), fixed=False, storm=1.0)[0]
        sb = sigma_for(inp, ages_prev[b] if np.isfinite(ages_prev[b]) else 24.0, np.array([h]), np.ones(3), fixed=False, storm=1.0)[0]
        return cov_teme(R[a, jt], sa) + cov_teme(R[b, jt], sb)
    # ---- T4 metrics vs reference
    t4 = {}
    for h in ERR_H:
        j = int(np.argmin(np.abs(h_arr - h))); ok = np.isfinite(F[:, j, :]).all(1) & np.isfinite(R[:, j, :]).all(1)
        if ok.sum() < 10: continue
        e = to_rtn(F[ok, j, :3] - R[ok, j, :3], R[ok, j, :3], R[ok, j, 3:]); e3 = np.linalg.norm(e, axis=1); z = e / SIG[ok, j, :]
        t4[h] = dict(n=int(ok.sum()), p50=float(np.median(e3)), p95=float(np.percentile(e3, 95)), cov95_t=float(np.mean(np.abs(z[:, 1]) <= 1.96)), nees=float(np.median(np.sum(z ** 2, 1)) / 3))
    # ---- T5: density at +24 h from the forecast vs reference; shell order from T5 output
    j = len(targets) - 1; ok = np.isfinite(F[:, j, :]).all(1) & np.isfinite(R[:, j, :]).all(1)
    tv = float(0.5 * np.abs(density(F[ok, j, :3]) - density(R[ok, j, :3])).sum()) if ok.sum() > 10 else np.nan
    t5 = inp["t5"]; pr = t5[(t5.t0 == t0) & (t5.h == 24)]
    if len(pr) == 0: pr = t5[t5.h == 24]
    dens = pr.groupby("alt_lo").pred_members.mean()
    s0 = R[:, 0, :]; alt0 = np.linalg.norm(s0[:, :3], axis=1) - C.RE_KM; shell = np.floor((alt0 - 200) / 50) * 50 + 200; d_obj = np.nan_to_num(dens.reindex(shell).to_numpy(), nan=0.0)
    # ---- T6: screening on forecast trajectories
    rng = np.random.default_rng(C.SEED + int(t0.value % 1000))
    if cfg["chained"] and cfg["T5"]: score = np.minimum(d_obj[pairs[:, 0]], d_obj[pairs[:, 1]])
    else: score = rng.random(len(pairs))
    k = len(pairs) if cfg.get("full_budget") else max(1, int(BUDGET * len(pairs))); order = np.argsort(-score, kind="stable")[:k]
    ts = time.perf_counter(); pred = W.screen(F, ids, targets, pairs[order], W.COARSE_KM, PRED_KM); timing["screen_s"] += time.perf_counter() - ts
    ref = ref_events[(ref_events.tca >= t0) & (ref_events.tca < targets[-1])].copy(); ref["key"] = list(zip(ref.i, ref.j))
    screened = set(map(tuple, np.sort(pairs[order], axis=1).tolist())); ids_pos = {int(n): q for q, n in enumerate(ids)}
    ref["screened"] = [tuple(sorted((ids_pos[a], ids_pos[b]))) in screened for a, b in ref.key]
    pcs = []
    for e in ref.itertuples():
        a, b = ids_pos[e.i], ids_pos[e.j]; jt = int(np.clip(round((e.tca - targets[0]).total_seconds() / 60), 0, len(targets) - 1)); h = max((e.tca - t0).total_seconds() / 3600, 0.01)
        pcs.append(pc_fast(np.array([e.rx, e.ry, e.rz]), np.array([e.vx, e.vy, e.vz]), P_true(a, b, jt, h), HBR_KM))
    ref["pc_true"] = pcs; ref["dangerous_pc"] = ref.pc_true >= PC_GATE
    rows = []
    for e in pred.itertuples():
        a, b = ids_pos[e.i], ids_pos[e.j]; jt = int(np.clip(round((e.tca - targets[0]).total_seconds() / 60), 0, len(targets) - 1))
        rel_r = np.array([e.rx, e.ry, e.rz]); rel_v = np.array([e.vx, e.vy, e.vz])
        P = cov_teme(F[a, jt], SIG[a, jt]) + cov_teme(F[b, jt], SIG[b, jt])
        pc = pc_fast(rel_r, rel_v, P, HBR_KM)
        m = ref[(ref.i == e.i) & (ref.j == e.j) & ((ref.tca - e.tca).abs() <= pd.Timedelta(minutes=MATCH_MIN))]
        ref_miss = float(m.miss_km.min()) if len(m) else np.nan
        # reference relative geometry at TCA for the true post-maneuver check
        if len(m):
            mm = m.loc[m.miss_km.idxmin()]; ref_r = np.array([mm.rx, mm.ry, mm.rz]); ref_v = np.array([mm.vx, mm.vy, mm.vz])
        else:
            jr = int(np.clip(round((e.tca - grid[0]).total_seconds() / 60), 0, len(grid) - 1)); ref_r = traj[a, jr, :3] - traj[b, jr, :3]; ref_v = traj[a, jr, 3:] - traj[b, jr, 3:]
            if not np.isfinite(ref_r).all(): ref_r = rel_r; ref_v = rel_v
        alert = (pc >= PC_GATE) if (cfg["chained"] and cfg["T6pc"]) else (e.miss_km < ALERT_KM)
        hh = max((e.tca - t0).total_seconds() / 3600, 0.01); Pt = P_true(a, b, jt, hh); pc_true = pc_fast(ref_r, ref_v, Pt, HBR_KM)
        dv = 0.0; post_ref_miss = float(np.linalg.norm(ref_r)); pc_post = pc_true
        if alert:
            mode = "pc" if (cfg["chained"] and cfg["T6pc"]) else "distance"
            dv, disp_m, _ = _avoid_disp(rel_r * 1000, rel_v * 1000, P, mode); post_ref_miss = float(np.linalg.norm(ref_r + disp_m / 1000)); pc_post = pc_fast(ref_r + disp_m / 1000, ref_v, Pt, HBR_KM)
        rows.append(dict(config=cfg["name"], t0=t0, i=e.i, j=e.j, tca=e.tca, pred_miss_km=e.miss_km, pc=pc, alert=bool(alert), ref_miss_km=ref_miss, ref_close=bool(np.isfinite(ref_miss) and ref_miss < CLOSE_KM),
                         ref_dangerous=bool(np.isfinite(ref_miss) and ref_miss < DANGER_KM), pc_true=pc_true, ref_dangerous_pc=bool(pc_true >= PC_GATE), dv_m_s=dv, post_ref_miss_km=post_ref_miss, pc_post=pc_post))
    ev = pd.DataFrame(rows)
    # ---- injected conjunctions (same forecast error, reference geometry shifted so that the event is genuinely close)
    inj_rows = []; rng_i = np.random.default_rng(C.SEED + 7 + int(t0.value % 1000))
    cand = ref[ref.tca >= t0 + pd.Timedelta(hours=1)]
    if len(cand):
        pick = cand.iloc[rng_i.choice(len(cand), min(INJ_PER_ORIGIN, len(cand)), replace=False)]
        targets_miss = np.exp(rng_i.uniform(np.log(INJ_MISS_KM[0]), np.log(INJ_MISS_KM[1]), len(pick)))
        for (e, tm) in zip(pick.itertuples(), targets_miss):
            a, b = ids_pos[e.i], ids_pos[e.j]; rel_r = np.array([e.rx, e.ry, e.rz]); rel_v = np.array([e.vx, e.vy, e.vz]); sfac = tm / max(np.linalg.norm(rel_r), 1e-6)
            shift = rel_r * (1 - sfac)                                     # r_b' = r_b + shift  ->  rel_r' = rel_r * sfac
            in_budget = tuple(sorted((a, b))) in screened
            jt = int(np.clip(round((e.tca - targets[0]).total_seconds() / 60), 0, len(targets) - 1)); hh = max((e.tca - t0).total_seconds() / 3600, 0.01)
            Pt = P_true(a, b, jt, hh); ref_r = rel_r * sfac; pc_true = pc_fast(ref_r, rel_v, Pt, HBR_KM)
            alert = False; pc = np.nan; pred_miss = np.nan; dv = 0.0; pc_post = pc_true; post_miss = float(np.linalg.norm(ref_r))
            if in_budget and np.isfinite(F[a]).all() and np.isfinite(F[b]).all():
                Fb = F[b].copy(); Fb[:, :3] += shift
                pe = W.screen(np.stack([F[a], Fb]), np.array([e.i, e.j]), targets, np.array([[0, 1]]), W.COARSE_KM, PRED_KM)
                if len(pe):
                    pe = pe.iloc[(pe.tca - e.tca).abs().argmin()]; prel_r = np.array([pe.rx, pe.ry, pe.rz]); prel_v = np.array([pe.vx, pe.vy, pe.vz]); pred_miss = float(pe.miss_km)
                    P = cov_teme(F[a, jt], SIG[a, jt]) + cov_teme(F[b, jt], SIG[b, jt]); pc = pc_fast(prel_r, prel_v, P, HBR_KM)
                    alert = (pc >= PC_GATE) if (cfg["chained"] and cfg["T6pc"]) else (pred_miss < ALERT_KM)
                    if alert:
                        dv, disp_m, _ = _avoid_disp(prel_r * 1000, prel_v * 1000, P, "pc" if (cfg["chained"] and cfg["T6pc"]) else "distance")
                        pc_post = pc_fast(ref_r + disp_m / 1000, rel_v, Pt, HBR_KM); post_miss = float(np.linalg.norm(ref_r + disp_m / 1000))
            inj_rows.append(dict(config=cfg["name"], t0=t0, i=e.i, j=e.j, tca=e.tca, ref_miss_km=float(tm), in_budget=in_budget, pred_miss_km=pred_miss, pc=pc, alert=bool(alert), pc_true=pc_true, dangerous=bool(pc_true >= PC_GATE), dv_m_s=dv, pc_post=pc_post, post_miss_km=post_miss, safe_after=bool(pc_post < PC_GATE)))
    inj = pd.DataFrame(inj_rows)
    n_ref = len(ref); n_ref_d = int((ref.miss_km < DANGER_KM).sum()); found = ev[ev.ref_close] if len(ev) else ev
    found_keys = set(zip(found.i, found.j)) if len(found) else set()
    rec_close = float(np.mean([k_ in found_keys for k_ in ref.key])) if n_ref else np.nan
    dang = ref[ref.miss_km < DANGER_KM]; found_d = ev[ev.ref_dangerous] if len(ev) else ev; fd_keys = set(zip(found_d.i, found_d.j)) if len(found_d) else set()
    rec_d = float(np.mean([k_ in fd_keys for k_ in dang.key])) if n_ref_d else np.nan
    al = ev[ev.alert] if len(ev) else ev
    n_ref_pc = int(ref.dangerous_pc.sum()); dpc = ref[ref.dangerous_pc]; fpc = ev[ev.ref_dangerous_pc] if len(ev) else ev; fpc_keys = set(zip(fpc.i, fpc.j)) if len(fpc) else set()
    rec_pc = float(np.mean([k_ in fpc_keys for k_ in dpc.key])) if n_ref_pc else np.nan
    t6 = dict(n_pairs_screened=int(k), n_pred_approaches=len(ev), n_ref_close=n_ref, n_ref_dangerous=n_ref_d, n_ref_dangerous_pc=n_ref_pc, recall_close=rec_close, recall_dangerous=rec_d, recall_dangerous_pc=rec_pc, n_alerts=len(al),
              alert_precision_close=float(al.ref_close.mean()) if len(al) else np.nan, alert_precision_pc=float(al.ref_dangerous_pc.mean()) if len(al) else np.nan, alerts_on_dangerous=int(al.ref_dangerous.sum()) if len(al) else 0,
              alerts_on_dangerous_pc=int(al.ref_dangerous_pc.sum()) if len(al) else 0, ref_screened_frac=float(ref.screened.mean()) if n_ref else np.nan)
    t7 = dict(n_maneuvers=len(al), dv_total_m_s=float(al.dv_m_s.sum()) if len(al) else 0.0, unnecessary=int((~al.ref_close).sum()) if len(al) else 0, unnecessary_pc=int((~al.ref_dangerous_pc).sum()) if len(al) else 0,
              dangerous_made_safe=int((al.ref_dangerous & (al.post_ref_miss_km >= DANGER_KM)).sum()) if len(al) else 0, missed_dangerous=int(n_ref_d - (al.ref_dangerous.sum() if len(al) else 0)),
              dangerous_pc_made_safe=int((al.ref_dangerous_pc & (al.pc_post < PC_GATE)).sum()) if len(al) else 0, missed_dangerous_pc=int(n_ref_pc - (al.ref_dangerous_pc.sum() if len(al) else 0)),
              sep_gain_km_median=float((al.post_ref_miss_km - al.ref_miss_km.fillna(al.pred_miss_km)).median()) if len(al) else np.nan)
    dg = inj[inj.dangerous] if len(inj) else inj
    tinj = dict(n=len(inj), n_dangerous=len(dg), in_budget=float(dg.in_budget.mean()) if len(dg) else np.nan, detected=int(dg.pred_miss_km.notna().sum()) if len(dg) else 0, alerted=int(dg.alert.sum()) if len(dg) else 0,
                safe_after=int(dg.safe_after.sum()) if len(dg) else 0, missed=int((~dg.alert).sum()) if len(dg) else 0, unnecessary=int((inj.alert & ~inj.dangerous).sum()) if len(inj) else 0, dv_m_s=float(inj.dv_m_s.sum()) if len(inj) else 0.0)
    return dict(t0=t0, config=cfg["name"], inj=tinj, inj_rows=inj, n_forecast=int(np.isfinite(F[:, 0, 0]).sum()), age_med_h=float(np.nanmedian(ages)), members_mean=float(nmem[nmem > 0].mean()) if (nmem > 0).any() else 1.0,
                phase_nonstable=phase_n, man_reset=man_n, ap_forecast=ap_fc, storm_factor=storm, t4=t4, t5_tv=tv, t6=t6, t7=t7), ev

# ------------------------------------------------------------------ main
CONFIGS = [dict(name="A0 independent (all pairs)", chained=False, T1=0, T2=0, T3=0, T4sig=0, T5=0, T6pc=0, full_budget=True),
           dict(name="A0 independent (random 20% budget)", chained=False, T1=0, T2=0, T3=0, T4sig=0, T5=0, T6pc=0),
           dict(name="A1 chained", chained=True, T1=1, T2=1, T3=1, T4sig=1, T5=1, T6pc=1),
           dict(name="A1 -T1 phase", chained=True, T1=0, T2=1, T3=1, T4sig=1, T5=1, T6pc=1),
           dict(name="A1 -T2 maneuver", chained=True, T1=1, T2=0, T3=1, T4sig=1, T5=1, T6pc=1),
           dict(name="A1 -T3 weather", chained=True, T1=1, T2=1, T3=0, T4sig=1, T5=1, T6pc=1),
           dict(name="A1 -T4 sigma (fixed)", chained=True, T1=1, T2=1, T3=1, T4sig=0, T5=1, T6pc=1),
           dict(name="A1 -T5 order (random)", chained=True, T1=1, T2=1, T3=1, T4sig=1, T5=0, T6pc=1),
           dict(name="A1 -T6 Pc (distance rule)", chained=True, T1=1, T2=1, T3=1, T4sig=1, T5=1, T6pc=0)]

def main(force=False):
    if stage_done("S8j_e2e") and not force: log.info("done"); return
    set_status("S8j_e2e", "running")
    try:
        with Timer() as T:
            traj, ids, grid = load_cohort(); log.info(f"cohort {len(ids)} objects, grid {grid[0]} -> {grid[-1]}")
            tles = load_tles(ids, grid[0] - pd.Timedelta(days=10), grid[-1]); log.info(f"TLEs for {len(tles)} objects")
            inp = task_inputs(ids, grid)
            first = np.argmax(np.isfinite(traj).all(2), axis=1); s0 = traj[np.arange(len(ids)), first, :]; pairs = W.apo_peri_pairs(s0)
            tr = time.perf_counter(); ref_events = W.screen(traj, ids, grid, pairs, W.COARSE_KM, CLOSE_KM); t_ref = time.perf_counter() - tr
            log.info(f"reference approaches < {CLOSE_KM} km: {len(ref_events)} (< {DANGER_KM} km: {int((ref_events.miss_km < DANGER_KM).sum())}), {len(pairs)} pairs, {t_ref:.1f} s")
            origins = pd.date_range(grid[0], grid[-1] - pd.Timedelta(hours=HZ_H), freq=f"{ORIGIN_STEP_H}h", tz="UTC")
            per = []; evs = []; timing = {c["name"]: dict(forecast_s=0.0, screen_s=0.0) for c in CONFIGS}
            for cfg in CONFIGS:
                for t0 in origins:
                    m, ev = run_origin(t0, traj, ids, grid, tles, inp, cfg, ref_events, pairs, timing[cfg["name"]]); per.append(m); evs.append(ev)
                log.info(f"{cfg['name']}: done {len(origins)} origins; forecast {timing[cfg['name']]['forecast_s']:.0f} s, screening {timing[cfg['name']]['screen_s']:.0f} s")
            EV = pd.concat([e for e in evs if len(e)], ignore_index=True); EV.to_parquet(C.RESULTS / "e2e_events.parquet", index=False)
            INJ = pd.concat([m["inj_rows"] for m in per if len(m["inj_rows"])], ignore_index=True); INJ.to_parquet(C.RESULTS / "e2e_injected.parquet", index=False)
            # ---- aggregate per configuration
            rows = []
            for cfg in CONFIGS:
                P = [m for m in per if m["config"] == cfg["name"]]
                r = dict(config=cfg["name"], n_origins=len(P), tle_age_med_h=float(np.nanmedian([m["age_med_h"] for m in P])), mtf_members=float(np.mean([m["members_mean"] for m in P])), phase_nonstable_obj=int(np.sum([m["phase_nonstable"] for m in P])), man_resets=int(np.sum([m["man_reset"] for m in P])), storm_factor=float(np.mean([m["storm_factor"] for m in P])))
                for h in ERR_H:
                    v = [m["t4"][h] for m in P if h in m["t4"]]
                    if v: r[f"T4_p50_{h:g}h"] = float(np.median([x["p50"] for x in v])); r[f"T4_p95_{h:g}h"] = float(np.median([x["p95"] for x in v])); r[f"T4_cov95_t_{h:g}h"] = float(np.mean([x["cov95_t"] for x in v])); r[f"T4_nees_{h:g}h"] = float(np.mean([x["nees"] for x in v]))
                r["T5_tv_24h"] = float(np.nanmean([m["t5_tv"] for m in P]))
                for kk in ("n_pairs_screened", "n_pred_approaches", "n_ref_close", "n_ref_dangerous", "n_ref_dangerous_pc", "n_alerts", "alerts_on_dangerous", "alerts_on_dangerous_pc"): r[f"T6_{kk}"] = float(np.sum([m["t6"][kk] for m in P]))
                for kk in ("recall_close", "recall_dangerous", "recall_dangerous_pc", "alert_precision_close", "alert_precision_pc", "ref_screened_frac"): r[f"T6_{kk}"] = float(np.nanmean([m["t6"][kk] for m in P]))
                for kk in ("n_maneuvers", "dv_total_m_s", "unnecessary", "unnecessary_pc", "dangerous_made_safe", "missed_dangerous", "dangerous_pc_made_safe", "missed_dangerous_pc"): r[f"T7_{kk}"] = float(np.sum([m["t7"][kk] for m in P]))
                for kk in ("n", "n_dangerous", "detected", "alerted", "safe_after", "missed", "unnecessary", "dv_m_s"): r[f"INJ_{kk}"] = float(np.sum([m["inj"][kk] for m in P]))
                r["INJ_in_budget"] = float(np.nanmean([m["inj"]["in_budget"] for m in P]))
                r["T7_sep_gain_km_median"] = float(np.nanmedian([m["t7"]["sep_gain_km_median"] for m in P])); r["time_forecast_s"] = timing[cfg["name"]]["forecast_s"]; r["time_screen_s"] = timing[cfg["name"]]["screen_s"]
                rows.append(r)
            dec = pd.DataFrame(rows); save_df(dec, "e2e_decision.csv")
            # ---- seven-task table: same for T1/T2/T3 across configs (no upstream), A0 vs A1 for T4-T7
            t1 = inp["t1"]; t1w = t1[(t1.t_end >= grid[0]) & (t1.t_start <= grid[-1]) & t1.silver.notna()]
            from sklearn.metrics import f1_score
            f1_t1 = float(f1_score(t1w.silver, t1w.pred_all, average="macro")) if len(t1w) else np.nan
            t2 = inp["t2"]; t2w = t2[(t2.epoch >= grid[0] - pd.Timedelta(days=3)) & (t2.epoch <= grid[-1])]; t2m = pd.read_csv(C.RESULTS / "t2_maneuver_metrics.csv"); t2ref = t2m[(t2m.satellite == "ALL") & (t2m.method == "SGP4-residual-T")]
            t3 = inp["t3"]; t3w = t3[(t3.method == "HistGB-AR30") & (t3.origin_date >= pd.Timestamp(grid[0].date())) & (t3.origin_date <= pd.Timestamp(grid[-1].date()))]
            t3mae = {v: float((g.pred - g.truth).abs().mean()) for v, g in t3w.groupby("variable")}
            A0 = dec[dec.config == "A0 independent (random 20% budget)"].iloc[0]; A0f = dec[dec.config == "A0 independent (all pairs)"].iloc[0]; A1 = dec[dec.config == "A1 chained"].iloc[0]
            t1row = dict(task="T1 Phase", metric="Macro-F1 vs rule labels (cohort, 3 seeds)", A0=f"{f1_t1:.3f}", A1=f"{f1_t1:.3f}", note="no upstream")
            t2row = dict(task="T2 Maneuver", metric="event F1 / F2 on Sentinel logs (test 2024-25H1)", A0=f"{float(t2ref.f1.iloc[0]):.3f} / {float(t2ref.f2.iloc[0]):.3f}", A1=f"{float(t2ref.f1.iloc[0]):.3f} / {float(t2ref.f2.iloc[0]):.3f}", note="no upstream")
            p1 = C.RESULTS / "uc_t1_cohort.csv"; p2 = C.RESULTS / "uc_t2_sentinel.csv"
            if p1.exists():
                u = pd.read_csv(p1); a0 = u[u.variant == "HistGB (A0)"].macro_f1.mean(); a1 = u[u.variant == "HistGB + T2 flags (A1)"].macro_f1.mean(); t1row.update(A0=f"{a0:.3f}", A1=f"{a1:.3f}", note="A1: + T2 maneuver-flag features")
            if p2.exists():
                u = pd.read_csv(p2); u = u[u.satellite == "ALL"]; a0 = u[u.variant == "SGP4-residual (A0)"].iloc[0]; a1 = u[u.variant == "SGP4-residual + T1 phase + T3 Ap (A1)"].iloc[0]; t2row.update(A0=f"{a0.f1:.3f} / {a0.f2:.3f}", A1=f"{a1.f1:.3f} / {a1.f2:.3f}", note="A1: threshold conditioned on T1 phase and T3 Ap")
            tab = [t1row, t2row,
                   dict(task="T3 Weather", metric="MAE Ap / F10.7, 1-3 d", A0=f"{t3mae.get('ap_avg', np.nan):.1f} / {t3mae.get('f107_obs', np.nan):.1f}", A1=f"{t3mae.get('ap_avg', np.nan):.1f} / {t3mae.get('f107_obs', np.nan):.1f}", note="no upstream"),
                   dict(task="T4 Forecast", metric="3D P50 / P95 at 24 h (km) vs L2 reference", A0=f"{A0['T4_p50_24h']:.2f} / {A0['T4_p95_24h']:.2f}", A1=f"{A1['T4_p50_24h']:.2f} / {A1['T4_p95_24h']:.2f}", note="reference = non-causal L2 product of the same TLEs (TLE-update consistency)"),
                   dict(task="T4 Uncertainty", metric="along-track 95% coverage / median NEES at 24 h", A0=f"{A0['T4_cov95_t_24h']:.2f} / {A0['T4_nees_24h']:.2f}", A1=f"{A1['T4_cov95_t_24h']:.2f} / {A1['T4_nees_24h']:.2f}", note="A0 fixed sigma; A1 sigma-age x g(T1,T2,T3)"),
                   dict(task="T5 Evolution", metric="TV of +24 h density vs reference", A0=f"{A0['T5_tv_24h']:.3f}", A1=f"{A1['T5_tv_24h']:.3f}", note="from the respective forecast"),
                   dict(task="T6 Warning", metric=f"recall of reference approaches < {CLOSE_KM:g} km at 20% budget (all pairs)", A0=f"{A0['T6_recall_close']:.3f} ({A0f['T6_recall_close']:.3f})", A1=f"{A1['T6_recall_close']:.3f}", note="A0 random order; A1 T5 order"),
                   dict(task="T6 Warning", metric="alerts / precision vs reference Pc >= 1e-4", A0=f"{int(A0['T6_n_alerts'])} / {A0['T6_alert_precision_pc']:.2f}", A1=f"{int(A1['T6_n_alerts'])} / {A1['T6_alert_precision_pc']:.2f}", note="A0 alerts on predicted miss < 5 km; A1 on predicted Pc >= 1e-4"),
                   dict(task="T7 Avoidance", metric="maneuvers / unnecessary (ref Pc < 1e-4) / total dv (m/s)", A0=f"{int(A0['T7_n_maneuvers'])} / {int(A0['T7_unnecessary_pc'])} / {A0['T7_dv_total_m_s']:.1f}", A1=f"{int(A1['T7_n_maneuvers'])} / {int(A1['T7_unnecessary_pc'])} / {A1['T7_dv_total_m_s']:.1f}", note="A0 distance rule; A1 Pc-gate"),
                   dict(task="T7 Avoidance", metric="natural dangerous events (ref Pc >= 1e-4): total / made safe / missed", A0=f"{int(A0['T6_n_ref_dangerous_pc'])} / {int(A0['T7_dangerous_pc_made_safe'])} / {int(A0['T7_missed_dangerous_pc'])}", A1=f"{int(A1['T6_n_ref_dangerous_pc'])} / {int(A1['T7_dangerous_pc_made_safe'])} / {int(A1['T7_missed_dangerous_pc'])}", note="reference geometry and covariance after maneuver"),
                   dict(task="T6+T7 injected", metric="injected dangerous conjunctions: total / in screening budget / alerted / safe after", A0=f"{int(A0['INJ_n_dangerous'])} / {A0['INJ_in_budget']:.2f} / {int(A0['INJ_alerted'])} / {int(A0['INJ_safe_after'])}", A1=f"{int(A1['INJ_n_dangerous'])} / {A1['INJ_in_budget']:.2f} / {int(A1['INJ_alerted'])} / {int(A1['INJ_safe_after'])}", note=f"{INJ_PER_ORIGIN}/origin, ref miss {INJ_MISS_KM[0]*1000:.0f}-{INJ_MISS_KM[1]*1000:.0f} m, same forecast error"),
                   dict(task="T6+T7 injected", metric="unnecessary maneuvers on injected non-dangerous / dv (m/s)", A0=f"{int(A0['INJ_unnecessary'])} / {A0['INJ_dv_m_s']:.1f}", A1=f"{int(A1['INJ_unnecessary'])} / {A1['INJ_dv_m_s']:.1f}", note=""),
                   dict(task="T7 Avoidance", metric=f"close events (ref miss < {DANGER_KM:g} km): total / separated / missed", A0=f"{int(A0['T6_n_ref_dangerous'])} / {int(A0['T7_dangerous_made_safe'])} / {int(A0['T7_missed_dangerous'])}", A1=f"{int(A1['T6_n_ref_dangerous'])} / {int(A1['T7_dangerous_made_safe'])} / {int(A1['T7_missed_dangerous'])}", note="distance view of the same events"),
                   dict(task="Cost", metric="screening time per origin (s) / pairs", A0=f"{A0['time_screen_s']/len(origins):.1f} / {int(A0['T6_n_pairs_screened']/len(origins))} (all: {A0f['time_screen_s']/len(origins):.1f} / {int(A0f['T6_n_pairs_screened']/len(origins))})", A1=f"{A1['time_screen_s']/len(origins):.1f} / {int(A1['T6_n_pairs_screened']/len(origins))}", note="")]
            save_df(pd.DataFrame(tab), "e2e_task_table.csv")
            abl = dec[dec.config.str.startswith("A1")][["config", "T4_p50_24h", "T4_cov95_t_24h", "T4_nees_24h", "T6_recall_close", "T6_recall_dangerous_pc", "T6_n_alerts", "T6_alert_precision_pc", "T7_n_maneuvers", "T7_unnecessary_pc", "T7_dv_total_m_s", "INJ_n_dangerous", "INJ_in_budget", "INJ_alerted", "INJ_safe_after", "INJ_missed", "INJ_unnecessary", "INJ_dv_m_s", "time_screen_s"]]
            save_df(abl, "e2e_ablation.csv")
            save_df(pd.DataFrame([dict(config=k, **v) for k, v in timing.items()] + [dict(config="reference screening (all pairs, 3 d)", forecast_s=0.0, screen_s=t_ref)]), "e2e_timing.csv")
            log.info("TASK TABLE:\n" + pd.DataFrame(tab).to_string(index=False)); log.info("ABLATION:\n" + abl.round(3).to_string(index=False))
        set_status("S8j_e2e", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("e2e failed"); set_status("S8j_e2e", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main(force="--force" in sys.argv)
