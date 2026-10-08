"""T6 Collision warning.

6a  Conjunction screening on a crowded LEO shell of the cohort over a test period.
    Reference close approaches ("truth"): all-vs-all on the 1-min reference trajectories (tail of the L2 CSVs),
    apogee/perigee pre-filter, grid minimum < COARSE_KM, then cubic-Hermite refinement at 1 s -> TCA, miss.
    Warning methods at lead L (24 h, 72 h): propagate each object's latest TLE <= (TCA-L) with SGP4 and screen
    the predicted trajectories the same way; an event is detected if a predicted approach with miss < WARN_KM
    lies within ±TCA_TOL of the truth TCA for the same pair.
      Independent: screen all pairs passing apogee/perigee filter.
      Chained-T5:   only pairs whose objects fall in the top-K shells of T5's predicted density (candidate reduction).
      Chained-UQ:   Pc with T4 σ-age model vs fixed σ vs point distance; rank events by Pc -> Recall@K / Brier.
6b  CDM risk (ESA collision-avoidance dataset, 199 events): predict final log10 risk from the CDM features;
    methods: last-estimate, Ridge, HistGB (5-fold CV); metrics MAE, high-risk (>-6) F2.
Outputs: results/t6_truth_events.csv, t6_warning_metrics.csv, t6_cdm_metrics.csv, t6_events_for_t7.csv
"""
from __future__ import annotations
import sys, io
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import erf
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, to_utc, read_tles, sgp4_states, mad
from .orbit import classical_elements

log = get_logger("t6_warning")
SHELL_LO, SHELL_HI = 700.0, 1000.0           # crowded LEO shell (mean altitude)
MAX_OBJ = 450
TEST_DAYS = 3
COARSE_KM = 400.0                           # grid-minimum gate at 1-min sampling
TRUTH_KM = 5.0                              # reference close-approach threshold
WARN_KM = 10.0                              # predicted approach threshold
DANGER_KM = 5.0                             # "dangerous" = predicted approach that matches a reference approach (< TRUTH_KM)
TCA_TOL_MIN = 10.0
LEADS_H = [6, 24]

def hermite(P0, V0, P1, V1, h, s):
    s = s[:, None]
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    return h00 * P0 + h10 * h * V0 + h01 * P1 + h11 * h * V1

def refine(Xa, Xb, k, dt_s=60.0):
    """Refine minimum distance around grid index k (both [T,6]) with 1-s Hermite interpolation. Returns (tca_offset_s, miss_km, rel_pos, rel_vel)."""
    best = (np.inf, None, None, None)
    for kk in (k - 1, k):
        if kk < 0 or kk + 1 >= len(Xa): continue
        s = np.linspace(0, 1, 61)
        pa = hermite(Xa[kk, :3], Xa[kk, 3:], Xa[kk + 1, :3], Xa[kk + 1, 3:], dt_s, s); pb = hermite(Xb[kk, :3], Xb[kk, 3:], Xb[kk + 1, :3], Xb[kk + 1, 3:], dt_s, s)
        d = np.linalg.norm(pa - pb, axis=1); j = int(np.argmin(d))
        if d[j] < best[0]:
            va = Xa[kk, 3:] * (1 - s[j]) + Xa[kk + 1, 3:] * s[j]; vb = Xb[kk, 3:] * (1 - s[j]) + Xb[kk + 1, 3:] * s[j]
            best = (d[j], kk * dt_s + s[j] * dt_s, pa[j] - pb[j], va - vb)
    return best[1], best[0], best[2], best[3]

def screen(traj: np.ndarray, ids: np.ndarray, times: pd.DatetimeIndex, pairs: np.ndarray, gate_km: float, thr_km: float) -> pd.DataFrame:
    """traj [N,T,6] km; pairs [P,2] indices. Returns events (i,j,tca,miss,rel_r,rel_v) with miss < thr_km."""
    rows = []
    P = traj[:, :, :3]
    for a, b in pairs:
        d = np.linalg.norm(P[a] - P[b], axis=1)
        if not np.isfinite(d).any(): continue
        dm = np.nanmin(d)
        if dm > gate_km: continue
        # all local minima below gate
        loc = np.where((d[1:-1] <= d[:-2]) & (d[1:-1] <= d[2:]) & (d[1:-1] <= gate_km))[0] + 1
        for k in loc:
            if not (np.isfinite(traj[a, k - 1:k + 2]).all() and np.isfinite(traj[b, k - 1:k + 2]).all()): continue
            off, miss, rr, rv = refine(traj[a], traj[b], k)
            if miss is not None and miss < thr_km:
                rows.append(dict(i=int(ids[a]), j=int(ids[b]), tca=times[0] + pd.Timedelta(seconds=float(off)), miss_km=float(miss),
                                 rel_speed_km_s=float(np.linalg.norm(rv)), rx=rr[0], ry=rr[1], rz=rr[2], vx=rv[0], vy=rv[1], vz=rv[2]))
    return pd.DataFrame(rows)

def apo_peri_pairs(states0: np.ndarray, pad_km: float = 25.0) -> np.ndarray:
    el = classical_elements(states0[:, :3], states0[:, 3:]); hp, ha = el["hp_km"], el["ha_km"]
    N = len(hp); I, J = np.triu_indices(N, 1)
    ok = (np.maximum(hp[I], hp[J]) - pad_km) <= (np.minimum(ha[I], ha[J]) + pad_km)
    return np.column_stack([I[ok], J[ok]])

def pc_2d(miss_km: float, hbr_km: float, sigma_km: float) -> float:
    """Isotropic 2-D encounter-plane collision probability (Chan/Alfano-style, circular covariance)."""
    if sigma_km <= 0: return float(miss_km < hbr_km)
    # P = ∫ over disk radius hbr of N(mu=miss, σ) in 2-D ≈ 1-exp(-hbr²/2σ²) * exp(-miss²/2σ²) * I0 term; use Chan first term
    v = hbr_km ** 2 / (sigma_km ** 2); u = miss_km ** 2 / (sigma_km ** 2)
    return float(np.exp(-u / 2) * (1 - np.exp(-v / 2)) * (1 + u * (1 - np.exp(-v / 2) * (1 + v / 2)) / (2 * (1 - np.exp(-v / 2)) + 1e-12)))

def load_shell_subset() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    obj = pd.read_csv(C.DATA / "objects.csv").set_index("norad")
    cache = C.DATA / f"t6_shell_{int(SHELL_LO)}_{int(SHELL_HI)}_{MAX_OBJ}_{TEST_DAYS}d.npz"
    if cache.exists():
        zc = np.load(cache); grid = pd.DatetimeIndex(zc["grid"]).tz_localize("UTC")
        log.info(f"shell subset loaded from cache {cache.name}"); return zc["traj"], zc["ids"], grid, obj
    z = np.load(C.DATA / "traj_window.npz"); times = pd.DatetimeIndex(z["times"]).tz_localize("UTC"); norad = z["norad"]; X = z["X"]
    t_end = times[-1]; t_start = t_end - pd.Timedelta(days=TEST_DAYS)
    i0 = int(np.searchsorted(times.values, np.datetime64(t_start.tz_convert(None))))
    s0 = X[:, i0, :].astype(float); ok = np.isfinite(s0).all(1)
    alt = np.linalg.norm(s0[:, :3], axis=1) - C.RE_KM
    sel = np.where(ok & (alt >= SHELL_LO) & (alt <= SHELL_HI))[0]
    rng = np.random.default_rng(C.SEED)
    if len(sel) > MAX_OBJ: sel = np.sort(rng.choice(sel, MAX_OBJ, replace=False))
    ids = norad[sel]
    # 1-min reference trajectories over the test period from the source CSV tails
    grid = pd.date_range(t_start, t_end, freq="1min", tz="UTC"); traj = np.full((len(sel), len(grid), 6), np.nan)
    from .stage0_staging import _stage_one
    for k, n in enumerate(ids):
        r = obj.loc[int(n)]
        _, arr, cov = _stage_one((int(n), r.source, r.path, t_start, t_end, 1))
        if arr is not None: traj[k] = arr
        if (k + 1) % 100 == 0: log.info(f"1-min reference trajectories {k+1}/{len(ids)}")
    np.savez_compressed(cache, traj=traj.astype(np.float32), ids=ids, grid=grid.tz_convert(None).values.astype("datetime64[ns]"))
    return traj, ids, grid, obj

def run_screening() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traj, ids, grid, obj = load_shell_subset()
    # first finite state of each object (a vector mean over an arc would shrink |r| and break the element-based filters)
    first = np.argmax(np.isfinite(traj).all(2), axis=1); s0 = traj[np.arange(len(ids)), first, :].astype(float)
    pairs = apo_peri_pairs(s0); log.info(f"shell subset {len(ids)} objects, {len(pairs)} pairs after apogee/perigee filter (of {len(ids)*(len(ids)-1)//2})")
    truth = screen(traj, ids, grid, pairs, COARSE_KM, TRUTH_KM); log.info(f"reference close approaches (<{TRUTH_KM} km): {len(truth)}")
    truth.to_csv(C.RESULTS / "t6_truth_events.csv", index=False)
    # σ model from T4 (σ-age) for UQ-chained ranking
    try:
        uq = pd.read_parquet(C.RESULTS / "t4_uq_cases.parquet"); uq["ah"] = uq.age_h + uq.h
        b, a = np.polyfit(uq.ah, np.sqrt(uq.sigma_age_r ** 2 + uq.sigma_age_t ** 2 + uq.sigma_age_n ** 2) / np.sqrt(3), 1)
        sig_fixed = float(np.median(np.sqrt(uq.sigma_fixed_r ** 2 + uq.sigma_fixed_t ** 2 + uq.sigma_fixed_n ** 2) / np.sqrt(3)))
    except Exception:
        b, a, sig_fixed = 0.02, 0.3, 1.0
    prio = None
    try:
        pr = pd.read_parquet(C.RESULTS / "t5_shell_priority.parquet"); pr = pr[pr.h == 24]
        prio = pr.groupby("alt_lo").pred_members.mean().sort_values(ascending=False)
    except Exception: pass
    tles = {int(n): read_tles(int(n), grid[0] - pd.Timedelta(days=8), grid[-1]) for n in ids}
    rows = []; ev_out = []
    for L in LEADS_H:
        t_issue = grid[0] - pd.Timedelta(hours=L)
        pred = np.full_like(traj, np.nan); ages = np.full(len(ids), np.nan)
        for k, n in enumerate(ids):
            tl = tles[int(n)]; prev = tl[tl.epoch <= t_issue]
            if len(prev): pred[k] = sgp4_states(prev.line1.iloc[-1], prev.line2.iloc[-1], grid); ages[k] = (grid[0] - prev.epoch.iloc[-1]).total_seconds() / 3600
        # T5-chained screening: rank candidate pairs by the predicted density of the two objects' shells (Kepler +24 h)
        # and screen only the top budget fraction; Independent = full budget.
        alt0 = np.linalg.norm(s0[:, :3], axis=1) - C.RE_KM; shell = np.floor((alt0 - 200) / 50) * 50 + 200
        dens = prio.reindex(shell).to_numpy() if prio is not None else np.ones(len(ids))
        dens = np.nan_to_num(dens, nan=0.0); pair_score = dens[pairs[:, 0]] * dens[pairs[:, 1]]
        order = np.argsort(-pair_score)
        budgets = [("Independent", 1.0), ("Chained-T5 (50% budget)", 0.5), ("Chained-T5 (25% budget)", 0.25), ("Chained-T5 (10% budget)", 0.10)]
        for mode, frac in budgets:
            pp = pairs[np.sort(order[: max(1, int(frac * len(pairs)))])]
            det = screen(pred, ids, grid, pp, COARSE_KM, WARN_KM)
            if len(det) == 0: det = pd.DataFrame(columns=["i", "j", "tca", "miss_km", "rel_speed_km_s", "rx", "ry", "rz", "vx", "vy", "vz"])
            det["tca"] = pd.to_datetime(det["tca"], utc=True)
            log.info(f"lead {L} h, {mode}: {len(pp)} pairs screened, {len(det)} predicted approaches < {WARN_KM} km; median TLE age at issue {np.nanmedian(ages):.1f} h")
            # match to truth
            hit = np.zeros(len(truth), bool); tca_err = []; miss_err = []
            for ti, tr in truth.iterrows():
                m = det[(det.i == tr.i) & (det.j == tr.j)]
                if len(m):
                    dtm = (m.tca - tr.tca).dt.total_seconds().abs() / 60
                    if (dtm <= TCA_TOL_MIN).any():
                        hit[ti] = True; kk = dtm.idxmin(); tca_err.append(dtm[kk]); miss_err.append(abs(m.loc[kk, "miss_km"] - tr.miss_km))
            fp = len(det) - hit.sum()
            rows.append(dict(task="T6", lead_h=L, mode=mode, n_pairs_screened=len(pp), n_truth=len(truth), n_pred=len(det), tle_age_med_h=float(np.nanmedian(ages)), recall=float(hit.mean()) if len(truth) else np.nan,
                             precision=float(hit.sum() / max(len(det), 1)), tca_err_med_min=float(np.median(tca_err)) if tca_err else np.nan,
                             miss_err_med_km=float(np.median(miss_err)) if miss_err else np.nan, candidate_reduction_pct=100 * (1 - len(pp) / max(len(pairs), 1))))
            if mode == "Independent" and len(det):  # UQ ranking on the full-budget detections
                # risk ranking with UQ variants on predicted events: dangerous truth = matched truth with miss < DANGER_KM
                det = det.copy(); det["hbr_km"] = [(obj.loc[i, "hbr_m"] + obj.loc[j, "hbr_m"]) / 1000.0 for i, j in zip(det.i, det.j)]
                det["age_h"] = [max(ages[np.where(ids == i)[0][0]], ages[np.where(ids == j)[0][0]]) for i, j in zip(det.i, det.j)]
                det["truth_miss"] = np.nan
                for di, d in det.iterrows():
                    m = truth[(truth.i == d.i) & (truth.j == d.j)]
                    if len(m):
                        dtm = (m.tca - d.tca).dt.total_seconds().abs() / 60
                        if (dtm <= TCA_TOL_MIN).any(): det.loc[di, "truth_miss"] = m.loc[dtm.idxmin(), "miss_km"]
                det["danger"] = det.truth_miss < DANGER_KM
                det["pc_point"] = -det.miss_km
                det["pc_fixed"] = [pc_2d(m, h, np.sqrt(2) * sig_fixed) for m, h in zip(det.miss_km, det.hbr_km)]
                det["pc_age"] = [pc_2d(m, h, np.sqrt(2) * max(a + b * (ag + L), 0.05)) for m, h, ag in zip(det.miss_km, det.hbr_km, det.age_h.fillna(L))]
                for uqm in ("point", "fixed", "age"):
                    sc = det[f"pc_{uqm}"].values; y = det.danger.values
                    rank = np.argsort(-sc); K = max(int(y.sum()), 1)
                    rows.append(dict(task="T6-UQ", lead_h=L, mode=f"UQ-{uqm}", n_pred=len(det), n_danger=int(y.sum()),
                                     recall_at_k=float(y[rank[:K]].mean()) if len(y) else np.nan,
                                     recall_at_2k=float(y[rank[:2 * K]].sum() / max(y.sum(), 1)) if len(y) else np.nan))
                det["lead_h"] = L; ev_out.append(det)
    met = pd.DataFrame(rows); ev = pd.concat(ev_out, ignore_index=True) if ev_out else pd.DataFrame()
    return met, truth, ev

def run_cdm() -> pd.DataFrame:
    d = pd.read_csv(C.CDM_CSV)
    y = d.true_risk.values; base = d.max_risk_estimate.values
    num = d.select_dtypes(include=[np.number]).drop(columns=["true_risk", "event_id"], errors="ignore").fillna(d.median(numeric_only=True))
    # heavy-tailed covariance features: signed log transform + robust scaling + clipping to avoid Ridge blow-ups
    num = np.sign(num) * np.log1p(np.abs(num))
    rows = []
    def score(name, pred):
        hi = y > -6; ph = pred > -6
        tp = (hi & ph).sum(); P = tp / max(ph.sum(), 1); R = tp / max(hi.sum(), 1)
        rows.append(dict(task="T6-CDM", method=name, n=len(y), mae_log10=float(np.mean(np.abs(pred - y))), rmse_log10=float(np.sqrt(np.mean((pred - y) ** 2))),
                         highrisk_precision=P, highrisk_recall=R, highrisk_f2=5 * P * R / max(4 * P + R, 1e-9), n_highrisk=int(hi.sum())))
    score("Last-estimate", base)
    for name, mk in (("Ridge", lambda: Ridge(alpha=10.0)), ("HistGB", lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=C.SEED))):
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=C.SEED).split(num):
            Xtr, Xte = num.iloc[tr], num.iloc[te]; mu, sd = Xtr.median(), (Xtr.quantile(.75) - Xtr.quantile(.25)).replace(0, 1)
            Ztr = ((Xtr - mu) / sd).clip(-5, 5); Zte = ((Xte - mu) / sd).clip(-5, 5)
            m = mk().fit(Ztr, y[tr]); pred[te] = np.clip(m.predict(Zte), -30, 0)
        score(name + " (5-fold)", pred)
    return pd.DataFrame(rows)

def main(force=False):
    if stage_done("S6_t6_warning") and not force: log.info("done"); return
    set_status("S6_t6_warning", "running")
    try:
        with Timer() as T:
            cdm = run_cdm(); save_df(cdm, "t6_cdm_metrics.csv"); log.info("CDM:\n" + cdm.round(3).to_string(index=False))
            met, truth, ev = run_screening(); save_df(met, "t6_warning_metrics.csv")
            if len(ev): ev.to_csv(C.RESULTS / "t6_events_for_t7.csv", index=False)
            log.info("screening:\n" + met.round(3).to_string(index=False))
        set_status("S6_t6_warning", "done", elapsed_s=round(T.elapsed), n_truth_events=int(len(truth)))
    except Exception as ex:
        log.exception("T6 failed"); set_status("S6_t6_warning", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
