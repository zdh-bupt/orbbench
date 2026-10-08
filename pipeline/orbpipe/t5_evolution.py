"""T5 Population / constellation evolution: forecast the cohort's altitude×latitude density at +1/6/24 h.

Closed cohort at each origin t0 (members with finite reference state at t0 and at t0+h).
Grid: altitude 200..2000 km / 50 km (36) × geocentric latitude 10° (18); frame-independent bins
(TEME longitude is not Earth-fixed, so longitude is not binned).
Methods (causal): Persistence (density at t0); Kepler (two-body from each member's t0 state);
                  SGP4-prev (each member's latest TLE <= t0 propagated to t0+h).
Reference: MTF/nearest L2 trajectories at t0+h (pseudo-truth; noted in report).
Metrics: density mass overlap % = 100·Σmin(p,q); total variation; regional relative MAE (2×2 pooled);
         hotspot F1 (top-10% cells); shell-ranking Spearman (altitude marginal).
Output: results/t5_evolution_metrics.csv, results/t5_shell_priority.parquet (per origin: predicted members per 50-km shell,
        used by T6 chained screening to prioritise candidate pairs).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, to_utc, read_tles, sgp4_states
from .orbit import kepler_propagate

log = get_logger("t5_evolution")
ALT_EDGES = np.arange(200, 2001, 50.0); LAT_EDGES = np.arange(-90, 91, 10.0)
HORIZONS_H = [1, 6, 24]
ORIGIN_STEP_H = 12

def bin_density(pos_km: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(pos_km, axis=1); alt = r - C.RE_KM; lat = np.degrees(np.arcsin(np.clip(pos_km[:, 2] / r, -1, 1)))
    H, _, _ = np.histogram2d(alt, lat, bins=[ALT_EDGES, LAT_EDGES])
    return H

def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    p = pred / max(pred.sum(), 1); q = truth / max(truth.sum(), 1)
    overlap = 100 * np.minimum(p, q).sum(); tv = 0.5 * np.abs(p - q).sum()
    P2 = pred.reshape(18, 2, 9, 2).sum((1, 3)); Q2 = truth.reshape(18, 2, 9, 2).sum((1, 3))
    m = Q2 > 0; rmae = float(np.mean(np.abs(P2[m] - Q2[m]) / Q2[m])) if m.any() else np.nan
    k = max(int(0.1 * truth.size), 1); tq = np.argsort(truth.ravel())[-k:]; tp = np.argsort(pred.ravel())[-k:]
    inter = len(set(tq) & set(tp)); f1 = 2 * inter / (2 * k)
    sp = spearmanr(pred.sum(1), truth.sum(1)).statistic
    return dict(overlap_pct=float(overlap), tv=float(tv), regional_rel_mae=rmae, hotspot_f1=float(f1), shell_spearman=float(sp))

def main(force=False):
    if stage_done("S5_t5_evolution") and not force: log.info("done"); return
    set_status("S5_t5_evolution", "running")
    try:
        with Timer() as T:
            z = np.load(C.DATA / "traj_window.npz"); times = pd.DatetimeIndex(z["times"]).tz_localize("UTC"); norad = z["norad"]; X = z["X"]
            obj = pd.read_csv(C.DATA / "objects.csv").set_index("norad")
            step = int(60 / C.STAGE_CADENCE_MIN)
            origins = np.arange(0, len(times) - 24 * step, ORIGIN_STEP_H * step)
            # TLEs per object (window ± 5 d) for SGP4-prev
            tles = {int(n): read_tles(int(n), times[0] - pd.Timedelta(days=5), times[-1]) for n in norad}
            rows = []; prio = []
            for oi, i0 in enumerate(origins):
                t0 = times[i0]; s0 = X[:, i0, :]
                for h in HORIZONS_H:
                    i1 = i0 + h * step; s1 = X[:, i1, :]
                    ok = np.isfinite(s0).all(1) & np.isfinite(s1).all(1)
                    if ok.sum() < 100: continue
                    truth = bin_density(s1[ok, :3]); preds = {"Persistence": bin_density(s0[ok, :3])}
                    rk, _ = kepler_propagate(s0[ok, :3].astype(float), s0[ok, 3:].astype(float), np.full(ok.sum(), h * 3600.0))
                    preds["Kepler"] = bin_density(rk)
                    sg = np.full((ok.sum(), 3), np.nan)
                    for j, n in enumerate(norad[ok]):
                        tl = tles[int(n)]; prev = tl[tl.epoch <= t0]
                        if len(prev):
                            st = sgp4_states(prev.line1.iloc[-1], prev.line2.iloc[-1], [times[i1]])[0]
                            if np.isfinite(st).all(): sg[j] = st[:3]
                    okg = np.isfinite(sg).all(1)
                    preds["SGP4-prev"] = bin_density(sg[okg]) * (ok.sum() / max(okg.sum(), 1))
                    for m, p in preds.items():
                        rows.append(dict(task="T5", t0=t0, h=h, method=m, n_members=int(ok.sum()), **metrics(p, truth)))
                    # shell priority from the best causal physics method (Kepler) for T6
                    shell = preds["Kepler"].sum(1)
                    prio.append(pd.DataFrame({"t0": t0, "h": h, "alt_lo": ALT_EDGES[:-1], "pred_members": shell, "true_members": truth.sum(1)}))
                if (oi + 1) % 50 == 0: log.info(f"origins {oi+1}/{len(origins)}")
            met = pd.DataFrame(rows); save_df(met, "t5_evolution_metrics.csv")
            pd.concat(prio, ignore_index=True).to_parquet(C.RESULTS / "t5_shell_priority.parquet", index=False)
            agg = met.groupby(["method", "h"])[["overlap_pct", "tv", "regional_rel_mae", "hotspot_f1", "shell_spearman"]].mean().round(3)
            log.info("mean over origins:\n" + agg.to_string())
            try:
                rep = pd.read_csv(C.REPORTED["evolution_voxel"]); save_df(rep, "t5_evolution_reported.csv")
            except Exception as ex: log.warning(f"reported import failed: {ex}")
        set_status("S5_t5_evolution", "done", elapsed_s=round(T.elapsed), n_rows=int(len(met)))
    except Exception as ex:
        log.exception("T5 failed"); set_status("S5_t5_evolution", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
