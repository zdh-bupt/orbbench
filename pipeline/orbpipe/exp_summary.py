"""Consolidation for §5.2 (task pipeline) and §5.3 (multi-source data).

E51  A0 vs A1 and per-task ablations with paired bootstrap 95% CIs, from the per-sample outputs of X1 (CDM events),
     X2 (Starlink forecasts, clustered by origin), X4 (Sentinel conjunctions, clustered by origin) and X3 (per-approach
     reach flags recomputed from the T5 ordering).
E52  Efficiency: candidate pairs and wall-clock of T6 screening on the 600-object LEO cohort, all pairs vs the
     T5-ordered 20 % budget; stage runtimes from status.json.
E53  Data ablation B0-B5 on decision-level metrics, assembled from X1/X2/X4, C1, T4 and the Starlink S4 table.
Outputs: results/e51_a0_a1_ci.csv, e52_efficiency.csv, e53_data_ablation.csv
"""
from __future__ import annotations
import sys, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df

log = get_logger("exp_summary")
NB = 2000

def paired_boot(stat_a, stat_b, N, clusters=None, n=NB, seed=C.SEED):
    """Paired bootstrap of stat_b(idx) - stat_a(idx) over items (or clusters of items)."""
    rng = np.random.default_rng(seed); clusters = np.arange(N) if clusters is None else np.asarray(clusters)
    uc, inv = np.unique(clusters, return_inverse=True); members = [np.where(inv == k)[0] for k in range(len(uc))]
    d0 = stat_b(np.arange(N)) - stat_a(np.arange(N)); ds = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(uc), len(uc)); idx = np.concatenate([members[k] for k in pick])
        ds[i] = stat_b(idx) - stat_a(idx)
    lo, hi = np.nanpercentile(ds, [2.5, 97.5]); return float(d0), float(lo), float(hi)

def _row(example, ref, alt, metric, va, vb, d, lo, hi, unit=""):
    return dict(example=example, reference=ref, variant=alt, metric=metric, value_ref=va, value_variant=vb, diff=d, ci_lo=lo, ci_hi=hi, unit=unit, significant=bool(lo > 0 or hi < 0))

# ------------------------------------------------------------------ E51
def e51_x1(rows):
    c = pd.read_parquet(C.RESULTS / "x1_cdm_cases.parquet"); ref = "A1 full (T3+T5+T4 -> T6 Pc -> T7 Pc-gate)"
    A = c[c.chain == ref].sort_values("event").reset_index(drop=True); N = len(A)
    for alt in [k for k in c.chain.unique() if k != ref and "x0" not in k and "oracle" not in k and "no maneuver" not in k]:
        B = c[c.chain == alt].sort_values("event").reset_index(drop=True)
        for metric, fa, fb, unit in (
            ("missed dangerous events", lambda i: float(((~A.maneuvered) & A.dangerous).values[i].sum()), lambda i: float(((~B.maneuvered) & B.dangerous).values[i].sum()), "count"),
            ("unnecessary maneuvers", lambda i: float((A.maneuvered & ~A.dangerous).values[i].sum()), lambda i: float((B.maneuvered & ~B.dangerous).values[i].sum()), "count"),
            ("total delta-v", lambda i: float(A.dv.values[i].sum()), lambda i: float(B.dv.values[i].sum()), "m/s"),
            ("safety rate (true Pc<1e-4)", lambda i: float((A.pc_post.values[i] < 1e-4).mean()), lambda i: float((B.pc_post.values[i] < 1e-4).mean()), "")):
            d, lo, hi = paired_boot(fa, fb, N); rows.append(_row("X1 CDM warning->avoidance (199 events)", ref, alt.strip(), metric, fa(np.arange(N)), fb(np.arange(N)), d, lo, hi, unit))

def e51_x2(rows):
    c = pd.read_parquet(C.RESULTS / "x2_starlink_cases.parquet"); ref = "A1 full (T1+T2+T3 -> T4 reliability -> T6 triage)"
    for h in (4.0, 24.0):
        A = c[(c.chain == ref) & (c.h == h)].reset_index(drop=True); N = len(A); cl = pd.factorize(A.t0)[0]
        for alt in [k for k in c.chain.unique() if k != ref]:
            B = c[(c.chain == alt) & (c.h == h)].reset_index(drop=True); assert len(B) == N
            def auc(P): return lambda i: float(roc_auc_score(P.y.values[i], P.p.values[i])) if 0 < P.y.values[i].sum() < len(i) else np.nan
            def rec(P):
                def f(i):
                    y = P.y.values[i]; p = P.p.values[i]; k = max(1, int(0.3 * len(i))); top = np.argsort(-p)[:k]; return float(y[top].sum() / max(y.sum(), 1))
                return f
            for metric, fa, fb in (("AUC", auc(A), auc(B)), ("recall at 30% flag budget", rec(A), rec(B))):
                d, lo, hi = paired_boot(fa, fb, N, clusters=cl, n=600); rows.append(_row(f"X2 Starlink triage ({h:g} h)", ref, alt.strip(), metric, fa(np.arange(N)), fb(np.arange(N)), d, lo, hi))

def e51_x4(rows):
    c = pd.read_parquet(C.RESULTS / "x4_sentinel_cases.parquet"); ref = "A1 full (T2 detector -> T4 sigma -> T6 Pc alert -> T7 Pc-gate)"
    for h in (4.0, 24.0):
        for split, mask_fn in (("post-maneuver", lambda P: P.post_man.values), ("all", lambda P: np.ones(len(P), bool))):
            A = c[(c.chain == ref) & (c.h == h)].reset_index(drop=True); m = mask_fn(A); A = A[m].reset_index(drop=True); N = len(A); cl = A.origin.values
            for alt in [k for k in c.chain.unique() if k != ref and "perfect T4" not in k and "no maneuver" not in k]:
                B = c[(c.chain == alt) & (c.h == h)].reset_index(drop=True); B = B[m].reset_index(drop=True); assert len(B) == N
                dg = A.dangerous.values
                def safety(P): return lambda i: float((P.pc_post.values[i][dg[i]] < 1e-4).mean()) if dg[i].any() else np.nan
                def recall(P): return lambda i: float((P.alert.values[i] & dg[i]).sum() / max(dg[i].sum(), 1))
                def dv(P): return lambda i: float(P.dv.values[i].sum())
                def unn(P): return lambda i: float((P.man.values[i] & ~dg[i]).sum())
                for metric, fa, fb, unit in (("T7 safety on dangerous conjunctions", safety(A), safety(B), ""), ("T6 alert recall", recall(A), recall(B), ""), ("total delta-v", dv(A), dv(B), "m/s"), ("unnecessary maneuvers", unn(A), unn(B), "count")):
                    d, lo, hi = paired_boot(fa, fb, N, clusters=cl, n=800); rows.append(_row(f"X4 Sentinel T2->UQ->T6->T7 ({h:g} h, {split})", ref, alt.strip(), metric, fa(np.arange(N)), fb(np.arange(N)), d, lo, hi, unit))

def e51_x3(rows):
    from .chain_proof import load_leo_subset
    from . import t6_warning as W
    traj, ids, grid, obj = load_leo_subset()
    first = np.argmax(np.isfinite(traj).all(2), axis=1); s0 = traj[np.arange(len(ids)), first, :].astype(float)
    pairs = W.apo_peri_pairs(s0)
    t0 = time.perf_counter(); truth = W.screen(traj, ids, grid, pairs, W.COARSE_KM, 10.0); t_all = time.perf_counter() - t0
    alt0 = np.linalg.norm(s0[:, :3], axis=1) - C.RE_KM; shell = np.floor((alt0 - 200) / 50) * 50 + 200
    pr = pd.read_parquet(C.RESULTS / "t5_shell_priority.parquet"); pr = pr[pr.h == 24]; dens = pr.groupby("alt_lo").pred_members.mean(); d_obj = np.nan_to_num(dens.reindex(shell).to_numpy(), nan=0.0)
    pos = {int(n): k for k, n in enumerate(ids)}; pair_index = {(int(ids[a]), int(ids[b])): k for k, (a, b) in enumerate(pairs)}
    tp = np.array([pair_index.get((int(i), int(j)), pair_index.get((int(j), int(i)), -1)) for i, j in zip(truth.i, truth.j)])
    scores = {"T5 shell density (min)": np.minimum(d_obj[pairs[:, 0]], d_obj[pairs[:, 1]]), "random": np.random.default_rng(C.SEED).random(len(pairs))}
    budget = 0.2; k = max(1, int(budget * len(pairs))); reached = {}
    for name, sc in scores.items():
        rank = np.empty(len(pairs), int); rank[np.argsort(-sc, kind="stable")] = np.arange(len(pairs)); reached[name] = (rank[tp] < k) & (tp >= 0)
    N = len(truth); fa = lambda i: float(reached["random"][i].mean()); fb = lambda i: float(reached["T5 shell density (min)"][i].mean())
    d, lo, hi = paired_boot(fa, fb, N); rows.append(_row("X3 LEO evolution->warning (139 approaches, 20% budget)", "T5-ordered screening (A1)", "random order (A0)", "recall of reference approaches", fb(np.arange(N)), fa(np.arange(N)), -d, -hi, -lo))
    # E52 timing: screen only the top-20 % pairs
    order = np.argsort(-scores["T5 shell density (min)"], kind="stable")[:k]
    t0 = time.perf_counter(); W.screen(traj, ids, grid, pairs[order], W.COARSE_KM, 10.0); t_bud = time.perf_counter() - t0
    st = json.loads((C.ROOT / "status.json").read_text())["stages"]
    eff = pd.DataFrame([dict(setting="A0: all candidate pairs", n_objects=len(ids), n_pairs=len(pairs), screening_time_s=round(t_all, 1), recall=float(reached["random"].mean() * 0 + 1.0)),
                        dict(setting="A1: T5-ordered, 20% budget", n_objects=len(ids), n_pairs=k, screening_time_s=round(t_bud, 1), recall=fb(np.arange(N))),
                        dict(setting="A0: random 20% budget", n_objects=len(ids), n_pairs=k, screening_time_s=round(t_bud, 1), recall=fa(np.arange(N)))])
    stage = pd.DataFrame([dict(setting=f"stage {kk}", n_objects=np.nan, n_pairs=np.nan, screening_time_s=v.get("elapsed_s", np.nan), recall=np.nan) for kk, v in st.items()])
    save_df(pd.concat([eff, stage], ignore_index=True), "e52_efficiency.csv"); log.info("E52:\n" + eff.to_string(index=False))

# ------------------------------------------------------------------ E53
def e53():
    rows = []
    def add(config, data_removed, example, metric, full, ablated, note=""): rows.append(dict(config=config, data_removed=data_removed, example=example, metric=metric, B0_full=full, ablated=ablated, note=note))
    x1 = pd.read_csv(C.RESULTS / "x1_cdm_chain.csv").set_index(pd.read_csv(C.RESULTS / "x1_cdm_chain.csv").chain.str.strip())
    g = lambda k, col: float(x1.loc[k, col]) if k in x1.index else np.nan
    full = "A1 full (T3+T5+T4 -> T6 Pc -> T7 Pc-gate)"
    for k, cfg, rem in (("-T3 (no space weather)", "B2 No-weather", "space-weather indices"), ("-T5 (no object/orbit attributes)", "B3 No-physical", "object type, size, area-to-mass, orbit attributes"),
                        ("-T4 (fixed sigma, no uncertainty task)", "B4 No-precise", "operator covariances for sigma calibration"), ("A0 none (fixed sigma + distance rule)", "B5 TLE-only", "all auxiliary data")):
        for col, m in (("missed_dangerous", "missed dangerous events (of 56)"), ("unnecessary_maneuvers", "unnecessary maneuvers"), ("dv_total_m_s", "total delta-v (m/s)")):
            add(cfg, rem, "X1 CDM warning->avoidance", m, g(full, col), g(k, col))
    x2 = pd.read_csv(C.RESULTS / "x2_starlink_triage.csv"); x2["chain"] = x2.chain.str.strip()
    for h in (4.0, 24.0):
        d = x2[x2.h == h].set_index("chain"); f = "A1 full (T1+T2+T3 -> T4 reliability -> T6 triage)"
        add("B2 No-weather", "space-weather indices", f"X2 Starlink triage ({h:g} h)", "AUC", float(d.loc[f, "auc"]), float(d.loc["-T3", "auc"]))
        add("B1 No-maneuver", "maneuver/phase information (T1, T2 inputs)", f"X2 Starlink triage ({h:g} h)", "AUC", float(d.loc[f, "auc"]), float(d.loc["-T1 -T2 -T3 (A0: age only)", "auc"]), "removes T1, T2 and T3 inputs")
    x4 = pd.read_csv(C.RESULTS / "x4_sentinel_chain.csv"); x4["chain"] = x4.chain.str.strip()
    d = x4[(x4.h == 4.0) & (x4.split == "post-maneuver (logged, 48 h)")].set_index("chain")
    add("B0 with maneuver logs", "(adds logged maneuvers to sigma)", "X4 Sentinel post-maneuver (4 h)", "T7 safety on dangerous conjunctions", float(d.loc["A1 with perfect T2 (logged maneuvers)", "t7_safety_dangerous"]), float(d.loc["A1 full (T2 detector -> T4 sigma -> T6 Pc alert -> T7 Pc-gate)", "t7_safety_dangerous"]), "full = detector only; logs = B0 with maneuver records")
    add("B1 No-maneuver", "maneuver awareness (detector and logs)", "X4 Sentinel post-maneuver (4 h)", "T7 safety on dangerous conjunctions", float(d.loc["A1 full (T2 detector -> T4 sigma -> T6 Pc alert -> T7 Pc-gate)", "t7_safety_dangerous"]), float(d.loc["-T2 (sigma-age, maneuver-blind)", "t7_safety_dangerous"]))
    c1 = pd.read_csv(C.RESULTS / "c1_maneuver_sigma.csv"); c1 = c1[(c1.component == "t") & (c1.h.astype(str) == "24.0") & (c1.split == "post-maneuver (logged)")].set_index("model")
    add("B1 No-maneuver", "maneuver logs", "T4-UQ Sentinel post-maneuver (24 h)", "along-track 95% coverage", float(c1.loc["sigma-age x g(logged maneuver)", "cov95"]), float(c1.loc["sigma-age x g(T2 score)", "cov95"]), "logs -> detector only")
    add("B1 No-maneuver", "maneuver awareness", "T4-UQ Sentinel post-maneuver (24 h)", "along-track 95% coverage", float(c1.loc["sigma-age x g(logged maneuver)", "cov95"]), float(c1.loc["sigma-age", "cov95"]))
    t4 = pd.read_csv(C.RESULTS / "t4_forecast_metrics.csv"); a = t4[t4.split == "all"].pivot_table(index="method", columns="h", values="3d_p50_km")
    for h in (1.0, 4.0, 24.0): add("B4 No-precise", "POD for corrector training", f"T4 Sentinel forecast ({h:g} h)", "3D P50 (km)", float(a.loc["HistGB-orbit", h]), float(a.loc["SGP4-MTF", h]), "full = POD-trained corrector; ablated = SGP4-MTF")
    u = pd.read_csv(C.RESULTS / "t4_uq_metrics.csv"); u = u[u.component == "t"].set_index("model")
    add("B4 No-precise", "POD for sigma calibration", "T4-UQ Sentinel (along-track)", "NEES", float(u.loc["σ-age", "nees"]), float(u.loc["σ-fixed", "nees"]))
    s4 = pd.read_csv(C.RESULTS / "s4_starlink_ablation.csv")
    for h in (4.0, 24.0):
        d = s4[s4.h == h].set_index("config"); f = [i for i in d.index if i.startswith("B0 full")][0]
        add("B1 No-maneuver", "maneuver awareness (gate / T1)", f"S4 Starlink forecast ({h:g} h)", "3D P90 (km)", float(d.loc[f, "p90"]), float(d.loc[[i for i in d.index if i.startswith("B1")][0], "p90"]))
        add("B5 TLE-only", "all auxiliary data", f"S4 Starlink forecast ({h:g} h)", "3D P90 (km)", float(d.loc[f, "p90"]), float(d.loc[[i for i in d.index if i.startswith("B5")][0], "p90"]))
    x3 = pd.read_csv(C.RESULTS / "x3_leo_chain.csv"); x3["chain"] = x3.chain.str.strip(); d = x3.set_index("chain")
    add("B5 TLE-only", "all auxiliary data", "X3 LEO screening (20% budget)", "recall of reference approaches", float(d.loc["A1 full (T5 order -> T6 Pc -> T7 Pc-gate)", "recall"]), float(d.loc["A0 none (random order + distance rule)", "recall"]))
    add("B5 TLE-only", "all auxiliary data", "X3 LEO screening (20% budget)", "total delta-v (m/s)", float(d.loc["A1 full (T5 order -> T6 Pc -> T7 Pc-gate)", "dv_total_m_s"]), float(d.loc["A0 none (random order + distance rule)", "dv_total_m_s"]))
    out = pd.DataFrame(rows); out["rel_change_pct"] = 100 * (out.ablated - out.B0_full) / out.B0_full.replace(0, np.nan); return out

def main(force=False):
    if stage_done("S8h_exp_summary") and not force: log.info("done"); return
    set_status("S8h_exp_summary", "running")
    try:
        with Timer() as T:
            rows = []
            e51_x1(rows); log.info("X1 CIs done"); e51_x2(rows); log.info("X2 CIs done"); e51_x4(rows); log.info("X4 CIs done"); e51_x3(rows); log.info("X3 CIs + timing done")
            ci = pd.DataFrame(rows); save_df(ci, "e51_a0_a1_ci.csv"); log.info("E51:\n" + ci.round(3).to_string(index=False))
            ab = e53(); save_df(ab, "e53_data_ablation.csv"); log.info("E53:\n" + ab.round(3).to_string(index=False))
        set_status("S8h_exp_summary", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("exp_summary failed"); set_status("S8h_exp_summary", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main(force="--force" in sys.argv)
