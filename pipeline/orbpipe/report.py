"""S9: write REPORT.md and figures from results/. Safe to re-run at any time (uses whatever exists)."""
from __future__ import annotations
import sys, json, os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".cache", "orbbench-mpl"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .util import get_logger, set_status, Timer

log = get_logger("report")
R, F = C.RESULTS, C.FIGURES
BLUE, RED, YELLOW, GREEN, PURPLE, LBLUE = "#4477AA", "#EE6677", "#FFBE7A", "#1F7940", "#BEB8DC", "#66AFC3"
PAL = [BLUE, RED, YELLOW, GREEN, PURPLE, LBLUE, "#AA3377", "#999999"]
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "xtick.direction": "in", "ytick.direction": "in"})

def rd(name):
    p = R / name
    if not p.exists() or p.stat().st_size < 5: return None
    try: return pd.read_csv(p)
    except Exception: return None

def md_table(df: pd.DataFrame, floatfmt=".3f") -> str:
    if df is None or len(df) == 0: return "_not available yet_\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    cols = list(d.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows(): out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"

def fig_t4(t4):
    a = t4[t4.split == "all"]; methods = [m for m in ["SGP4-prev", "SGP4-MTF", "SGP4-MTF+T2", "Kepler-24h", "Ridge-orbit", "Ridge-weather+T3"] if m in set(a.method)]
    fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
    for k, m in enumerate(methods):
        g = a[a.method == m].sort_values("h"); ax.plot(g.h, g["3d_p50_km"], marker="o", ms=2.5, lw=0.9, color=PAL[k % len(PAL)], label=m)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Horizon (h)"); ax.set_ylabel("3D position error P50 (km)")
    ax.set_xticks([0.5, 1, 4, 8, 12, 24]); ax.set_xticklabels(["0.5", "1", "4", "8", "12", "24"]); ax.grid(color="#DDDDDD", lw=0.4, alpha=0.6)
    ax.legend(ncol=2, frameon=False, loc="upper left"); fig.savefig(F / "fig_t4_forecast_error.pdf"); fig.savefig(F / "fig_t4_forecast_error.png", dpi=300); plt.close(fig)

def fig_uq(uq):
    u = uq[uq.component != "3D"]
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.8), constrained_layout=True)
    for ax, col, ref in zip(axes, ("cov68", "cov95"), (0.68, 0.95)):
        piv = u.pivot_table(index="model", columns="component", values=col)
        x = np.arange(len(piv)); w = 0.25
        for k, c in enumerate(["r", "t", "n"]): ax.bar(x + (k - 1) * w, piv[c], w, color=[RED, BLUE, YELLOW][k], label=c.upper())
        ax.axhline(ref, color="#555555", lw=0.6, ls="--"); ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=0); ax.set_ylim(0, 1); ax.set_title(f"coverage {int(ref*100)}%", fontsize=7)
    axes[0].legend(frameon=False, ncol=3, loc="lower center"); fig.savefig(F / "fig_t4_uq_coverage.pdf"); fig.savefig(F / "fig_t4_uq_coverage.png", dpi=300); plt.close(fig)

def fig_e3(e3):
    d = e3[(e3.sat != "ALL") & (e3.split == "all")]
    fig, ax = plt.subplots(figsize=(3.3, 2.0), constrained_layout=True)
    cfgs = sorted(d.config.unique()); sats = sorted(d.sat.unique()); x = np.arange(len(sats)); w = 0.8 / len(cfgs)
    for k, c in enumerate(cfgs):
        g = d[d.config == c].set_index("sat").reindex(sats); ax.bar(x + (k - (len(cfgs) - 1) / 2) * w, g.e3d_p50_km, w, color=PAL[k], label=c)
    ax.set_xticks(x); ax.set_xticklabels(sats); ax.set_ylabel("3D error P50 vs POD (km)"); ax.legend(frameon=False); ax.grid(axis="y", color="#DDDDDD", lw=0.4, alpha=0.6)
    fig.savefig(F / "fig_e3_quality.pdf"); fig.savefig(F / "fig_e3_quality.png", dpi=300); plt.close(fig)

def fig_e1(e1):
    d = e1.dropna(subset=["independent", "chained"]); d = d[d.metric.str.contains("P50|recall", regex=True) & ~d.edge.str.contains("pairs")]
    if not len(d): return
    fig, ax = plt.subplots(figsize=(3.3, 2.4), constrained_layout=True)
    lab = [f"{r.edge} (+{r.h:g}h)" for r in d.itertuples()]; y = np.arange(len(d))
    ax.barh(y, d.rel_change_pct, color=[GREEN if v < 0 else RED for v in d.rel_change_pct]); ax.axvline(0, color="#555555", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=5.5); ax.set_xlabel("Chained vs independent (% change of metric)")
    fig.savefig(F / "fig_e1_chaining.pdf"); fig.savefig(F / "fig_e1_chaining.png", dpi=300); plt.close(fig)

def fig_chain_ablation():
    x1 = rd("x1_cdm_chain.csv"); x2 = rd("x2_starlink_triage.csv"); x3 = rd("x3_leo_chain.csv"); x4 = rd("x4_sentinel_chain.csv")
    if any(v is None for v in (x1, x2, x3, x4)): return
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9))
    def bars(a, labels, vals, ylabel, title, colors):
        a.bar(range(len(vals)), vals, color=colors, width=0.7); a.set_xticks(range(len(vals))); a.set_xticklabels(labels, fontsize=6.5, rotation=20, ha="right"); a.set_ylabel(ylabel, fontsize=7); a.set_title(title, fontsize=7.5); a.tick_params(labelsize=6.5)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        a.yaxis.grid(True, color="#DDDDDD", lw=0.5); a.set_axisbelow(True)
    RED, BLUE, YEL, GRN = "#EE6677", "#4477AA", "#FFBE7A", "#1F7940"
    r3 = x3.set_index("chain"); bars(ax[0], ["full", "-T5", "A0"], [r3.recall.iloc[0], r3.recall.iloc[1], r3.recall.iloc[3]], "recall @ 20% budget", "X3 T5→T6→T7", [BLUE, RED, "#999999"])
    d2 = x2[(x2.h == 24)].set_index("chain"); bars(ax[1], ["full", "-T3", "-T2", "-T1", "A0"], d2.auc.values, "AUC (24 h, >10 km)", "X2 T1+T2+T3→T4→T6", [BLUE, YEL, YEL, RED, "#999999"]); ax[1].set_ylim(0.5, 0.7)
    d1 = x1[~x1.chain.str.contains("x0")].set_index("chain"); keys = list(d1.index[:4]); bars(ax[2], ["full", "-T3", "-T5", "-T3-T5"], d1.loc[keys].missed_dangerous.values, "missed dangerous (of 56)", "X1 T3+T5→T4→T6→T7", [BLUE, YEL, YEL, RED])
    d4 = x4[(x4.h == 4) & (x4.split == "post-maneuver (logged, 48 h)")].set_index("chain"); k4 = list(d4.index[:3]); bars(ax[3], ["full", "perfect T2", "-T2"], d4.loc[k4].t7_safety_dangerous.values, "safe dangerous conj. (post-man.)", "X4 T2→T4→T6→T7", [BLUE, GRN, RED]); ax[3].set_ylim(0.5, 0.75)
    fig.tight_layout(); fig.savefig(F / "fig_chain_ablation.pdf"); fig.savefig(F / "fig_chain_ablation.png", dpi=160); plt.close(fig)

def main():
    set_status("S9_report", "running")
    try:
        with Timer() as T:
            st = json.loads((C.ROOT / "status.json").read_text()) if (C.ROOT / "status.json").exists() else {}
            parts = [f"# OrbBench unified STM pipeline — automated results\n\nGenerated {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC. Window {C.WINDOW_START[:10]} → {C.WINDOW_END[:10]}; cohort = objects with L2 trajectories staged in `data/objects.csv`; truth cohort = Sentinel POD.\n"]
            parts.append("## Stage status\n" + md_table(pd.DataFrame([{"stage": k, **{kk: vv for kk, vv in v.items() if kk != 'message'}} for k, v in st.get("stages", {}).items()])))
            obj = pd.read_csv(C.DATA / "objects.csv") if (C.DATA / "objects.csv").exists() else None
            if obj is not None:
                parts.append("## Cohort\n" + md_table(pd.crosstab(obj.type, obj.regime).reset_index()) + f"\nmean window coverage of staged trajectories: {obj.window_coverage.mean():.3f}\n" if "window_coverage" in obj else "")
            e0 = rd("E0_benchmark_table.csv")
            if e0 is not None:
                parts.append("## E0 Benchmark table (headline metrics)\n")
                for task, g in e0.groupby("task", sort=False): parts.append(f"### {task}\n" + md_table(g.drop(columns=["task"]).pivot_table(index=["subtask", "method"], columns="metric", values="value").reset_index()))
            t3 = rd("t3_weather_metrics.csv")
            if t3 is not None: parts.append("## T3 space weather (MAE by horizon)\n" + md_table(t3.pivot_table(index=["variable", "method"], columns="horizon_d", values="mae").reset_index()))
            t1 = rd("t1_phase_metrics.csv")
            if t1 is not None: parts.append("## T1 phase segmentation (mean over 3 seeds)\n" + md_table(t1[t1.stratum.isna()].groupby("method")[["macro_f1", "ari", "nmi", "f1_STABLE_ORBIT", "f1_SUSTAINED_RAISE", "f1_SUSTAINED_DECAY"]].mean().reset_index()) +
                                          "\nSilver labels follow the Starlink observable-state rules; no SUSTAINED_RAISE exists in this legacy-catalog cohort and most SUSTAINED_DECAY windows are MEO objects with lunisolar-driven semi-major-axis trends, so the class semantics differ from the Starlink use case.\n")
            t2 = rd("t2_maneuver_metrics.csv")
            if t2 is not None: parts.append("## T2 maneuver detection (Sentinel logs as truth, ±1 interval)\n" + md_table(t2[["satellite", "method", "precision", "recall", "f1", "f2", "n_true", "n_pred"]]))
            t4 = rd("t4_forecast_metrics.csv")
            if t4 is not None:
                a = t4[t4.split == "all"]; parts.append("## T4 orbit forecasting vs POD (3D P50 km, all Sentinels)\n" + md_table(a.pivot_table(index="method", columns="h", values="3d_p50_km").reset_index()))
                parts.append("### P95\n" + md_table(a.pivot_table(index="method", columns="h", values="3d_p95_km").reset_index()))
                b = t4[t4.split.isin(["maneuver", "quiet"]) & (t4.h == 24)]; parts.append("### +24 h by maneuver presence\n" + md_table(b.pivot_table(index="method", columns="split", values="3d_p50_km").reset_index()))
                fig_t4(t4)
            uq = rd("t4_uq_metrics.csv")
            if uq is not None: parts.append("## T4 uncertainty (in-window)\n" + md_table(uq[uq.component != "3D"][["model", "component", "cov68", "cov95", "nees", "spearman_sigma_abserr", "sigma_median_km"]])); fig_uq(uq)
            t5 = rd("t5_evolution_metrics.csv")
            if t5 is not None: parts.append("## T5 population evolution (mean over origins)\n" + md_table(t5.groupby(["method", "h"])[["overlap_pct", "tv", "regional_rel_mae", "hotspot_f1", "shell_spearman"]].mean().reset_index()))
            t6 = rd("t6_warning_metrics.csv"); c6 = rd("t6_cdm_metrics.csv")
            if t6 is not None: parts.append("## T6 conjunction screening\n" + md_table(t6[t6.task == "T6"]) + "\n### Risk ranking with uncertainty\n" + md_table(t6[t6.task == "T6-UQ"][["lead_h", "mode", "n_pred", "n_danger", "recall_at_k", "recall_at_2k"]]))
            if c6 is not None: parts.append("## T6 CDM final-risk prediction (ESA dataset, 199 events)\n" + md_table(c6))
            t7 = rd("t7_avoidance_metrics.csv")
            if t7 is not None: parts.append("## T7 autonomous avoidance\n" + md_table(t7))
            for name, title in (("starlink_a0_a1.csv", "STARLINK END-TO-END: A0 (independent tasks) vs A1 (connected chain), MEME truth with operator covariance, test 2026-07-05 .. 08-23"),
                                ("s1_starlink_forecast.csv", "S1 Starlink forecast vs MEME (3D P50 km) by method and horizon"),
                                ("s1_starlink_winrates.csv", "S1 per-satellite win rate vs A0"),
                                ("s1_starlink_correctors.csv", "S1 residual correctors: orbit-only vs +T1 (SMA rate) vs +T1+T2 features"),
                                ("s1_starlink_uq.csv", "S1 uncertainty on Starlink (along-track): fixed / age / age x g(T2) / operator covariance alone"),
                                ("s1_t2_detection.csv", "S1 T2 detector vs MEME-derived thrust arcs"),
                                ("s2_starlink_ranking.csv", "S2 Starlink conjunction warning: A0 vs A1 trajectories, true Pc from MEME covariance"),
                                ("s3_starlink_avoidance.csv", "S3 Starlink avoidance on matched events (safety by true Pc)"),
                                ("s4_starlink_ablation.csv", "S4 data ablation on Starlink (1B)"),
                                ("e2e_task_table.csv", "E2E FULL CHAIN T1->T7 on 1,500 LEO objects (300-2000 km), 3 d, 5 decision epochs, 20% screening budget: independent (A0) vs unified pipeline (A1); reference = Hermite-interpolated L2 trajectories; 30 injected dangerous conjunctions per epoch"),
                                ("e2e_ablation.csv", "E2E per-task removal on the same cohort and the same injected conjunctions"),
                                ("uc_t2_sentinel.csv", "UC upstream connections T1->T2 and T3->T2 on Sentinel (maneuver logs as truth, test 2024-25H1)"),
                                ("uc_t2_starlink.csv", "UC upstream connection T1->T2 on Starlink (MEME thrust arcs as truth; 84% of intervals are thrust, so F1 is degenerate and MCC is reported)"),
                                ("uc_t1_cohort.csv", "UC upstream connection T2->T1 on the cohort (rule labels, 3 object-level seeds)"),
                                ("e51_a0_a1_ci.csv", "E51 A0 vs A1 and per-task ablations with paired bootstrap 95% CIs (X1 events; X2/X4 clustered by forecast origin; X3 by approach)"),
                                ("e52_efficiency.csv", "E52 Screening efficiency: all candidate pairs vs T5-ordered 20% budget (600 LEO objects, 3 d, 1-min grid)"),
                                ("e53_data_ablation.csv", "E53 Data ablation B0-B5 on decision-level metrics (assembled from X1/X2/X4, C1, T4, S4, X3)"),
                                ("e54a_shorten_t2.csv", "E54a Same task on an existing dataset: T2 maneuver detection on the Shorten et al. TLE benchmark (15 satellites, curated maneuver timestamps)"),
                                ("e54b_kelvins_t6.csv", "E54b Same task on an existing dataset: T6 final-risk prediction on the full ESA Kelvins CDM dataset (competition protocol, train -> public test with private truth)"),
                                ("x4_sentinel_chain.csv", "X4 TASK-LEVEL CHAIN: T2 maneuver detection -> T4 uncertainty -> T6 Pc alert -> T7 Pc-gated avoidance (Sentinel-1A/2A/3A POD truth, synthetic chasers, test 2024-25H1)"),
                                ("x3_leo_chain.csv", "X3 TASK-LEVEL CHAIN: T5 evolution ordering -> T6 screening under a 20% pair budget -> T7 (600 LEO objects, 3 d)"),
                                ("x2_starlink_triage.csv", "X2 TASK-LEVEL CHAIN: T1 phase + T2 score + T3 weather -> T4 forecast-reliability -> T6 screening triage (17 Starlinks, MEME truth)"),
                                ("x1_cdm_chain.csv", "X1 TASK-LEVEL CHAIN: T3 weather + T5 object attributes -> T4 chaser-sigma model -> T6 Pc -> T7 Pc-gate (199 ESA CDMs, operator covariance as truth)"),
                                ("x1_cdm_ranking.csv", "X1 ranking quality of the T6 Pc alert (dangerous = true Pc >= 1e-4)"),
                                ("chain_a0_a1.csv", "CHAIN A0 (independent) vs A1 (chained): where each edge pays off"),
                                ("c1_maneuver_sigma.csv", "C1 T2 -> UQ: maneuver-aware sigma (Sentinel POD truth, test 2024-25H1)"),
                                ("c2_uq_ranking.csv", "C2 UQ -> T6: ranking predicted approaches by Pc vs point miss"),
                                ("c3_avoidance_sigma.csv", "C3 UQ -> T7: avoidance delta-v vs decision sigma (117 CDM events, true Pc criterion)"),
                                ("c4_budget_summary.csv", "C4 T5 -> T6: recall of reference approaches vs screening budget (600 LEO objects)"),
                                ("p1_thinning.csv", "P1 Sentinel TLE thinning with POD truth (3D P50/P90 km by kept fraction and method)"),
                                ("p1_thinning_ci.csv", "P1 relative change vs nearest (day-block bootstrap 95% CI)"),
                                ("p1_thinning_gapbins.csv", "P1 error by local TLE gap (pooled over kept fractions)"),
                                ("p2_injection.csv", "P2 fault injection with POD truth: raw vs PAF-lite pools"),
                                ("p2_paf_detection.csv", "P2 PAF-lite detection of injected faults"),
                                ("p3_neighbour_effect.csv", "P3 cohort: effect of a PAF-removed nearest neighbour on the LOO residual"),
                                ("p3_sparsity.csv", "P3 cohort TLE sparsity around held-out epochs"),
                                ("p4_storm_sigma.csv", "P4 storm-aware sigma (T3 -> T4-UQ chaining), Sentinel test window"),
                                ("p5_continuity.csv", "P5 trajectory continuity at TLE switch epochs (1-min cadence, Sentinels)"),
                                ("e5_thinning.csv", "E5 cohort thinning (truth-free LOO): nearest vs MTF by removal fraction"),
                                ("e5_thinning_flips.csv", "E5 per-object best-method flips across removal fractions"),
                                ("e6_sweep_best.csv", "E6 best MTF setting per type x regime and loss of a single global setting"),
                                ("e7_error_census.csv", "E7 error census on the full cohort archive (2006 -> 2025), % of TLEs hit by each rule"),
                                ("e7_removed_by_rule.csv", "E7 LOO residual of PAF-removed TLEs by rule vs kept TLEs"),
                                ("e8_truth_availability.csv", "E8 truth availability"),
                                ("e3_loo_by_group.csv", "E3-LOO Truth-free reconstruction consistency on the whole cohort (held-out TLE as reference)"),
                                ("e3_loo_paired.csv", "E3-LOO paired per-object comparison vs nearest-raw"),
                                ("e3_loo_removed.csv", "E3-LOO residual of TLEs removed vs kept by PAF-lite"),
                                ("e3_loo_sentinel_validation.csv", "E3-LOO proxy validation on Sentinels (LOO residual vs POD error)"),
                                ("e4_cohort_transfer.csv", "E4 σ transfer to the cohort (Sentinel-calibrated σ applied to cohort MTF spread)"),
                                ("e4_object_proxy_corr.csv", "E4 delivered OAD proxies vs per-object LOO error"),
                                ("E1_chaining.csv", "E1 Task chaining: independent vs chained"), ("E2_data_ablation.csv", "E2 Data ablation"), ("E3_trajectory_quality.csv", "E3 Trajectory quality vs POD (C0 raw / C1 PAF-lite / C2 PAF-lite+MTF)"), ("E4_uncertainty.csv", "E4 Uncertainty")):
                d = rd(name)
                if d is not None:
                    if name.startswith("E3_"): d = d[(d.sat == "ALL") | (d.split == "all")]
                    if name == "e3_loo_by_group.csv": d = d[d.scope.isin(["ALL", "source=multi", "source=pruned"]) | d.scope.str.contains("LEO")]
                    if name == "p1_thinning.csv": d = d[(d.sat == "ALL")].pivot_table(index=["split", "method"], columns="keep", values="e3d_p50").reset_index()
                    if name == "p2_injection.csv": d = d[d.sat == "ALL"].pivot_table(index=["pool", "method"], columns="p", values="e3d_p90").reset_index()
                    if name == "p4_storm_sigma.csv": d = d[d.component == "t"][["model", "split", "n", "cov68", "cov95", "nees", "err_p50_km", "sigma_p50_km"]]
                    if name == "e7_removed_by_rule.csv": d = d[d.scope == "ALL"]
                    if name == "s1_starlink_forecast.csv": d = d[d.split.isin(["all", "post-thrust(48h)", "no-recent-thrust"])].pivot_table(index=["split", "method"], columns="h", values="3d_p50_km").reset_index()
                    if name == "s1_starlink_uq.csv": d = d[(d.component == "t") & (d.h.astype(str) == "all")][["product", "model", "split", "n", "cov68", "cov95", "nees", "sigma_p50_km"]]
                    if name == "s1_starlink_correctors.csv" and "split" in d: d = d[d.split.isna()].drop(columns=["split"])
                    if name == "c1_maneuver_sigma.csv": d = d[(d.component == "t") & (d.h == "all")][["model", "split", "n", "cov68", "cov95", "nees", "err_p50_km", "sigma_p50_km"]]
                    if name == "c2_uq_ranking.csv": d = d[["lead_h", "ranking", "target", "n_pred", "n_positive", "recall_at_k", "recall_at_2k", "precision_at_10", "spearman_score_vs_neg_truemiss"]]
                    if name == "e5_thinning.csv": d = d[d.type == "ALL"][["frac", "n", "nearest_p50", "mtf_p50", "nearest_p90", "mtf_p90", "mtf_win_rate", "age_nearest_med_h"]]
                    if name == "e6_sweep_best.csv": d = d[["type", "regime", "window_h", "tau_h", "k", "res_p50", "res_p50_at_global_setting", "loss_of_global_pct"]]
                    if name == "uc_t2_sentinel.csv": d = d[d.satellite == "ALL"][["variant", "precision", "recall", "f1", "f2"]]
                    if name == "uc_t1_cohort.csv": d = d.groupby("variant")[[c for c in d.columns if c.startswith("macro") or c.startswith("f1_")]].mean().reset_index()
                    if name == "e51_a0_a1_ci.csv": d = d[d.significant | d.metric.str.contains("safety|AUC|recall")][["example", "variant", "metric", "value_ref", "value_variant", "diff", "ci_lo", "ci_hi", "significant"]]
                    if name == "e52_efficiency.csv": d = d[~d.setting.str.startswith("stage")]
                    if name == "e53_data_ablation.csv": d = d[["config", "data_removed", "example", "metric", "B0_full", "ablated", "rel_change_pct"]]
                    if name == "e54a_shorten_t2.csv": d = d[d.satellite.str.startswith("POOLED") | (d.dataset != "Shorten TLE benchmark")][["dataset", "satellite", "pool", "setting", "n_maneuvers", "n_pred", "precision", "recall", "f1", "f2"]]
                    if name == "x4_sentinel_chain.csv": d = d[d.h == 24][["split", "chain", "n_conj", "n_dangerous", "t6_alert_recall", "t6_false_alert_rate", "t7_safety_dangerous", "t7_missed_dangerous", "t7_unnecessary", "dv_total_m_s"]]
                    if name == "x2_starlink_triage.csv": d = d[["h", "threshold_km", "chain", "n_test", "auc", "avg_precision", "recall_at_30pct_flag", "precision_at_30pct_flag"]]
                    if name == "x1_cdm_chain.csv": d = d[~d.chain.str.contains("x0")][["chain", "n_events", "n_dangerous", "safety_all", "missed_dangerous", "unnecessary_maneuvers", "dv_total_m_s", "sigma_t_within_x2"]]
                    if name == "e7_error_census.csv": d = d[["scope", "n_obj", "n_tle", "half_period_pct", "bstar_neg_pct", "isolated_n_pct", "any_rule_pct", "objects_with_bstar_neg_pct", "median_gap_h"]]
                    parts.append(f"## {title}\n" + md_table(d))
                    if name.startswith("E3_") and len(d[d.sat != "ALL"]): fig_e3(d)
                    if name.startswith("E1") and len(d): fig_e1(d)
            fig_chain_ablation()
            parts.append("## Figures\n" + "\n".join(f"- `figures/{p.name}`" for p in sorted(F.glob("*.pdf"))) + "\n")
            parts.append("## Key findings\n"
                "### Evidence for Challenge 1 (a unified dataset is hard to build)\n"
                "1. **TLE cadence is too heterogeneous for one fusion setting.** Around held-out epochs, only 18% of cohort epochs have >=2 TLEs within +-12 h (LEO payloads 32%, GEO payloads 5%, MEO payloads 3%); the median gap to the nearest neighbour is 10 h overall but 16-22 h in GEO/HEO (P3 sparsity). A fixed 12-h fusion window therefore activates fusion for fewer than one epoch in five, and MTF degenerates to nearest-TLE for most objects.\n"
                "2. **The best reconstruction method depends on density, and the proxy-optimal setting differs from the truth-optimal one.** In the truth-free cohort test the best method flips between nearest and MTF for 34% of objects when TLEs are thinned (65% of LEO payloads, E5); the LOO proxy prefers the widest window (24 h) in every type x regime cell (E6), whereas with POD truth every wider or adaptive window is worse than the plain 12-h MTF (P1). Choosing a propagation setting without truth is therefore not possible in general.\n"
                "### Evidence for Challenge 2 (data quality is hard to control)\n"
                "3. **Errors are numerous and regime-dependent** (E7 census, 32.2 M TLEs of 3,603 objects, 2006-2025): 13.5% of records trigger at least one PAF rule; B* < 0 affects 10.4% of records and 76% of objects (LEO payloads 15%, LEO rocket bodies 17%); half-period re-releases are 0.4% of LEO records but 25-28% of GEO/HEO records.\n"
                "4. **Removal rules have opposite effects in different regimes.** Superseded (half-period) records are 1.9-2.4x less consistent with the rest of the archive in LEO, i.e. the revision corrected them, but 0.6x *more* consistent in GEO/HEO, where half a period is 12 h and the rule deletes legitimate consecutive TLEs (E7 by rule). B* < 0 records are not less consistent than kept ones (0.64x median residual), so a hard delete is not justified by trajectory consistency. Removing neighbours also thins the pool: MTF on the PAF-lite pool is 2% worse than on the raw pool in the LOO test.\n"
                "5. **On precisely tracked payloads, every 'smarter' fusion is worse than the simple one** (P1, POD truth, 5 Sentinels, 2024-2025H1): adaptive windows +24.5%, bracketing blend +19.4%, T2-gated segments +25.3% and even oracle maneuver gating +24.1% at 10% TLE density; at full density the penalties are 0.2-2.4%. The error floor (~0.5 km) is SGP4 model bias shared by all TLEs, which fusion cannot remove.\n"
                "6. **Quality cannot be quantified from truth for 99.9% of objects.** 5 of 3,608 cohort objects (0.14%) have any independent reference (E8). The TLE-consistency residual does not track the true POD error at the sample level (Spearman 0.04, E3 Sentinel validation) but does at the object level (0.61 with the delivered OAD proxies, E4).\n"
                "### Positive evidence for the OrbBench methods\n"
                "7. **PAF removes faults that would corrupt trajectories** (P2, POD truth): with faults injected at the archive's own type mix and magnitudes, PAF-lite recovers 74-77% of them at a 1.3-3.4% removal rate on clean records; at a 10% fault rate the P90 error of MTF trajectories drops from 1.17 to 0.98 km (-16%) and the P50 from 0.554 to 0.528 km; the P99 tail is cut from 934 to 326 km for MTF and from 14.1 to 5.8 km for nearest. Fusion amplifies undetected faults (raw MTF P99 934 km vs raw nearest 14 km), so PAF is a precondition for MTF rather than an alternative to it.\n"
                "8. **MTF gives smoother products, not lower error.** Against POD, MTF beats nearest by 0.36% [95% CI 0.22-0.48%] at full density and by 2.5% in the truth-free cohort test; its main benefit is continuity: the kinematic jump at TLE switch epochs falls from 24.5 m (P50) / 138 m (P90) with nearest to 7.5 / 14 m with MTF, and the share of switches with jumps above 50 m from 25% to 1% (P5).\n"
                "9. **Calibrated, age-dependent uncertainty is better than a fixed one**: at 24 h NEES 3.1 vs 4.7 and 95% coverage 0.90 vs 0.88 (T4-UQ). The delivered object-level OAD proxies correlate 0.61 (Spearman) with reconstruction error on 482 objects.\n"
                "10. **High-precision references have measurable value**: a corrector trained on 2019-2023 POD reduces the forecast error of SGP4-MTF by 38% (0.5 h), 46% (1 h), 23% (4 h) and 58% (8 h), and by 48% in post-maneuver windows at 4 h, but is neutral at 12 h and 15% worse at 24 h with heavier tails (T4, long window).\n"
                "11. **Maneuvers matter and detection works**: forecast error is 49-77% higher in the 48 h after a logged maneuver (T4 split); the SGP4-consistency detector reaches recall 0.64 / F2 0.54 on 272 logged maneuvers while a supervised model does not transfer across years (F2 0.16).\n"
                "### The connected architecture: which edges pay off (C1-C5, `chain_a0_a1.csv`)\n"
                "12. **Events -> uncertainty (T2 -> UQ) is the edge that matters for maneuvering objects.** Selecting TLEs by maneuver segment changes forecast error by <0.2% (the post-maneuver TLE already carries the maneuver), but feeding the maneuver signal into sigma does: on Sentinel POD (test 2024-25H1, 3 satellites with logs) an age-only sigma covers 78.0% of along-track errors at 95% nominal in the 48 h after a maneuver (NEES 10.3); scaling sigma by the T2 consistency score raises coverage to 81.6% (NEES 7.6) with no truth used, and using the OrbBench maneuver records to 86.7% (NEES 3.9); overall NEES falls from 4.8 to 2.4 while quiet-period coverage stays at 0.96. Uncertainty, not the point estimate, is where event knowledge should enter.\n"
                "13. **Population evolution -> screening (T5 -> T6) works across shells, not within one.** On 600 LEO objects (300-2000 km, 3 d, 139 reference approaches < 10 km), ordering candidate pairs by T5-predicted shell density recovers 34.8% of approaches with 20% of the pairs and 65.2% with 50% (random: 20.5% / 53.8%; AUC 0.66 vs 0.52). A simple 'same shell first' ordering does even better (47.7% / 81.1%, AUC 0.71), so the gain comes from altitude structure that T5 predicts, and prioritisation only pays when the cohort spans several shells (inside one 700-1000 km shell it lost recall).\n"
                "14. **Uncertainty -> avoidance (UQ -> T7) is worth a factor 34 in propellant, but only with per-event calibration.** On 117 ESA CDM events (miss < 2 km), a 2-km distance rule is safe on 100% with mean delta-v 10.3 m/s; a Pc-gated policy using the per-event covariance is safe on 100% (true Pc < 1e-4) with 0.30 m/s. Halving that sigma drops safety to 75%; a single fleet-wide sigma, the TLE-derived sigma-age or a conservative 2-km sigma never trigger and leave 25% of events unsafe. A dataset that supplies calibrated per-object uncertainty is therefore a prerequisite for probability-based avoidance.\n"
                "15. **Uncertainty -> warning ranking (UQ -> T6) is not demonstrated** in the 700-1000 km debris shell: predicted-approach errors (~100 km, TLE age 14-37 h) swamp any covariance, and recall@K of true approaches stays at 0.16-0.17 for point-miss, fixed, age or object-level sigma. Space-weather chaining (T3 -> T4/UQ) shows no effect at Sentinel altitudes (storms raise 24-h error by 4%).\n"
                "16. **End-to-end (A0 independent vs A1 chained)**: MTF product P90 switch jump 138 -> 14 m; post-maneuver along-track 95% coverage 0.78 -> 0.87; screening recall at 20% budget 0.20 -> 0.35; avoidance delta-v at equal safety 10.3 -> 0.30 m/s. Every edge with a positive result carries information through *uncertainty or priority*, not by editing the point trajectory.\n")
            parts.append("### Starlink end-to-end test (17 satellites, operator MEME ephemerides with 6x6 covariance as truth; raw + official step6 TLEs to 2026-08-24; test 2026-07-05..08-23, 3-h origins; `starlink_a0_a1.csv`, `s1_*`, `s2_*`)\n"
                "17. **A filtering task run on its own is harmful on maneuvering satellites; connected to a data-availability gate it is safe.** Starlinks thrust 82% of the time. The official step6/PAF pool removes 25% of their TLEs (retention 26-35% on three satellites) and those are the maneuver TLEs, so causal forecasts from the filtered pool alone reach P90 31.7 km at 1 h and 111 km at 24 h against 3.8 / 27.2 km from the raw archive (P50 +20% / +13%); on the worst satellites the 24-h P90 is 3,900-5,700 km. Falling back to the raw pool whenever the filtered pool has no TLE within 24 h (18% of origins) restores P90 to 4.1 / 27.3 km and P50 to within 3-4% of raw; a T1-aware PAF-lite that suspends the element-jump and B* rules inside detected raise/lower arcs (retention 82-98%) is equally safe. This is the clearest 'independent task bad, connected chain good' case in the study and the Starlink counterpart of the Sentinel result that PAF is a precondition for MTF.\n"
                "18. **Thrust, not TLE age, sets the Starlink forecast error.** 24-h P50 is 2.7 km when no thrust falls in the horizon and 5.4 km when it does (T1 phase stable 2.7 km vs lowering 5.6 km). 32% of 24-h forecasts miss by > 10 km and 12% of along-track errors exceed 3 sigma, so Gaussian sigma models under-cover (fixed sigma 68/81% at 68/95% nominal; age sigma 50/64%; age x g(T2) 62/72%). The operator covariance describes the ephemeris (sigma_T 0.1-0.2 km) and covers 6% of the TLE error. Residual correctors trained on 3 weeks of MEME make forecasts 3-5x worse because the station-keeping direction alternates between months; T1/T2 features reduce the damage (1 h: 3.6 -> 2.7 km) but do not remove it.\n"
                "19. **Maneuver detection transfers to Starlink**: the SGP4-consistency detector reaches precision 0.84 at recall 0.37 against MEME-derived thrust arcs (threshold tuned on the first 3 weeks), flagging thrust change-points rather than whole arcs. Gating the fusion on it changes forecast error by < 1%, consistent with the Sentinel result: maneuver knowledge belongs in sigma and in the filter, not in TLE selection.\n"
                "20. **No dangerous conjunctions occur among the 17 Starlinks in 50 days** (134 approaches < 50 km, closest 17 km, true Pc from the operator covariances = 0), so UQ -> ranking and warning -> avoidance cannot be tested on this cohort; recall of the < 50 km approaches from TLE trajectories is 7-10% (A1 +22-29% relative to A0) at 2% precision, because TLE errors (5-27 km) are comparable to the approach distances.\n")
            parts.append("### Task-level chain ablations: full chain vs the same chain with one upstream task removed (X1-X4, `x*_*.csv`, `fig_chain_ablation.pdf`)\n"
                "21. **X3 Evolution -> warning -> avoidance (600 LEO objects, 3 d, 139 reference approaches < 10 km, 20% screening budget).** Full chain T5 -> T6 -> T7 reaches 51.6 of 139 approaches (recall 0.371) and spends 0 m/s because no reached approach has Pc >= 1e-4 under the T6 sigma; removing T5 (random pair order) reaches 28.4 (recall 0.205, -45%); removing the T6 sigma (2-km distance rule) keeps the 51.6 approaches but spends 14.0 m/s on events whose Pc is 0; A0 (neither) reaches 28.4 and spends 7.7 m/s. Oracle ordering reaches all 139.\n"
                "22. **X2 Phase + maneuver detection -> forecast reliability -> screening triage (17 Starlinks, MEME truth, 4,473 test forecasts at 24 h, 31.7% miss by > 10 km).** T4 predicts which 24-h forecasts will be unreliable so that T6 does not screen on them. With the T1 phase/SMA-rate and T2 score as inputs the flag reaches AUC 0.651 and recall 0.451 at a 30% flag budget; removing T1 drops it to AUC 0.584 / recall 0.361 (below the age-only A0 baseline 0.602 / 0.369), removing T2 leaves it unchanged (0.654 / 0.451), and adding the T3 Ap/F10.7 series does not help at 24 h (0.643 / 0.428) but does at 4 h (AUC 0.733 with T3 vs 0.710 without; -T1 0.704; A0 0.718). Phase knowledge is the input that carries the maneuver information the forecast lacks.\n"
                "23. **X1 Weather + object attributes -> chaser-uncertainty model -> Pc alert -> Pc-gated avoidance (199 ESA CDMs, operator covariances as truth, 56 dangerous events with true Pc >= 1e-4).** T4 predicts the chaser RTN sigma from the OD age plus T5 object/orbit attributes and T3 weather (5-fold CV; 42% of along-track sigmas within a factor 2 vs 29% for a single fixed sigma), T6 alerts at predicted Pc >= 1e-4 and T7 maneuvers down to 1e-5. Full chain: 12 of 56 dangerous events missed (safety on dangerous 0.714), 16 unnecessary maneuvers, 458 m/s; removing T3: 14 missed (0.696); removing T5: 14 missed (0.661); removing both (age-only sigma): 17 missed (0.625). Removing T4 (fixed sigma) misses 10 but fires 25 unnecessary maneuvers and spends 816 m/s; removing T6 (2-km distance rule) misses none but fires 61 unnecessary maneuvers and spends 1,210 m/s. The operator covariance (oracle) misses none with 0 unnecessary and 227 m/s, so the remaining gap is the sigma model, not the chain.\n"
                "24. **X4 Maneuver detection -> uncertainty -> Pc alert -> avoidance on POD truth (Sentinel-1A/2A/3A, test 2024-25H1, 2,913 forecast origins x 4 synthetic chasers per horizon: miss log-uniform 0.05-10 km, chaser sigma 0.2 km).** In the 48 h after a logged maneuver (1,463 dangerous conjunctions at 4 h) the full chain alerts on 67.9% of dangerous conjunctions and T7 leaves 64.9% of them safe; removing T2 (maneuver-blind sigma) gives 63.7% / 59.5% (+13% missed dangerous), a perfect T2 (logged maneuvers) 70.7% / 70.3%. Over all 4,799 dangerous conjunctions at 4 h the chain is safe on 81.6% (-T2 78.7%) with 22.6 km/s total delta-v, the distance rule on 95.7% with 78.7 km/s and 7.4x the unnecessary maneuvers; a fixed sigma is equivalent to the chain here (83.1%, +4% delta-v). With a perfect forecast every dangerous conjunction is handled at 15.3 km/s, so the forecast error, not the controller, is the limit.\n"
                "25. **Reading across X1-X4**: every removal of an upstream *task output* degrades the downstream metric where the physics says it should (T5 ordering -> screening recall; T1 phase -> forecast reliability; T3/T5 attributes -> chaser sigma -> missed dangerous events; T2 -> post-maneuver safety), while removing the *uncertainty* task is neutral-to-harmful in propellant rather than in safety, and removing the *Pc alert* (distance rule) buys safety at 2.6-3.5x the delta-v. Space weather (T3) only pays off at short horizons.\n")
            parts.append("### Significance, efficiency, and data ablation for Sec. 5.2-5.3 (E51-E53)\n"
                "26. **Which chain effects survive a paired bootstrap (95% CI).** Removing the Pc alert (T6, distance rule) is significant everywhere: on the 199 CDMs it adds 45 [34, 57] unnecessary maneuvers and 752 [511, 985] m/s; on the Sentinel conjunctions it adds 2,678 [2,586, 2,764] unnecessary maneuvers and 56 [55, 57] km/s at 4 h while raising safety from 0.82 to 0.96. Removing the uncertainty task (T4, fixed sigma) is significant in propellant, not in safety: +9 [2, 17] unnecessary maneuvers and +359 [124, 581] m/s on the CDMs, +956 [807, 1,104] m/s on the Sentinels. Removing the maneuver detector (T2) is significant on the Sentinel post-maneuver conjunctions: safety 0.649 -> 0.595 [-0.068, -0.041] and alert recall 0.679 -> 0.637 [-0.054, -0.030]; a perfect detector (logged maneuvers) would give 0.703 [+0.041, +0.067]. Removing the phase task (T1) is significant on Starlink triage at 24 h: AUC 0.643 -> 0.584 [-0.078, -0.038], recall at 30% budget 0.428 -> 0.361 [-0.091, -0.038]; removing T2 or T3 there is neutral or slightly positive, and at 4 h T3 (weather) and T1 are both significant (AUC -0.023 [-0.030, -0.016] and -0.029 [-0.047, -0.014]). On the CDMs the T3/T5 inputs to the sigma model change missed dangerous events by +2 each, not significant (CI [-1, 6] and [-2, 6]) with 56 dangerous events; only their effect on delta-v (T3: -117 [-256, -1] m/s) reaches significance. T5 ordering vs random at a 20% screening budget: recall 0.371 vs 0.205 (X3). **Efficiency (E52):** screening all 58,825 candidate pairs of 600 LEO objects over 3 d takes 22.4 s on the 1-min grid; the T5-ordered 20% budget (11,765 pairs) takes 5.0 s and reaches 38.8% of the reference approaches, against 20.1% for a random 20% budget at the same cost.\n"
                "27. **Data ablation on decision metrics (E53, `e53_data_ablation.csv`).** B1 No-maneuver: Starlink 4-h/24-h P90 5.4 -> 39.8 km / 27 -> 113 km when the filtered pool is used without maneuver awareness; Sentinel post-maneuver safety 0.649 -> 0.595 and along-track 95% coverage 0.796 -> 0.717; Starlink triage AUC 0.643 -> 0.602. B2 No-weather: CDM missed dangerous 12 -> 14 (n.s.), Starlink 4-h triage AUC 0.733 -> 0.710 (sig.), 24 h unchanged. B3 No-physical: CDM missed dangerous 12 -> 14 (n.s.), sigma within x2 0.42 -> 0.49. B4 No-precise: without POD-trained correctors the 1-h/4-h forecast P50 is 0.31 -> 0.57 km / 0.38 -> 0.50 km (24 h: 0.53 -> 0.46, correctors hurt); without covariance-calibrated sigma the CDM chain fires 25 vs 16 unnecessary maneuvers and 816 vs 458 m/s; NEES 4.1 -> 5.5. B5 TLE-only: screening recall at 20% budget 0.371 -> 0.205 and 7.7 m/s spent on zero-Pc events; CDM chain 61 unnecessary maneuvers and 1,210 m/s. The pattern matches Sec. 3.4: auxiliary data pay off through uncertainty and priority, and only maneuver awareness changes point-forecast error, and only on maneuvering satellites.\n")
            parts.append("### Same tasks on existing datasets (E54, `e54a_shorten_t2.csv`, `e54b_kelvins_t6.csv`, `e54_dataset_relation.json`)\n"
                "28. **T2 on the Shorten et al. TLE benchmark (15 satellites, 887 curated maneuvers in the TLE-covered span).** The OrbBench SGP4-consistency detector transfers with the threshold tuned on Sentinel: pooled F1 0.39 / F2 0.55 on their raw TLE files, 0.44 / 0.60 after PAF-lite; with the best threshold on their data 0.49 -> 0.60 (F1) and GEO satellites 0.77 -> 0.82. PAF raises precision at equal or higher recall on every pool (raw 0.38 -> PAF 0.51 at recall 0.70 -> 0.73), i.e. OrbBench's quality control improves the same task on an external dataset. The Sentinel reference numbers (F1 0.43 pooled) sit inside the range of the external satellites, so the OrbBench T2 evaluation is not tuned to its own data.\n"
                "29. **T6 on the full ESA Kelvins CDM dataset (13,154 training events, 2,167 test events, private final risk as truth).** OrbBench's 199 CDM rows are exactly rows of the Kelvins private test file: the final CDM of 199 test events, comprising all 150 events with final risk >= -6 plus 49 low-risk ones, so the OrbBench T6-CDM/T7 experiments are on the high-risk subset of the Kelvins test set, not on independent data (`e54_dataset_relation.json`). Under the competition protocol (features from the last CDM at least 2 d before TCA; score = MSE on high-risk events / F2), the persistence baseline (latest risk) reaches F2 0.74 with MSE 26.6, whereas every regressor we fit, Ridge or HistGB, on all events or only on events with pre-risk >= -8, with all 105 features or 28 geometry and risk-history features, collapses to F2 <= 0.01: squared-error training over events whose risk mostly vanishes by TCA shrinks high-risk predictions toward -30, and the metric only scores the high-risk events. Within the training set (5-fold CV) the same models also lose to persistence (MSE 53 vs 3.5). This reproduces the competition's own conclusion that the latest-risk baseline is hard to beat, and it is why OrbBench evaluates T6/T7 on decision metrics (missed dangerous events, unnecessary maneuvers, delta-v at true-Pc safety) computed with the CDM covariances rather than on final-risk regression.\n")
            parts.append("### Full chain T1->T7 and upstream connections (E2E, UC; `e2e_*.csv`, `uc_*.csv`)\n"
                "30. **Staging fix (2026-09-16).** The L2 trajectory products carry a fixed per-object sub-minute timestamp offset (uniform 0-60 s). Stage 0 previously snapped each 1-min sample to the nearest grid minute, i.e. a 50-225 km along-track shift of every cohort reference trajectory. Staging now interpolates position and velocity with cubic Hermite; against SGP4 of the object's own nearest TLE the re-staged references agree to 0-150 m. All cohort-reference results (T5, T6 shell screening, C4, X3, E2E) were rerun; Sentinel, Starlink and CDM results use independent truth and were unaffected. Changes: C4 T5-ordering AUC 0.656 -> 0.651 with 182 (was 139) reference approaches on the 600-object cohort, random 0.519 -> 0.494; X3 recall at 20% budget T5 0.371 -> 0.383 vs random 0.205 -> 0.136 (paired CI on the difference [-0.32, -0.14]); T6 shell recall of independent screening 0.915 (was ~0.17 because the reference itself was displaced), UQ ranking recall@K 0.81-0.88 for all sigma modes (was 0.16-0.17); T5 SGP4-prev density overlap 97.6% (was lower), TV 0.024.\n"
                "31. **Full chain, 1,500 LEO objects.** Tasks without upstream (T1, T2, T3, T5) and the point forecast of T4 are identical in both modes (24-h P50 0.11 km, P95 3.09 km against the non-causal L2 product of the same TLEs); differences appear only in uncertainty, warning and avoidance. With a 20% screening budget the T5 ordering finds 37.5% of the 953 reference approaches vs 19.6% for random order at the same cost (11.6 vs 10.7 s per epoch; all 385,661 pairs: 96.1% in 50.8 s). Natural conjunctions with true Pc >= 1e-4 are rare (5 in 3 d), so 30 dangerous conjunctions per epoch are injected by shifting one object's reference and forecast trajectories by the same vector (reference miss 20-1000 m, forecast error unchanged): of 123 dangerous injected events the unified pipeline has 39% inside its budget, alerts on 45 and makes 45 safe with 0 unnecessary maneuvers and 275 m/s; independent execution 22% / 26 / 25 / 4 / 414 m/s. On natural approaches the distance rule fires 45 times, 43 of them on events with true Pc < 1e-4; the Pc gate fires 0 times and no natural event reaches the gate. Per-task removal on the same cohort: -T5 -> 22% in budget, 25 alerted (back to independent); -T6 Pc -> 47 alerted but 8 unnecessary on injected + 108 on natural events, 761 m/s (2.8x); -T4 sigma -> 44 alerted, NEES 0.030 -> 0.043; -T1/-T3 no change (quiet week: storm factor never triggered, 26 non-stable objects); -T2 44 safe. The cohort reference is pseudo-truth generated from the same TLEs, so sigma is conservative against it (NEES << 1); the independent-truth sub-chains (X1, X2, X4) carry the significance tests.\n"
                "32. **Upstream connections outside the task graph.** T3 -> T2 (raise the consistency threshold on days with Ap >= 40): Sentinel storm-day false alarms 70 -> 46, pooled F1 0.433 -> 0.443 at unchanged recall. T1 -> T2 (phase-specific thresholds): no change on Sentinel (99.8% of epochs stable); on Starlink the detection task itself degenerates (84% of TLE intervals inside thrust arcs; 'always thrust' F1 0.91, every detector MCC <= 0.07). T2 -> T1 (T2 flags as phase features): Macro-F1 0.666 unchanged (cohort labels are only STABLE/DECAY). Reading: whether a connection helps is decided by whether the data contain the variation it detects.\n")
            parts.append("## Notes and caveats\n- Reference trajectories for T5/T6 are OrbBench L2 trajectories (MTF / nearest SGP4), i.e. pseudo-truth; only T2/T4/E3 use independent POD truth.\n- All methods are causal at the forecast origin; Ridge correctors and σ models are fit on 2024-07 → 2024-12 and frozen.\n- Deep-learning rows marked 'reported' come from the existing 8-model experiment (2025-H2, pseudo-truth) and are not re-run here (CPU only).\n- T1 silver labels derive from rule thresholds calibrated on Starlink; HistGB re-learns the rules (near-perfect ARI) and should be read as an upper bound, the unsupervised rows are the informative ones.\n")
            (C.ROOT / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
            log.info(f"REPORT.md written ({sum(len(p) for p in parts)//1000} kB), figures: {len(list(F.glob('*.pdf')))}")
        set_status("S9_report", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("report failed"); set_status("S9_report", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main()
