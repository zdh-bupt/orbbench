#!/usr/bin/env python3
"""Assemble Table 4 (task-level comparisons, independent vs unified pipeline) from the result files shipped in this
release, write it to tables/table4_assembled.md (+ one CSV per block), and optionally check every cell against the
numbers printed in the paper (--check).

Sources (all relative to the release root):
  T1  t1_phase_classification/reports/observable_regime_feature_labels_v2/retest/summary_by_system.csv   (split == test)
  T2  pipeline/results/detail_table.csv            block "T2 Maneuver"  (F1, False alarms)
  T3  t3_space_weather/metrics/backtest_metrics_summary.json            (ctx{30,60,90}_pred{3,7} -> MAE_mean)
  T4  pipeline/results/detail_table.csv            block "T4 Forecast"  (95 % coverage, 24 h)  -> PICP95
      pipeline/results/deep_forecast_table_1h.csv / deep_forecast_table_24h.csv                -> position error (km)
  T5  pipeline/results/t5_evolution_reported.csv   (TV, regional relative MAE, +1 h)
  T6  pipeline/results/detail_t6_found_per_origin.csv                    -> ACR (%)  [stored rounded to 0.1]
      pipeline/results/detail_table.csv            block "T6 Warning"   -> DCR (%) = "In budget (%)", DAC = "Alerted"
  T7  pipeline/results/detail_table.csv            block "T7 Avoidance" -> DRC = "Resolved", dv_total = "dv"
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "pipeline" / "results"
OUT = ROOT / "tables"

STARLINK = {"Starlink 44752": "Starlink-1047", "Starlink 44723": "Starlink-1017", "Starlink 44751": "Starlink-1046"}
T4_COLS = ["Sentinel-1A", "Sentinel-2A", "Sentinel-2B", "Sentinel-3A", "Sentinel-3B",
           "Starlink-1047", "Starlink-1017", "Starlink-1046", "All Sentinel", "All Starlink", "All cohort"]
T6_COLS = ["Payload", "Rocket body", "Debris", "300-800", "800-1200", "1200-2000", "Overall"]
T6_KEY = {"Payload": "payload", "Rocket body": "rocket body", "Debris": "debris",
          "300-800": "300-800 km", "800-1200": "800-1200 km", "1200-2000": "1200-2000 km", "Overall": "All"}

# ----------------------------------------------------------------------------------------------- paper values
# Transcribed from Table 4 of the manuscript; used only by --check.
PAPER = {
    "T1": {"Constant stable": (0.3148, 0.0000), "PCA + KMeans": (0.7010, 0.7938),
           "PELT + KMeans": (0.4708, 0.6598), "Transformer + KMeans": (0.3244, -0.0168)},
    "T2": {("F1", "Independent"): [0.411, 0.467, 0.479, 0.433], ("F1", "Unified"): [0.420, 0.492, 0.479, 0.443],
           ("FA", "Independent"): [223, 67, 72, 362], ("FA", "Unified"): [207, 60, 72, 339]},
    "T3": {(30, 3): (11.763, 1.033, 15.302), (60, 3): (12.001, 0.911, 14.028), (90, 3): (11.225, 0.878, 13.550),
           (30, 7): (9.839, 1.337, 14.336), (60, 7): (10.244, 0.980, 12.003), (90, 7): (9.548, 0.861, 10.956)},
    "T4_PICP": {"Independent": [0.748, 0.825, 0.880, 0.881, 0.846, 0.933, 0.934, 0.721, 0.835, 0.857, 0.924],
                "Unified":     [0.797, 0.846, 0.887, 0.897, 0.861, 0.982, 0.958, 0.775, 0.858, 0.909, 0.932]},
    "T5": {"Last-frame persistence": (0.6681, 1.0034), "Kepler + CIC": (0.1721, 0.1997), "Physics + U-Net": (0.1836, 0.2222)},
    "T6": {("ACR", "Independent"): [18.106, 17.808, 21.0, 22.775, 18.022, 20.885, 19.622],
           ("ACR", "Unified"):     [47.786, 36.073, 31.700, 28.743, 39.552, 40.265, 37.461],
           ("DCR", "Independent"): [27.500, 26.667, 17.647, 20.0, 19.178, 30.0, 21.951],
           ("DCR", "Unified"):     [50.0, 26.667, 35.294, 55.0, 32.877, 43.333, 39.024],
           ("DAC", "Independent"): [11, 4, 11, 4, 13, 9, 26], ("DAC", "Unified"): [18, 4, 23, 10, 23, 12, 45]},
    "T7": {("DRC", "Independent"): [11, 4, 10, 4, 12, 9, 25], ("DRC", "Unified"): [18, 4, 23, 10, 23, 12, 45],
           ("dv", "Independent"): [198.300, 69.300, 146.400, 58.800, 189.0, 166.200, 414.0],
           ("dv", "Unified"):     [119.400, 23.400, 132.0, 52.500, 123.300, 99.0, 274.800]},
}

# ----------------------------------------------------------------------------------------------- blocks
def t1() -> pd.DataFrame:
    p = ROOT / "t1_phase_classification/reports/observable_regime_feature_labels_v2/retest/summary_by_system.csv"
    d = pd.read_csv(p); d = d[d.split == "test"]
    names = {"constant_stable": "Constant stable", "raw_pca_kmeans": "PCA + KMeans", "pelt_kmeans": "PELT + KMeans",
             "transformer_embedding_kmeans": "Transformer + KMeans"}
    out = pd.DataFrame({"Method": d.system.map(names), "Macro-F1": d.macro_f1_mean.round(4), "ARI": d.adjusted_rand_index_mean.round(4)})
    return out.reset_index(drop=True)

def _detail() -> pd.DataFrame:
    return pd.read_csv(R / "detail_table.csv")

def t2() -> pd.DataFrame:
    d = _detail(); d = d[d.block == "T2 Maneuver"]; cols = ["Sentinel-1A", "Sentinel-2A", "Sentinel-3A", "All"]
    rows = []
    for metric, label in (("F1", "F1"), ("False alarms", "FA")):
        for mode in ("independent", "unified"):
            v = [float(d[(d.column == c) & (d.metric == metric)][mode].iloc[0]) for c in cols]
            rows.append([label, mode.capitalize()] + v)
    return pd.DataFrame(rows, columns=["Metric", "Mode", "Sentinel-1A", "Sentinel-2A", "Sentinel-3A", "Overall"])

def t3() -> pd.DataFrame:
    j = json.loads((ROOT / "t3_space_weather/metrics/backtest_metrics_summary.json").read_text(encoding="utf-8"))
    rows = []
    for ctx in (30, 60, 90):
        r = [ctx]
        for h in (3, 7):
            b = j[f"ctx{ctx}_pred{h}"]; r += [b["f107_obs"]["MAE_mean"], b["kp_daily_avg"]["MAE_mean"], b["ap_avg"]["MAE_mean"]]
        rows.append(r)
    return pd.DataFrame(rows, columns=["History (days)", "3d F10.7", "3d Kp", "3d Ap", "7d F10.7", "7d Kp", "7d Ap"])

def t4_picp() -> pd.DataFrame:
    d = _detail(); d = d[d.block == "T4 Forecast"].copy()
    d["column"] = d.column.replace(STARLINK).replace({"All (Sentinel)": "All Sentinel", "All (Starlink)": "All Starlink", "All (cohort)": "All cohort"})
    rows = []
    for mode in ("independent", "unified"):
        rows.append(["PICP95", mode.capitalize()] + [float(d[d.column == c][mode].iloc[0]) for c in T4_COLS])
    return pd.DataFrame(rows, columns=["Metric", "Mode"] + T4_COLS)

def t4_pos(h: str) -> pd.DataFrame:
    d = pd.read_csv(R / f"deep_forecast_table_{h}.csv")
    d = d[d.model.isin(["LSTM", "GRU", "DLinear", "TCN", "Informer", "FEDformer", "Chronos", "Chronos-2"])]
    return d.rename(columns={"model": f"Position error {h}"}).reset_index(drop=True)

def t5() -> pd.DataFrame:
    d = pd.read_csv(R / "t5_evolution_reported.csv")
    names = {"persistence": "Last-frame persistence", "physics": "Kepler + CIC", "learned": "Physics + U-Net"}
    out = pd.DataFrame({"Method": d.method_key.map(names), "TV": d.tv, "Regional relative MAE": d.regional_relative_mae})
    return out.set_index("Method").loc[list(names.values())].reset_index()

def t6() -> pd.DataFrame:
    d = _detail(); acr = pd.read_csv(R / "detail_t6_found_per_origin.csv").set_index("group")
    rows = []
    for mode in ("independent", "unified"):
        rows.append(["ACR (%)", mode.capitalize()] + [float(acr.loc[T6_KEY[c], mode]) for c in T6_COLS])
    for label, metric in (("DCR (%)", "In budget (%)"), ("DAC", "Alerted")):
        w = d[(d.block == "T6 Warning") & (d.metric == metric)]
        for mode in ("independent", "unified"):
            rows.append([label, mode.capitalize()] + [float(w[w.column == T6_KEY[c]][mode].iloc[0]) for c in T6_COLS])
    return pd.DataFrame(rows, columns=["Metric", "Mode"] + T6_COLS)

def t7() -> pd.DataFrame:
    d = _detail(); rows = []
    for label, metric in (("DRC", "Resolved"), ("dv_total (m/s)", "dv")):
        w = d[(d.block == "T7 Avoidance") & (d.metric == metric)]
        for mode in ("independent", "unified"):
            rows.append([label, mode.capitalize()] + [float(w[w.column == T6_KEY[c]][mode].iloc[0]) for c in T6_COLS])
    return pd.DataFrame(rows, columns=["Metric", "Mode"] + T6_COLS)

# ----------------------------------------------------------------------------------------------- check
def check(blocks: dict) -> int:
    bad = 0
    def cmp(name, got, exp, tol):
        nonlocal bad
        ok = abs(float(got) - float(exp)) <= tol
        bad += (not ok); print(f"{'ok ' if ok else 'BAD'}  {name:<48s} file={float(got):<12.4f} paper={float(exp)}")
    for _, r in blocks["T1"].iterrows():
        e = PAPER["T1"][r.Method]; cmp(f"T1 {r.Method} Macro-F1", r["Macro-F1"], e[0], 5e-5); cmp(f"T1 {r.Method} ARI", r["ARI"], e[1], 5e-5)
    for _, r in blocks["T2"].iterrows():
        for c, e in zip(["Sentinel-1A", "Sentinel-2A", "Sentinel-3A", "Overall"], PAPER["T2"][(r.Metric, r.Mode)]):
            cmp(f"T2 {r.Metric} {r.Mode} {c}", r[c], e, 5e-4 if r.Metric == "F1" else 0.5)
    for _, r in blocks["T3"].iterrows():
        for h in (3, 7):
            e = PAPER["T3"][(int(r["History (days)"]), h)]
            for v, name in zip(e, ("F10.7", "Kp", "Ap")): cmp(f"T3 ctx{int(r['History (days)'])} {h}d {name}", r[f"{h}d {name}"], v, 5e-4)
    for _, r in blocks["T4 PICP"].iterrows():
        for c, e in zip(T4_COLS, PAPER["T4_PICP"][r.Mode]): cmp(f"T4 PICP95 {r.Mode} {c}", r[c], e, 5e-4)
    for h in ("1h", "24h"):
        paper = pd.read_csv(OUT / f"table4_position_error_p50_{h}.csv").set_index("model")
        got = blocks[f"T4 pos {h}"].set_index(f"Position error {h}")
        for m in got.index:
            for c in T4_COLS: cmp(f"T4 pos {h} {m} {c}", got.loc[m, c], paper.loc[m, c], 5e-4)
    for _, r in blocks["T5"].iterrows():
        e = PAPER["T5"][r.Method]; cmp(f"T5 {r.Method} TV", r.TV, e[0], 5e-5); cmp(f"T5 {r.Method} RRMAE", r["Regional relative MAE"], e[1], 5e-5)
    for blk, key in (("T6", "T6"), ("T7", "T7")):
        for _, r in blocks[blk].iterrows():
            short = r.Metric.split(" ")[0].replace("dv_total", "dv")
            for c, e in zip(T6_COLS, PAPER[key][(short, r.Mode)]):
                cmp(f"{blk} {short} {r.Mode} {c}", r[c], e, 0.06 if short == "ACR" else 5e-4)
    print(f"\n{'ALL CELLS MATCH' if bad == 0 else str(bad) + ' MISMATCHES'} (ACR compared at 0.06 because detail_t6_found_per_origin.csv is stored rounded to 0.1)")
    return bad

# ----------------------------------------------------------------------------------------------- main
def md_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown renderer (no tabulate dependency); floats to 3 decimals, integers unchanged."""
    def fmt(v):
        if isinstance(v, float): return f"{int(v)}" if v.is_integer() and abs(v) < 1e6 else f"{v:.3f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"; sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)])

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare every cell with the numbers in the paper")
    a = ap.parse_args()
    blocks = {"T1": t1(), "T2": t2(), "T3": t3(), "T4 PICP": t4_picp(), "T4 pos 1h": t4_pos("1h"), "T4 pos 24h": t4_pos("24h"),
              "T5": t5(), "T6": t6(), "T7": t7()}
    titles = {"T1": "T1 Phase classification", "T2": "T2 Maneuver detection (F1 up, false alarms down)", "T3": "T3 Space weather forecasting (Chronos-2 backtest MAE)",
              "T4 PICP": "T4 Orbit forecasting: PICP95 of the 24 h along-track interval", "T4 pos 1h": "T4 Orbit forecasting: 3D position error P50 at 1 h (km)",
              "T4 pos 24h": "T4 Orbit forecasting: 3D position error P50 at 24 h (km)", "T5": "T5 Population forecasting (+1 h)",
              "T6": "T6 Collision warning (1,500-object LEO cohort, 20 % screening budget)", "T7": "T7 Collision avoidance (same cohort)"}
    OUT.mkdir(exist_ok=True); md = ["# Table 4 assembled from the released result files\n"]
    for k, df in blocks.items():
        fn = "table4_" + k.lower().replace(" ", "_") + "_assembled.csv"; df.to_csv(OUT / fn, index=False)
        md += [f"## {titles[k]}\n", md_table(df), f"\n_source CSV: tables/{fn}_\n"]
    (OUT / "table4_assembled.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md)); print(f"\nwritten: {OUT / 'table4_assembled.md'}")
    return check(blocks) if a.check else 0

if __name__ == "__main__":
    sys.exit(main())
