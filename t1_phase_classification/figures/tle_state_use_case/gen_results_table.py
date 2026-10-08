#!/usr/bin/env python3
"""Generate Markdown and LaTeX tables for the TLE clustering use case."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "reports/observable_regime_feature_labels_v2/retest"
DATASET_DIR = PROJECT_ROOT / "output/tle_observable_regime_feature_labels_v2"

METHODS = ["raw_pca_kmeans", "pelt_kmeans", "transformer_embedding_kmeans"]
DISPLAY_ROWS = ["constant_stable", *METHODS]
METHOD_LABELS = {
    "constant_stable": "Constant stable (control)",
    "raw_pca_kmeans": "Raw PCA + KMeans",
    "pelt_kmeans": "PELT + KMeans",
    "transformer_embedding_kmeans": "Transformer embedding + KMeans",
}
STATE_LABELS = {
    "STABLE_ORBIT": "Stable orbit",
    "SUSTAINED_RAISE": "Sustained raise",
    "SUSTAINED_DECAY": "Sustained decay",
    "OFFSET_PLATEAU": "Offset plateau",
    "SHORT_EXCURSION_RETURN": "Short excursion and return",
    "UNKNOWN_TRANSITION": "Unknown transition",
}

METRICS = [
    ("adjusted_rand_index", "ARI"),
    ("normalized_mutual_information", "NMI"),
    ("macro_f1", "Macro-F1"),
    ("symmetric_segment_iou", "Symmetric segment IoU"),
    ("boundary_f1_24h", "Boundary-F1@24h"),
]


def value_text(mean: float, std: float, *, bold: bool = False) -> str:
    text = f"{mean:.4f} ± {std:.4f}"
    return f"**{text}**" if bold else text


def latex_value(mean: float, std: float, *, bold: bool = False) -> str:
    text = f"{mean:.4f} $\\pm$ {std:.4f}"
    return f"\\textbf{{{text}}}" if bold else text


def main() -> None:
    summary = pd.read_csv(RESULT_DIR / "summary_by_system.csv")
    test = summary[
        summary["split"].eq("test") & summary["system"].isin(DISPLAY_ROWS)
    ].set_index("system").loc[DISPLAY_ROWS]
    per_class = pd.read_csv(RESULT_DIR / "per_class_summary.csv")
    per_class = per_class[
        per_class["split"].eq("test") & per_class["system"].isin(METHODS)
    ]
    build = json.loads((DATASET_DIR / "build_summary.json").read_text(encoding="utf-8"))

    best = {
        metric: float(test.loc[METHODS, f"{metric}_mean"].max())
        for metric, _ in METRICS
    }
    lines = [
        "# TLE Observable-State Use Case: Figures and Results",
        "",
        "## Figure assets",
        "",
        "![Observable-state definition workflow](fig1_observable_state_definition.png)",
        "",
        "**Figure 1. Observable-state feature-rule relabeling workflow.** Candidate boundaries are "
        "inherited from the prior rule pipeline and are additionally split at target gaps longer "
        "than 36 h. Within each candidate segment, labels are assigned only from six observable "
        "fragment statistics using the first satisfied predicate in the ordered rule set. This is "
        "feature-rule relabeling within inherited candidates, not an intent-free segmentation pipeline. "
        "Counts refer to the full train+validation+test v2 label set.",
        "",
        "![Illustrative fragments for all six states](fig2_observable_state_examples.png)",
        "",
        "**Figure 2. Illustrative, hand-selected TLE-observable fragments.** Each panel shows semimajor-axis "
        "change relative to the fragment start against true elapsed UTC time. Axes are independently "
        "scaled because state amplitudes span more than three orders of magnitude. Panels show "
        "(a) stable orbit, (b) sustained raise, (c) sustained decay, (d) offset plateau, "
        "(e) short excursion and return, and (f) unknown transition. Four examples come from "
        "STARLINK-2706; the complete data contain only one offset-plateau fragment and five "
        "short-excursion fragments, so these two panels are qualitative cases rather than population summaries.",
        "",
        "![Comparison of three clustering approaches](fig3_method_comparison.png)",
        "",
        "**Figure 3. Retrospective K=3 comparison under the v2 observable-state proxy labels.** "
        "Only stable, sustained-raise, and sustained-decay classes enter quantitative evaluation; "
        "cluster-to-class Hungarian mappings are fit using training labels only. Panels report "
        "(a) clustering and mapped classification scores, (b) per-state F1 with test support, "
        "(c) symmetric segment IoU and boundary F1, and (d) predicted boundary counts. Bars show "
        "mean and standard deviation over seeds 42/43/44 on the same retrospective slice of "
        "10,209 windows and 376 NORAD targets. Decay has 101 windows and the slice has only 8 "
        "proxy-label boundaries. Gray diamonds in (c) show the constant-stable control (segment "
        "IoU 0.8705, boundary F1 0); high segment IoU is therefore strongly influenced by the "
        "9,135 stable windows. Labels describe observable trajectory shape, not operational intent.",
        "",
        "## Dataset composition",
        "",
        "Counts in this table cover the complete train+validation+test v2 dataset, not only the test slice.",
        "",
        "| Observable state | Fragments | Windows | Evaluation role |",
        "|---|---:|---:|---|",
    ]
    roles = {
        "STABLE_ORBIT": "Primary quantitative",
        "SUSTAINED_RAISE": "Primary quantitative",
        "SUSTAINED_DECAY": "Primary quantitative",
        "OFFSET_PLATEAU": "Qualitative case",
        "SHORT_EXCURSION_RETURN": "Qualitative case",
        "UNKNOWN_TRANSITION": "Abstention / excluded from mapping",
    }
    state_order = list(STATE_LABELS)
    for state in state_order:
        lines.append(
            f"| {STATE_LABELS[state]} (`{state}`) | "
            f"{build['label_fragment_counts'][state]:,} | "
            f"{build['label_sample_counts'][state]:,} | {roles[state]} |"
        )

    lines.extend(
        [
            "",
            "## Overall test performance",
            "",
            "| Method | ARI | NMI | Macro-F1 | Symmetric segment IoU | Boundary-F1@24h | Predicted boundaries |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in DISPLAY_ROWS:
        row = test.loc[method]
        values = []
        for metric, _ in METRICS:
            mean = float(row[f"{metric}_mean"])
            std = float(row[f"{metric}_std"])
            if method == "constant_stable":
                values.append(f"{mean:.4f}")
            else:
                values.append(value_text(mean, std, bold=abs(mean - best[metric]) < 1e-12))
        if method == "constant_stable":
            boundaries = f"{float(row['predicted_boundaries_24h_mean']):.0f}"
        else:
            boundaries = (
                f"{float(row['predicted_boundaries_24h_mean']):.1f} ± "
                f"{float(row['predicted_boundaries_24h_std']):.1f}"
            )
        lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(values + [boundaries]) + " |")

    lines.extend(
        [
            "",
            "The constant-stable control is deterministic rather than a three-seed estimate. The test "
            "slice contains 8 proxy-label primary-state boundaries. Higher is better for "
            "all score columns; predicted-boundary count is diagnostic and should be interpreted "
            "relative to 8 rather than maximized. The control's 0.8705 segment IoU shows that this "
            "metric is strongly affected by the dominance of stable windows.",
            "",
            "## Per-state test performance",
            "",
            "| Method | State | Support | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    class_order = ["STABLE_ORBIT", "SUSTAINED_RAISE", "SUSTAINED_DECAY"]
    indexed = per_class.set_index(["system", "LABEL_NAME"])
    for method in METHODS:
        for state in class_order:
            row = indexed.loc[(method, state)]
            lines.append(
                f"| {METHOD_LABELS[method]} | {STATE_LABELS[state]} | {int(row['support']):,} | "
                f"{row['precision_mean']:.4f} ± {row['precision_std']:.4f} | "
                f"{row['recall_mean']:.4f} ± {row['recall_std']:.4f} | "
                f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Reporting boundary",
            "",
            "These values measure recovery of mutually exclusive **TLE-observable trajectory "
            "shapes**, not operator intent, maneuver cause, or an operational lifecycle phase. "
            "Because the v2 rules were organized after the original frozen test had been consumed, "
            "the numbers are a retrospective relabeling analysis rather than a new unseen final test. "
            "Boundary metrics are diagnostic because the test slice contains only 8 proxy-label boundaries.",
            "",
            "Raw sources: "
            "[`summary_by_system.csv`](../../reports/observable_regime_feature_labels_v2/retest/summary_by_system.csv), "
            "[`per_class_summary.csv`](../../reports/observable_regime_feature_labels_v2/retest/per_class_summary.csv), "
            "and [`metrics_long.csv`](../../reports/observable_regime_feature_labels_v2/retest/metrics_long.csv).",
            "",
        ]
    )
    (FIGURE_DIR / "RESULTS_TABLE.md").write_text("\n".join(lines), encoding="utf-8")

    latex_lines = [
        "% Auto-generated by gen_results_table.py; requires booktabs.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Retrospective $K=3$ clustering performance on the three quantitative "
        "TLE-observable proxy classes. Cluster-to-class Hungarian mappings use training labels only. "
        "All methods use the same 10,209-window, 376-target test slice; decay has 101 windows and "
        "boundary scores use only eight proxy-label boundaries. Learned-method values are mean "
        "$\\pm$ standard deviation over three seeds. The labels describe observable shape, not operational intent.}",
        "\\label{tab:tle-observable-clustering}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & ARI & NMI & Macro-F1 & Sym. segment IoU & Boundary-F1@24h & Pred. boundaries \\\\",
        "\\midrule",
    ]
    latex_names = {
        "constant_stable": "Constant stable (control)",
        "raw_pca_kmeans": "Raw + KMeans",
        "pelt_kmeans": "PELT + KMeans",
        "transformer_embedding_kmeans": "Transformer + KMeans",
    }
    for method in DISPLAY_ROWS:
        row = test.loc[method]
        values = []
        for metric, _ in METRICS:
            mean = float(row[f"{metric}_mean"])
            std = float(row[f"{metric}_std"])
            if method == "constant_stable":
                values.append(f"{mean:.3f}")
            else:
                text = f"{mean:.3f} $\\pm$ {std:.3f}"
                values.append(f"\\textbf{{{text}}}" if abs(mean - best[metric]) < 1e-12 else text)
        boundary_mean = float(row["predicted_boundaries_24h_mean"])
        boundary_std = float(row["predicted_boundaries_24h_std"])
        values.append(
            f"{boundary_mean:.0f}" if method == "constant_stable" else f"{boundary_mean:.1f} $\\pm$ {boundary_std:.1f}"
        )
        latex_lines.append(latex_names[method] + " & " + " & ".join(values) + " \\\\")
    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{2pt}",
            "\\parbox{0.98\\textwidth}{\\footnotesize The test slice contains 10,209 primary-state "
            "windows from 376 NORAD targets and 8 proxy-label boundaries. The deterministic "
            "constant-stable control attains 0.8705 segment IoU, showing this metric's sensitivity "
            "to the 9,135 stable windows.}",
            "\\end{table*}",
            "",
        ]
    )
    (FIGURE_DIR / "TABLE_clustering_results.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )

    include_lines = [
        "% Auto-generated figure include snippets.",
        "\\begin{figure*}[t]",
        "  \\centering",
        "  \\includegraphics[width=\\textwidth]{figures/tle_state_use_case/fig1_observable_state_definition.pdf}",
        "  \\caption{Observable-state feature-rule relabeling workflow. Candidate boundaries are "
        "inherited from the prior rule pipeline and additionally split at target gaps longer than "
        "36 h. Within each candidate, the state label uses only observable fragment statistics and "
        "the first satisfied predicate in the ordered rule set. Thus this is feature-only relabeling "
        "within inherited candidates, not an intent-free segmentation pipeline. Counts cover the "
        "full train+validation+test v2 label set.}",
        "  \\label{fig:tle-state-definition}",
        "\\end{figure*}",
        "",
        "\\begin{figure*}[t]",
        "  \\centering",
        "  \\includegraphics[width=\\textwidth]{figures/tle_state_use_case/fig2_observable_state_examples.pdf}",
        "  \\caption{Illustrative, hand-selected TLE-observable fragments. Each panel shows semimajor-axis "
        "change relative to the fragment start against true elapsed time; panel axes are independently "
        "scaled. Panels show (a) stable orbit, (b) sustained raise, (c) sustained decay, "
        "(d) offset plateau, (e) short excursion and return, and (f) unknown transition. Four panels "
        "use STARLINK-2706; only one offset-plateau and five short-excursion fragments exist, so "
        "those panels are qualitative cases rather than population summaries.}",
        "  \\label{fig:tle-state-examples}",
        "\\end{figure*}",
        "",
        "\\begin{figure*}[t]",
        "  \\centering",
        "  \\includegraphics[width=\\textwidth]{figures/tle_state_use_case/fig3_method_comparison.pdf}",
        "  \\caption{Retrospective $K=3$ comparison on the stable, sustained-raise, and sustained-decay "
        "TLE-observable proxy classes. Cluster-to-class Hungarian mappings use training labels only. "
        "Panels report (a) clustering and mapped classification scores, (b) per-state F1 and support, "
        "(c) symmetric segment IoU and boundary F1 with gray diamonds for the constant-stable control, "
        "and (d) predicted boundary counts. Bars show mean $\\pm$ standard deviation over seeds 42--44 "
        "on the same retrospective 10,209-window, 376-target slice. Decay has 101 windows and only "
        "eight proxy-label boundaries exist. Segment IoU is strongly stable-class dominated; the labels "
        "describe observable trajectory shape, not operational intent.}",
        "  \\label{fig:tle-method-comparison}",
        "\\end{figure*}",
        "",
        "\\input{figures/tle_state_use_case/TABLE_clustering_results.tex}",
        "",
    ]
    (FIGURE_DIR / "latex_includes.tex").write_text("\n".join(include_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
