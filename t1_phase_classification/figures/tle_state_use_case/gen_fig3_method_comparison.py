#!/usr/bin/env python3
"""Render the three-method comparison in the requested compact line-panel style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from paper_plot_style import configure, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "reports/observable_regime_feature_labels_v2/retest"
SUMMARY_PATH = RESULT_DIR / "summary_by_system.csv"
PER_CLASS_PATH = RESULT_DIR / "per_class_summary.csv"

METHODS = ["raw_pca_kmeans", "pelt_kmeans", "transformer_embedding_kmeans"]
METHOD_LABELS = {
    "raw_pca_kmeans": "Raw PCA + KMeans",
    "pelt_kmeans": "PELT + KMeans",
    "transformer_embedding_kmeans": "Transformer embedding + KMeans",
}
SHORT_LABELS = {
    "raw_pca_kmeans": "Raw",
    "pelt_kmeans": "PELT",
    "transformer_embedding_kmeans": "Transformer",
}
METHOD_COLORS = {
    "raw_pca_kmeans": "#0072B2",
    "pelt_kmeans": "#E69F00",
    "transformer_embedding_kmeans": "#009E73",
}
METHOD_MARKERS = {
    "raw_pca_kmeans": "o",
    "pelt_kmeans": "s",
    "transformer_embedding_kmeans": "^",
}
METHOD_LINESTYLES = {
    "raw_pca_kmeans": "-",
    "pelt_kmeans": "--",
    "transformer_embedding_kmeans": ":",
}


def style_panel(axis) -> None:
    axis.set_facecolor("#E5E6E8")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#737A7E")
        spine.set_linewidth(0.75)
    axis.tick_params(width=0.75, length=3.0, color="#4D555A")
    axis.grid(False)


def method_profile(axis, frame, metrics, tick_labels, *, ylim) -> None:
    positions = np.arange(len(metrics), dtype=float)
    for method in METHODS:
        row = frame.loc[method]
        means = np.array([float(row[f"{metric}_mean"]) for metric in metrics])
        stds = np.array([float(row[f"{metric}_std"]) for metric in metrics])
        axis.errorbar(
            positions,
            means,
            yerr=stds,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.15,
            marker=METHOD_MARKERS[method],
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            capsize=2.0,
            elinewidth=0.75,
            zorder=3,
        )
    axis.axhline(0.0, color="#747B7F", linewidth=0.55, zorder=1)
    axis.set_xticks(positions, tick_labels)
    axis.set_ylim(*ylim)
    axis.set_ylabel("Score")
    axis.margins(x=0.09)


def panel_caption(axis, label: str, text: str, *, y: float = -0.29) -> None:
    axis.text(
        0.5,
        y,
        f"({label}) {text}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        clip_on=False,
    )


def main() -> None:
    configure()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.3,
        }
    )

    all_summary = pd.read_csv(SUMMARY_PATH)
    test = all_summary[
        all_summary["split"].eq("test") & all_summary["system"].isin(METHODS)
    ].set_index("system").loc[METHODS]
    control = all_summary[
        all_summary["split"].eq("test") & all_summary["system"].eq("constant_stable")
    ].iloc[0]
    per_class = pd.read_csv(PER_CLASS_PATH)
    per_class = per_class[
        per_class["split"].eq("test") & per_class["system"].isin(METHODS)
    ]

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), facecolor="white")
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.86, wspace=0.28, hspace=0.70)
    for axis in axes.flat:
        style_panel(axis)

    method_profile(
        axes[0, 0],
        test,
        ["adjusted_rand_index", "normalized_mutual_information", "macro_f1"],
        ["ARI", "NMI", "Macro-F1"],
        ylim=(-0.07, 0.88),
    )
    panel_caption(axes[0, 0], "a", "Overall metrics")

    state_order = ["STABLE_ORBIT", "SUSTAINED_RAISE", "SUSTAINED_DECAY"]
    support_by_state = per_class.drop_duplicates("LABEL_NAME").set_index("LABEL_NAME")["support"]
    state_ticks = [
        f"{name}\n(n={int(support_by_state.loc[state]):,})"
        for name, state in zip(["Stable", "Raise", "Decay"], state_order)
    ]
    state_positions = np.arange(len(state_order), dtype=float)
    for method in METHODS:
        selected = per_class[per_class["system"].eq(method)].set_index("LABEL_NAME")
        means = np.array([float(selected.loc[state, "f1_mean"]) for state in state_order])
        stds = np.array([float(selected.loc[state, "f1_std"]) for state in state_order])
        axes[0, 1].errorbar(
            state_positions,
            means,
            yerr=stds,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.15,
            marker=METHOD_MARKERS[method],
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            capsize=2.0,
            elinewidth=0.75,
            zorder=3,
        )
    axes[0, 1].set_xticks(state_positions, state_ticks)
    axes[0, 1].set_ylim(-0.03, 1.04)
    axes[0, 1].set_ylabel("F1 score")
    axes[0, 1].margins(x=0.09)
    panel_caption(axes[0, 1], "b", "Per-state performance", y=-0.36)

    method_profile(
        axes[1, 0],
        test,
        ["symmetric_segment_iou", "boundary_f1_24h"],
        ["Symmetric\nsegment IoU", "Boundary F1\n@24 h"],
        ylim=(-0.05, 0.98),
    )
    control_values = [
        float(control["symmetric_segment_iou_mean"]),
        float(control["boundary_f1_24h_mean"]),
    ]
    axes[1, 0].plot(
        np.arange(2),
        control_values,
        color="#555B5F",
        linestyle=(0, (4, 3)),
        linewidth=0.9,
        marker="D",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="Constant-stable control",
        zorder=2,
    )
    axes[1, 0].legend(loc="upper right", frameon=True, facecolor="#E5E6E8", edgecolor="#777777")
    panel_caption(axes[1, 0], "c", "Segmentation quality", y=-0.36)

    axis = axes[1, 1]
    positions = np.arange(len(METHODS), dtype=float)
    boundary_means = test["predicted_boundaries_24h_mean"].to_numpy(dtype=float)
    boundary_stds = test["predicted_boundaries_24h_std"].to_numpy(dtype=float)
    axis.plot(positions, boundary_means, color="#666D71", linewidth=1.05, zorder=2)
    for position, method, mean, std in zip(positions, METHODS, boundary_means, boundary_stds):
        axis.errorbar(
            [position],
            [mean],
            yerr=[std],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.1,
            capsize=2.0,
            elinewidth=0.75,
            zorder=3,
        )
        axis.text(position, mean * 1.15, f"{mean:.0f}", ha="center", va="bottom", fontsize=7.4)
    proxy_boundaries = int(test["true_boundaries_24h"].iloc[0])
    axis.axhline(
        proxy_boundaries,
        color="#555B5F",
        linestyle=(0, (4, 3)),
        linewidth=0.9,
        label=f"Proxy-label boundaries = {proxy_boundaries}",
    )
    axis.set_yscale("log")
    axis.set_ylim(6.0, 2600.0)
    axis.set_xticks(positions, [SHORT_LABELS[method] for method in METHODS])
    axis.set_ylabel("Predicted boundaries (log scale)")
    axis.legend(loc="upper left", frameon=True, facecolor="#E5E6E8", edgecolor="#777777")
    axis.margins(x=0.09)
    panel_caption(axis, "d", "Boundary fragmentation", y=-0.29)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.15,
            marker=METHOD_MARKERS[method],
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.975),
        ncol=3,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.1,
    )

    save_figure(figure, "fig3_method_comparison")
    plt.close(figure)


if __name__ == "__main__":
    main()
