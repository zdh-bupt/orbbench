#!/usr/bin/env python3
"""Generate the observable-state definition workflow for the paper use case."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_plot_style import FIGURE_DIR, configure, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = (
    PROJECT_ROOT / "output/tle_observable_regime_feature_labels_v2/fragment_catalog.csv"
)
METADATA_PATH = FIGURE_DIR / "state_metadata.csv"


def rounded_box(axis, x, y, width, height, *, face, edge, radius=1.6, linewidth=1.0):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.45,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    axis.add_patch(patch)
    return patch


def arrow(axis, start, end, *, color="#5F6B73", linewidth=1.1):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
            connectionstyle="arc3,rad=0",
        )
    )


def main() -> None:
    configure()
    catalog = pd.read_csv(CATALOG_PATH)
    metadata = pd.read_csv(METADATA_PATH).sort_values("priority")
    fragment_counts = catalog.groupby("OBSERVABLE_STATE").size()
    window_counts = catalog.groupby("OBSERVABLE_STATE")["SAMPLE_COUNT"].sum()

    figure, axis = plt.subplots(figsize=(7.2, 5.35))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.axis("off")

    stage_labels = [
        (13.0, "1  Candidate fragmentation"),
        (40.5, "2  Observable features"),
        (67.0, "3  Ordered assignment"),
        (88.2, "Observable state"),
    ]
    for x, label in stage_labels:
        axis.text(x, 97.2, label, ha="center", va="top", fontsize=9.2, fontweight="bold")

    rounded_box(axis, 1.5, 35.5, 23.0, 31.0, face="#F4F7F9", edge="#8FA3AF")
    axis.text(13.0, 61.5, "TLE sequence in a candidate segment", ha="center", va="center", fontsize=8.5)
    xs = [4.5, 7.5, 10.5, 13.5, 16.5, 21.5]
    ys = [52.0, 54.0, 51.0, 55.0, 53.0, 56.0]
    axis.plot(xs[:5], ys[:5], color="#0072B2", linewidth=1.3)
    axis.scatter(xs[:5], ys[:5], s=19, color="#0072B2", edgecolor="white", linewidth=0.45, zorder=3)
    axis.plot([19.3, 20.4], [54.6, 55.3], color="#8FA3AF", linewidth=0.8, linestyle="--")
    axis.scatter(xs[5:], ys[5:], s=19, color="#0072B2", edgecolor="white", linewidth=0.45, zorder=3)
    axis.text(13.0, 46.1, "inherited candidate boundaries", ha="center", va="center", fontsize=8.2)
    axis.text(13.0, 40.8, "split again when target gap > 36 h", ha="center", va="center", fontsize=8.0, color="#455A64")

    rounded_box(axis, 29.0, 35.5, 23.0, 31.0, face="#F4F7F9", edge="#8FA3AF")
    axis.text(40.5, 61.0, "Fragment-level statistics", ha="center", va="center", fontsize=8.8)
    feature_lines = [
        r"$N$: valid windows",
        r"$T$: duration (days)",
        r"$\Delta a$: net semimajor-axis change",
        r"$R_a$: semimajor-axis range",
        r"$p_+, p_-$: directional consistency",
    ]
    axis.text(
        31.0,
        55.5,
        "\n".join(feature_lines),
        ha="left",
        va="top",
        fontsize=8.1,
        linespacing=1.45,
    )
    axis.text(
        40.5,
        39.8,
        "State predicates use only observable features",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#455A64",
        fontstyle="italic",
    )

    rounded_box(axis, 56.5, 35.5, 17.5, 31.0, face="#FFF8E7", edge="#C89F35")
    axis.text(65.25, 59.5, "First matching rule", ha="center", va="center", fontsize=8.8, fontweight="bold")
    axis.text(65.25, 52.0, "priority 1 → 5", ha="center", va="center", fontsize=8.6)
    axis.text(65.25, 45.3, "otherwise", ha="center", va="center", fontsize=8.1, color="#455A64")
    axis.text(65.25, 40.4, "UNKNOWN", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#555555")

    arrow(axis, (24.5, 51.0), (29.0, 51.0))
    arrow(axis, (52.0, 51.0), (56.5, 51.0))

    card_x = 76.7
    card_width = 22.4
    card_height = 11.5
    centers = [88.6, 72.8, 57.0, 41.2, 25.4, 9.6]
    decision_exit = (74.0, 51.0)
    for (_, row), center_y in zip(metadata.iterrows(), centers):
        state = row["state"]
        color = row["color"]
        rounded_box(
            axis,
            card_x,
            center_y - card_height / 2,
            card_width,
            card_height,
            face="#FFFFFF",
            edge=color,
            radius=1.25,
            linewidth=1.25,
        )
        axis.add_patch(
            FancyBboxPatch(
                (card_x + 0.35, center_y - card_height / 2 + 0.35),
                1.45,
                card_height - 0.7,
                boxstyle="round,pad=0.0,rounding_size=0.55",
                facecolor=color,
                edgecolor=color,
                linewidth=0,
            )
        )
        axis.text(
            card_x + 2.5,
            center_y + 3.3,
            f"{int(row['priority'])}. {row['display_name']}",
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#20262A",
        )
        rule = "\n".join(textwrap.wrap(str(row["rule"]), width=31))
        axis.text(
            card_x + 2.5,
            center_y - 0.2,
            rule,
            ha="left",
            va="center",
            fontsize=6.8,
            color="#39464D",
            linespacing=1.1,
        )
        fragments = int(fragment_counts.get(state, 0))
        windows = int(window_counts.get(state, 0))
        axis.text(
            card_x + 21.4,
            center_y - 4.0,
            f"{fragments:,} fragments / {windows:,} windows",
            ha="right",
            va="center",
            fontsize=6.4,
            color="#61717A",
        )
        branch_x = 75.5
        axis.plot(
            [decision_exit[0], branch_x, branch_x],
            [decision_exit[1], decision_exit[1], center_y],
            color="#86939A",
            linewidth=0.75,
        )
        arrow(axis, (branch_x, center_y), (card_x, center_y), color="#86939A", linewidth=0.8)

    axis.text(
        1.5,
        0.5,
        f"Dataset: {len(catalog):,} fragments and {int(catalog['SAMPLE_COUNT'].sum()):,} TLE windows. "
        "Feature-rule relabeling is performed within inherited legacy candidate segments.",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#455A64",
    )

    save_figure(figure, "fig1_observable_state_definition")
    plt.close(figure)


if __name__ == "__main__":
    main()
