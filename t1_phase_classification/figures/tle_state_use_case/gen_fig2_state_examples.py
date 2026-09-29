#!/usr/bin/env python3
"""Plot one real TLE fragment for each observable state."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import FIGURE_DIR, configure, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from plot_tle_observable_state_division import build_catalog, load_points  # noqa: E402


DATASET_DIR = PROJECT_ROOT / "output/tle_observable_regime_feature_labels_v2"
METADATA_PATH = FIGURE_DIR / "state_metadata.csv"
EXAMPLES_PATH = FIGURE_DIR / "state_examples.csv"


def main() -> None:
    configure()
    metadata = pd.read_csv(METADATA_PATH).set_index("state")
    examples = pd.read_csv(EXAMPLES_PATH)
    points = load_points(DATASET_DIR)
    catalog, fragments = build_catalog(points)
    catalog = catalog.set_index("FRAGMENT_ID")

    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.65))
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.97, wspace=0.34, hspace=0.40)
    selected_rows: list[dict[str, object]] = []

    for panel_index, (axis, (_, selection)) in enumerate(zip(axes.flat, examples.iterrows())):
        state = selection["state"]
        fragment_id = selection["fragment_id"]
        if fragment_id not in fragments or fragment_id not in catalog.index:
            raise KeyError(f"Selected fragment is missing: {fragment_id}")
        fragment = fragments[fragment_id].sort_values("TARGET_TIME")
        record = catalog.loc[fragment_id]
        style = metadata.loc[state]
        color = style["color"]

        elapsed_days = (
            fragment["TARGET_TIME"] - fragment["TARGET_TIME"].iloc[0]
        ).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        relative_a = fragment["SEMI_MAJOR_AXIS_KM"].to_numpy(dtype=float)
        relative_a = relative_a - relative_a[0]

        axis.plot(elapsed_days, relative_a, color=color, linewidth=1.6, zorder=2)
        axis.scatter(
            elapsed_days,
            relative_a,
            s=12,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        axis.axhline(0.0, color="#98A5AC", linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
        axis.grid(axis="y", color="#E2E7EA", linewidth=0.55)
        axis.text(
            0.02,
            0.96,
            f"({chr(97 + panel_index)})  {style['display_name']}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            color="#20262A",
        )
        axis.text(
            0.02,
            0.07,
            f"{record['OBJECT_NAME']}  ·  N={int(record['SAMPLE_COUNT'])}\n"
            f"T={record['DURATION_DAYS']:.1f} d,  Δa={record['NET_A_CHANGE_KM']:+.2f} km,  "
            f"Rₐ={record['A_RANGE_KM']:.2f} km",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="#455A64",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        )
        span = max(float(np.ptp(relative_a)), 0.04)
        lower = float(relative_a.min()) - 0.14 * span
        upper = float(relative_a.max()) + 0.23 * span
        axis.set_ylim(lower, upper)
        axis.margins(x=0.04)
        if panel_index % 3 == 0:
            axis.set_ylabel("Relative semimajor axis (km)")
        if panel_index >= 3:
            axis.set_xlabel("Elapsed time (days)")

        selected_rows.append(
            {
                "panel": chr(97 + panel_index),
                "state": state,
                "display_name": style["display_name"],
                "fragment_id": fragment_id,
                "norad_cat_id": int(record["NORAD_CAT_ID"]),
                "object_name": record["OBJECT_NAME"],
                "sample_count": int(record["SAMPLE_COUNT"]),
                "duration_days": float(record["DURATION_DAYS"]),
                "net_a_change_km": float(record["NET_A_CHANGE_KM"]),
                "a_range_km": float(record["A_RANGE_KM"]),
                "selection_note": selection["selection_note"],
            }
        )

    save_figure(figure, "fig2_observable_state_examples")
    pd.DataFrame(selected_rows).to_csv(FIGURE_DIR / "selected_state_examples.csv", index=False)
    plt.close(figure)


if __name__ == "__main__":
    main()
