#!/usr/bin/env python3
"""Classify TLE fragments by observable features and plot the state division.

The classifier consumes already proposed contiguous fragments.  Operational
intent names are retained only as provenance columns and are never used by the
state decision rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "output/tle_observable_regime_testset"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/observable_regime_demo/observable_state_division.png"

MAX_GAP_HOURS = 36.0
STATE_ORDER = (
    "STABLE_ORBIT",
    "SUSTAINED_RAISE",
    "SUSTAINED_DECAY",
    "OFFSET_PLATEAU",
    "SHORT_EXCURSION_RETURN",
    "UNKNOWN_TRANSITION",
)
STATE_STYLE = {
    "STABLE_ORBIT": ("稳定轨道", "#277DA1", "o"),
    "SUSTAINED_RAISE": ("持续升高", "#F28E2B", "^"),
    "SUSTAINED_DECAY": ("持续降低", "#7B61A8", "v"),
    "OFFSET_PLATEAU": ("长时偏移平台", "#2A9D8F", "s"),
    "SHORT_EXCURSION_RETURN": ("短时偏移后回归", "#D1495B", "D"),
    "UNKNOWN_TRANSITION": ("不确定过渡", "#7A7F87", "X"),
}
PREFERRED_EXAMPLES = {
    "STABLE_ORBIT": "48481:4",
    "SUSTAINED_RAISE": "48481:1",
    "SUSTAINED_DECAY": "44249:1",
    "OFFSET_PLATEAU": "48481:5",
    "SHORT_EXCURSION_RETURN": "48481:3",
    "UNKNOWN_TRANSITION": "47168:2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def state_from_features(
    sample_count: int,
    duration_days: float,
    net_change_km: float,
    range_km: float,
    positive_fraction: float,
    negative_fraction: float,
) -> tuple[str, str]:
    """Return a mutually exclusive observable state and the matching rule."""
    if (
        sample_count >= 3
        and duration_days < 10.0
        and range_km >= 4.0
        and abs(net_change_km) <= 1.0
        and positive_fraction >= 0.25
        and negative_fraction >= 0.25
    ):
        return "SHORT_EXCURSION_RETURN", "T<10d; R_a>=4km; |Delta a|<=1km; p+/- >=0.25"
    if (
        sample_count >= 5
        and duration_days >= 10.0
        and range_km >= 5.0
        and abs(net_change_km) <= 1.0
        and positive_fraction >= 0.30
        and negative_fraction >= 0.30
    ):
        return "OFFSET_PLATEAU", "T>=10d; R_a>=5km; |Delta a|<=1km; p+/- >=0.30"
    if (
        sample_count >= 5
        and duration_days >= 2.0
        and net_change_km >= 4.0
        and positive_fraction >= 0.55
    ):
        return "SUSTAINED_RAISE", "T>=2d; Delta a>=4km; p+>=0.55"
    if (
        sample_count >= 5
        and duration_days >= 3.0
        and net_change_km <= -0.9
        and negative_fraction >= 0.70
    ):
        return "SUSTAINED_DECAY", "T>=3d; Delta a<=-0.9km; p->=0.70"
    if sample_count >= 5 and range_km <= 2.5 and abs(net_change_km) < 0.9:
        return "STABLE_ORBIT", "N>=5; R_a<=2.5km; |Delta a|<0.9km"
    return "UNKNOWN_TRANSITION", "no exclusive rule matched"


def load_points(dataset_dir: Path) -> pd.DataFrame:
    index = pd.read_csv(dataset_dir / "dataset_index.csv")
    schema = json.loads((dataset_dir / "feature_schema.json").read_text(encoding="utf-8"))
    feature_names = list(schema["feature_names"])
    semi_major_index = feature_names.index("semi_major_axis_km")
    da_dt_index = feature_names.index("da_dt_km_day")
    with np.load(dataset_dir / "tle_observable_regime_dataset.npz", allow_pickle=False) as archive:
        if len(index) != len(archive["next_raw"]):
            raise ValueError("dataset_index.csv and NPZ row counts differ")
        if not np.array_equal(index["SAMPLE_ID"].astype(str), archive["sample_ids"].astype(str)):
            raise ValueError("dataset_index.csv and NPZ SAMPLE_ID order differs")
        index["SEMI_MAJOR_AXIS_KM"] = archive["next_raw"][:, semi_major_index]
        index["DA_DT_KM_DAY"] = archive["next_raw"][:, da_dt_index]

    index["TARGET_TIME"] = pd.to_datetime(index["TARGET_EPOCH_UTC"], utc=True)
    index = index.sort_values(["NORAD_CAT_ID", "TARGET_TIME"]).reset_index(drop=True)
    gap_hours = index.groupby("NORAD_CAT_ID", sort=False)["TARGET_TIME"].diff().dt.total_seconds() / 3600.0
    new_fragment = (
        index["NORAD_CAT_ID"].ne(index["NORAD_CAT_ID"].shift())
        | index["LEGACY_SEGMENT_ID"].ne(index["LEGACY_SEGMENT_ID"].shift())
        | gap_hours.gt(MAX_GAP_HOURS)
    )
    index["FRAGMENT_NUMBER"] = new_fragment.cumsum()
    return index


def build_catalog(points: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records: list[dict[str, object]] = []
    fragments: dict[str, pd.DataFrame] = {}
    part_counter: dict[str, int] = {}
    for _, fragment in points.groupby("FRAGMENT_NUMBER", sort=False):
        fragment = fragment.sort_values("TARGET_TIME").copy()
        legacy_id = str(fragment["LEGACY_SEGMENT_ID"].iloc[0])
        part_counter[legacy_id] = part_counter.get(legacy_id, 0) + 1
        fragment_id = f"{legacy_id}:part{part_counter[legacy_id]}"
        duration_days = float(
            (fragment["TARGET_TIME"].iloc[-1] - fragment["TARGET_TIME"].iloc[0]).total_seconds()
            / 86400.0
        )
        values = fragment["SEMI_MAJOR_AXIS_KM"].to_numpy(dtype=float)
        slopes = fragment["DA_DT_KM_DAY"].to_numpy(dtype=float)
        net_change_km = float(values[-1] - values[0])
        range_km = float(values.max() - values.min())
        positive_fraction = float(np.mean(slopes > 0.0))
        negative_fraction = float(np.mean(slopes < 0.0))
        gaps = fragment["TARGET_TIME"].diff().dt.total_seconds().div(3600.0).dropna()
        max_gap_hours = float(gaps.max()) if len(gaps) else 0.0
        state, rule = state_from_features(
            len(fragment),
            duration_days,
            net_change_km,
            range_km,
            positive_fraction,
            negative_fraction,
        )
        records.append(
            {
                "FRAGMENT_ID": fragment_id,
                "NORAD_CAT_ID": int(fragment["NORAD_CAT_ID"].iloc[0]),
                "OBJECT_NAME": str(fragment["OBJECT_NAME"].iloc[0]),
                "START_UTC": fragment["TARGET_TIME"].iloc[0].isoformat(),
                "END_UTC": fragment["TARGET_TIME"].iloc[-1].isoformat(),
                "SAMPLE_COUNT": len(fragment),
                "DURATION_DAYS": duration_days,
                "NET_A_CHANGE_KM": net_change_km,
                "A_RANGE_KM": range_km,
                "POSITIVE_DA_FRACTION": positive_fraction,
                "NEGATIVE_DA_FRACTION": negative_fraction,
                "MAX_TARGET_GAP_HOURS": max_gap_hours,
                "OBSERVABLE_STATE": state,
                "MATCHED_RULE": rule,
                "LEGACY_SEGMENT_ID": legacy_id,
                "LEGACY_LABEL_NAME": str(fragment["LEGACY_LABEL_NAME"].iloc[0]),
            }
        )
        fragments[fragment_id] = fragment
    return pd.DataFrame.from_records(records), fragments


def select_example(catalog: pd.DataFrame, state: str) -> pd.Series:
    preferred = catalog[
        catalog["OBSERVABLE_STATE"].eq(state)
        & catalog["LEGACY_SEGMENT_ID"].eq(PREFERRED_EXAMPLES[state])
    ]
    if len(preferred):
        return preferred.sort_values("SAMPLE_COUNT", ascending=False).iloc[0]
    candidates = catalog[catalog["OBSERVABLE_STATE"].eq(state)]
    if candidates.empty:
        raise RuntimeError(f"No representative fragment for {state}")
    return candidates.sort_values(["SAMPLE_COUNT", "A_RANGE_KM"], ascending=False).iloc[0]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#263238",
            "xtick.color": "#52616B",
            "ytick.color": "#52616B",
            "text.color": "#17252E",
        }
    )


def plot(catalog: pd.DataFrame, fragments: dict[str, pd.DataFrame], output_path: Path) -> None:
    configure_style()
    figure = plt.figure(figsize=(16, 10), facecolor="#F7F9FA")
    grid = figure.add_gridspec(
        3,
        4,
        left=0.055,
        right=0.988,
        bottom=0.07,
        top=0.88,
        wspace=0.32,
        hspace=0.42,
        width_ratios=[1.35, 1.35, 1.0, 1.0],
    )
    scatter_axis = figure.add_subplot(grid[:, :2])

    for state in STATE_ORDER:
        selected = catalog[catalog["OBSERVABLE_STATE"].eq(state)]
        if selected.empty:
            continue
        label_zh, color, marker = STATE_STYLE[state]
        marker_sizes = np.clip(20.0 + 18.0 * np.log1p(selected["A_RANGE_KM"]), 22.0, 130.0)
        alpha = 0.20 if state == "STABLE_ORBIT" else 0.72
        scatter_axis.scatter(
            selected["DURATION_DAYS"].clip(lower=0.08),
            selected["NET_A_CHANGE_KM"],
            s=marker_sizes,
            c=color,
            marker=marker,
            alpha=alpha,
            linewidths=0.35,
            edgecolors="white",
            rasterized=True,
            label=f"{label_zh}  n={len(selected):,}",
        )

    scatter_axis.axhspan(-0.9, 0.9, color="#277DA1", alpha=0.055, zorder=0)
    scatter_axis.axhline(4.0, color="#F28E2B", lw=0.9, ls=(0, (4, 4)), alpha=0.8)
    scatter_axis.axhline(-0.9, color="#7B61A8", lw=0.9, ls=(0, (4, 4)), alpha=0.8)
    scatter_axis.axvline(10.0, color="#52616B", lw=0.9, ls=(0, (2, 4)), alpha=0.55)
    scatter_axis.set_xscale("log")
    scatter_axis.set_yscale("symlog", linthresh=0.8, linscale=0.9)
    scatter_axis.set_xlabel("片段持续时间 T（天，对数刻度）")
    scatter_axis.set_ylabel("半长轴净变化 Δa（km，对称对数刻度）")
    scatter_axis.set_title("A  全部 TLE 片段的特征划分", loc="left", pad=12)
    scatter_axis.grid(True, which="major", color="#D7DEE2", lw=0.7, alpha=0.8)
    scatter_axis.grid(True, which="minor", axis="x", color="#E7ECEF", lw=0.5, alpha=0.65)
    scatter_axis.spines[["top", "right"]].set_visible(False)
    scatter_axis.text(
        0.02,
        0.985,
        "点面积 ∝ 半长轴变化范围 R_a\n灰色表示未命中任一互斥规则",
        transform=scatter_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#52616B",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=STATE_STYLE[state][2],
            color="none",
            markerfacecolor=STATE_STYLE[state][1],
            markeredgecolor="white",
            markersize=8,
            label=f"{STATE_STYLE[state][0]}  n={(catalog['OBSERVABLE_STATE'] == state).sum():,}",
        )
        for state in STATE_ORDER
    ]
    scatter_axis.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        ncol=2,
        fontsize=9.5,
        handletextpad=0.5,
        columnspacing=1.1,
    )

    for position, state in enumerate(STATE_ORDER):
        axis = figure.add_subplot(grid[position // 2, 2 + position % 2])
        record = select_example(catalog, state)
        fragment = fragments[str(record.FRAGMENT_ID)]
        times = fragment["TARGET_TIME"]
        duration = max((times.iloc[-1] - times.iloc[0]).total_seconds(), 1.0)
        progress = (times - times.iloc[0]).dt.total_seconds().to_numpy() / duration * 100.0
        delta_a = fragment["SEMI_MAJOR_AXIS_KM"].to_numpy(dtype=float)
        delta_a -= delta_a[0]
        label_zh, color, _ = STATE_STYLE[state]
        axis.plot(progress, delta_a, color=color, lw=2.0, marker="o", ms=2.8)
        axis.axhline(0.0, color="#AAB5BC", lw=0.7, zorder=0)
        axis.set_xlim(-3.0, 103.0)
        padding = max(float(np.ptp(delta_a)) * 0.15, 0.08)
        axis.set_ylim(float(delta_a.min()) - padding, float(delta_a.max()) + padding)
        axis.set_title(f"{label_zh} · {state}", loc="left", fontsize=10.5, pad=6)
        metadata_at_bottom = state in {"OFFSET_PLATEAU", "SHORT_EXCURSION_RETURN"}
        axis.text(
            0.01,
            0.04 if metadata_at_bottom else 0.96,
            f"{record.OBJECT_NAME} / {int(record.NORAD_CAT_ID)}\n"
            f"T={record.DURATION_DAYS:.1f} d  Δa={record.NET_A_CHANGE_KM:+.2f} km  "
            f"R_a={record.A_RANGE_KM:.2f} km",
            transform=axis.transAxes,
            ha="left",
            va="bottom" if metadata_at_bottom else "top",
            fontsize=8.2,
            color="#52616B",
            bbox={"facecolor": "#F7F9FA", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        )
        axis.set_xlabel("片段进度（%）", fontsize=8.8)
        axis.set_ylabel("相对起点 Δa（km）", fontsize=8.8)
        axis.tick_params(labelsize=8)
        axis.grid(True, color="#E1E6E9", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    window_counts = catalog.groupby("OBSERVABLE_STATE")["SAMPLE_COUNT"].sum()
    figure.suptitle(
        "TLE 数据片段的可观测运动状态划分",
        fontsize=22,
        fontweight="bold",
        x=0.03,
        y=0.974,
        ha="left",
    )
    figure.text(
        0.03,
        0.925,
        f"{len(catalog):,} 个连续片段 / {int(window_counts.sum()):,} 个 5→1 TLE 窗口；"
        "只按持续时间、半长轴净变化、变化范围和方向一致率分类",
        fontsize=11,
        color="#52616B",
        ha="left",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, facecolor=figure.get_facecolor(), bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_path = args.out.resolve()
    points = load_points(dataset_dir)
    catalog, fragments = build_catalog(points)
    catalog_path = output_path.with_name("tle_fragment_state_catalog.csv")
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)
    plot(catalog, fragments, output_path)
    counts = catalog.groupby("OBSERVABLE_STATE")["SAMPLE_COUNT"].agg(["count", "sum"])
    print(counts.reindex(STATE_ORDER).fillna(0).astype(int).to_string())
    print(f"Wrote {catalog_path}")
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
