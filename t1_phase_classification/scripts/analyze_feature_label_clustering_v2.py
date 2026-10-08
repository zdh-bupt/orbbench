#!/usr/bin/env python3
"""Aggregate the feature-label v2 clustering retest into tables and findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "reports/observable_regime_feature_labels_v2/retest"
DEFAULT_V1 = PROJECT_ROOT / "reports/observable_regime_demo/final_test_once/final_test_summary.json"
SYSTEM_ORDER = [
    "constant_stable",
    "raw_pca_kmeans",
    "pelt_kmeans",
    "transformer_embedding_kmeans",
]
DISPLAY_NAME = {
    "constant_stable": "Constant stable",
    "raw_pca_kmeans": "Raw PCA + KMeans",
    "pelt_kmeans": "PELT + KMeans",
    "transformer_embedding_kmeans": "Transformer embedding + KMeans",
}
METRICS = [
    "adjusted_rand_index",
    "normalized_mutual_information",
    "macro_f1",
    "symmetric_segment_iou",
    "segment_macro_duration_iou",
    "boundary_f1_24h",
    "predicted_boundaries_24h",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--v1-summary", type=Path, default=DEFAULT_V1)
    return parser.parse_args()


def mean_std(values: pd.Series) -> tuple[float, float]:
    numeric = values.astype(float)
    return float(numeric.mean()), float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0


def aggregate_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (system, family, split), group in raw.groupby(["system", "family", "split"], sort=False):
        row: dict[str, object] = {
            "system": system,
            "display_name": DISPLAY_NAME[system],
            "family": family,
            "split": split,
            "seed_count": int(group["seed"].ge(0).sum()) or 1,
            "sample_count": int(group["sample_count"].iloc[0]),
            "satellite_count": int(group["satellite_count"].iloc[0]),
            "true_boundaries_24h": int(group["true_boundaries_24h"].iloc[0]),
        }
        for metric in METRICS:
            mean, std = mean_std(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    result = pd.DataFrame(rows)
    for split in result["split"].unique():
        split_rows = result["split"].eq(split)
        baseline = result.loc[
            split_rows & result["system"].eq("raw_pca_kmeans"), "macro_f1_mean"
        ].iloc[0]
        segment_baseline = result.loc[
            split_rows & result["system"].eq("raw_pca_kmeans"),
            "symmetric_segment_iou_mean",
        ].iloc[0]
        result.loc[split_rows, "macro_f1_relative_vs_raw_pct"] = (
            result.loc[split_rows, "macro_f1_mean"] / baseline - 1.0
        ) * 100.0
        result.loc[split_rows, "segment_iou_relative_vs_raw_pct"] = (
            result.loc[split_rows, "symmetric_segment_iou_mean"] / segment_baseline - 1.0
        ) * 100.0
    return result


def aggregate_per_class(results: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for system in ["raw_pca_kmeans", "pelt_kmeans", "transformer_embedding_kmeans"]:
        for seed_dir in sorted((results / system).glob("seed_*")):
            seed = int(seed_dir.name.split("_")[-1])
            for split in ["validation", "test"]:
                frame = pd.read_csv(seed_dir / f"evaluation_{split}/per_class.csv")
                frame.insert(0, "split", split)
                frame.insert(0, "seed", seed)
                frame.insert(0, "system", system)
                rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    aggregate = (
        raw.groupby(["system", "split", "LABEL_ID", "LABEL_NAME"], as_index=False)
        .agg(
            support=("support", "first"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .fillna(0.0)
    )
    aggregate.insert(1, "display_name", aggregate["system"].map(DISPLAY_NAME))
    return aggregate


def v1_comparison(summary: pd.DataFrame, v1_path: Path) -> pd.DataFrame:
    document = json.loads(v1_path.read_text(encoding="utf-8"))["segmentation"]
    v1_names = {
        "raw_pca_kmeans": ["raw_window"],
        "pelt_kmeans": ["pelt"],
        "transformer_embedding_kmeans": [
            "tle100_seed42",
            "tle100_seed43",
            "tle100_seed44",
        ],
    }
    rows: list[dict[str, object]] = []
    v2 = summary[summary["split"].eq("test")].set_index("system")
    for system, names in v1_names.items():
        old = [document[name] for name in names]
        old_values = {
            "adjusted_rand_index": np.mean([item["adjusted_rand_index"] for item in old]),
            "normalized_mutual_information": np.mean(
                [item["normalized_mutual_information"] for item in old]
            ),
            "macro_f1": np.mean([item["macro_f1"] for item in old]),
            "symmetric_segment_iou": np.mean(
                [
                    item["segments"]["symmetric_segment_iou_with_false_positive_penalty"]
                    for item in old
                ]
            ),
            "boundary_f1_24h": np.mean([item["boundary"]["24h"]["f1"] for item in old]),
        }
        for metric, old_value in old_values.items():
            new_value = float(v2.loc[system, f"{metric}_mean"])
            rows.append(
                {
                    "system": system,
                    "display_name": DISPLAY_NAME[system],
                    "metric": metric,
                    "v1_value": old_value,
                    "v2_value": new_value,
                    "absolute_change": new_value - old_value,
                    "relative_change_pct": (new_value / old_value - 1.0) * 100.0
                    if abs(old_value) > 1e-12
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def write_report(
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    test = summary[summary["split"].eq("test")].set_index("system")
    lines = [
        "# TLE 特征标签 v2 聚类复测结果",
        "",
        "## 评测口径",
        "",
        "本次复测保持 TLE 输入、NORAD 级 70/15/15 划分、无监督模型和三种子不变，",
        "仅将评测标签替换为互斥的数据特征状态标签。簇到状态的 Hungarian 映射只使用训练集。",
        "测试结果属于标签重组后的回顾性评测，不是新的未见冻结测试。",
        "",
        "主指标只包含 `STABLE_ORBIT`、`SUSTAINED_RAISE`、`SUSTAINED_DECAY`。",
        "测试主样本为 10,209 个窗口、376 个含主状态的目标、8 个主状态边界。",
        "",
        "## 原始结果表",
        "",
        "| 方法 | ARI | NMI | Macro-F1 | 对称 segment IoU | Boundary-F1@24h | Macro-F1 相对 Raw |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in ["constant_stable", "raw_pca_kmeans", "pelt_kmeans", "transformer_embedding_kmeans"]:
        row = test.loc[system]
        lines.append(
            f"| {DISPLAY_NAME[system]} | "
            f"{fmt(row.adjusted_rand_index_mean, row.adjusted_rand_index_std)} | "
            f"{fmt(row.normalized_mutual_information_mean, row.normalized_mutual_information_std)} | "
            f"{fmt(row.macro_f1_mean, row.macro_f1_std)} | "
            f"{fmt(row.symmetric_segment_iou_mean, row.symmetric_segment_iou_std)} | "
            f"{fmt(row.boundary_f1_24h_mean, row.boundary_f1_24h_std)} | "
            f"{row.macro_f1_relative_vs_raw_pct:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 测试集逐类 F1",
            "",
            "| 方法 | 稳定轨道 | 持续升高 | 持续降低 |",
            "|---|---:|---:|---:|",
        ]
    )
    test_class = per_class[per_class["split"].eq("test")]
    for system in ["raw_pca_kmeans", "pelt_kmeans", "transformer_embedding_kmeans"]:
        selected = test_class[test_class["system"].eq(system)].set_index("LABEL_NAME")
        values = [
            fmt(selected.loc[label, "f1_mean"], selected.loc[label, "f1_std"])
            for label in ["STABLE_ORBIT", "SUSTAINED_RAISE", "SUSTAINED_DECAY"]
        ]
        lines.append(f"| {DISPLAY_NAME[system]} | " + " | ".join(values) + " |")

    raw = test.loc["raw_pca_kmeans"]
    pelt = test.loc["pelt_kmeans"]
    transformer = test.loc["transformer_embedding_kmeans"]
    constant = test.loc["constant_stable"]
    raw_class = test_class[test_class["system"].eq("raw_pca_kmeans")].set_index("LABEL_NAME")
    pelt_class = test_class[test_class["system"].eq("pelt_kmeans")].set_index("LABEL_NAME")
    transformer_class = test_class[
        test_class["system"].eq("transformer_embedding_kmeans")
    ].set_index("LABEL_NAME")
    old_new_f1 = comparison[comparison["metric"].eq("macro_f1")].set_index("system")
    lines.extend(
        [
            "",
            "## 关键发现",
            "",
            f"1. **Raw PCA + KMeans 仍是最强方法。** 测试 ARI={raw.adjusted_rand_index_mean:.4f}、"
            f"NMI={raw.normalized_mutual_information_mean:.4f}、Macro-F1={raw.macro_f1_mean:.4f}。"
            f"稳定和持续升高 F1 分别为 {raw_class.loc['STABLE_ORBIT', 'f1_mean']:.4f}、"
            f"{raw_class.loc['SUSTAINED_RAISE', 'f1_mean']:.4f}。",
            "",
            f"2. **持续降低仍是主要薄弱点。** Raw 方法对该类 recall="
            f"{raw_class.loc['SUSTAINED_DECAY', 'recall_mean']:.4f}，但 precision="
            f"{raw_class.loc['SUSTAINED_DECAY', 'precision_mean']:.4f}，F1 只有 "
            f"{raw_class.loc['SUSTAINED_DECAY', 'f1_mean']:.4f}；说明许多稳定窗口被误报为降低。",
            "",
            f"3. **PELT 更平滑，但分类能力较弱。** 对称 segment IoU={pelt.symmetric_segment_iou_mean:.4f}，"
            f"比 Raw 高 {pelt.segment_iou_relative_vs_raw_pct:+.2f}%，但 Macro-F1 低 "
            f"{abs(pelt.macro_f1_relative_vs_raw_pct):.2f}%，并且持续降低 F1="
            f"{pelt_class.loc['SUSTAINED_DECAY', 'f1_mean']:.4f}。高 segment IoU 主要受稳定长段占比驱动。",
            "",
            f"4. **Transformer 自监督预测表征没有形成可用状态簇。** ARI="
            f"{transformer.adjusted_rand_index_mean:.4f}、NMI={transformer.normalized_mutual_information_mean:.4f}，"
            f"Macro-F1={transformer.macro_f1_mean:.4f}，只比恒定稳定控制的 "
            f"{constant.macro_f1_mean:.4f} 略高；三个种子平均产生 "
            f"{transformer.predicted_boundaries_24h_mean:.0f} 个边界，而真值只有 8 个。",
            "",
            f"5. **标签重组没有改变方法排序。** 相对 v1，Raw Macro-F1 仅变化 "
            f"{old_new_f1.loc['raw_pca_kmeans', 'absolute_change']:+.4f}，PELT 变化 "
            f"{old_new_f1.loc['pelt_kmeans', 'absolute_change']:+.4f}，Transformer 变化 "
            f"{old_new_f1.loc['transformer_embedding_kmeans', 'absolute_change']:+.4f}。",
            "",
            "## 建议的下一步实验",
            "",
            "1. 为 Transformer 增加变化点、局部斜率或片段一致性目标，而不是继续只优化下一 TLE 预测。",
            "2. 对持续降低类使用按目标重采样或代价敏感的簇解释，重点压低误报并扩充独立降低目标。",
            "3. 单独调节 PELT penalty/min-size，并以 Macro-F1 与边界误报数联合选参，避免被稳定长段 IoU 误导。",
            "4. 重新收集一份阈值和模型均未查看的新目标集合后，再进行一次真正的冻结最终测试。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = args.results.resolve()
    raw = pd.read_csv(results / "metrics_long.csv")
    summary = aggregate_metrics(raw)
    per_class = aggregate_per_class(results)
    comparison = v1_comparison(summary, args.v1_summary.resolve())
    summary.to_csv(results / "summary_by_system.csv", index=False)
    per_class.to_csv(results / "per_class_summary.csv", index=False)
    comparison.to_csv(results / "change_vs_v1.csv", index=False)
    write_report(summary, per_class, comparison, results / "RESULTS.md")
    print(summary[summary["split"].eq("test")].to_string(index=False))
    print(f"Wrote {results / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
