#!/usr/bin/env python3
"""Build a sample-aligned dataset using the feature-only fragment labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from plot_tle_observable_state_division import (
    MAX_GAP_HOURS,
    STATE_ORDER,
    build_catalog,
    load_points,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "output/tle_observable_regime_testset"
DEFAULT_OUTPUT = PROJECT_ROOT / "output/tle_observable_regime_feature_labels_v2"
STATE_TO_ID = {state: position for position, state in enumerate(STATE_ORDER)}
PRIMARY_STATES = {"STABLE_ORBIT", "SUSTAINED_RAISE", "SUSTAINED_DECAY"}
CASE_ONLY_STATES = {"OFFSET_PLATEAU", "SHORT_EXCURSION_RETURN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_state_map(
    catalog: pd.DataFrame, fragments: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    state_by_fragment = catalog.set_index("FRAGMENT_ID")[
        ["OBSERVABLE_STATE", "MATCHED_RULE"]
    ].to_dict("index")
    records: list[pd.DataFrame] = []
    for fragment_id, fragment in fragments.items():
        metadata = state_by_fragment[fragment_id]
        records.append(
            pd.DataFrame(
                {
                    "SAMPLE_ID": fragment["SAMPLE_ID"].astype(str).to_numpy(),
                    "FRAGMENT_ID": fragment_id,
                    "LABEL_NAME": metadata["OBSERVABLE_STATE"],
                    "MATCHED_RULE": metadata["MATCHED_RULE"],
                }
            )
        )
    result = pd.concat(records, ignore_index=True)
    if not result["SAMPLE_ID"].is_unique:
        raise RuntimeError("A sample was assigned to multiple fragments")
    return result


def recompute_segment_ids(index: pd.DataFrame) -> pd.Series:
    ordered = index.sort_values(["NORAD_CAT_ID", "TARGET_EPOCH_UTC"]).copy()
    times = pd.to_datetime(ordered["TARGET_EPOCH_UTC"], utc=True)
    gaps = ordered.groupby("NORAD_CAT_ID", sort=False)["TARGET_EPOCH_UTC"].transform(
        lambda values: pd.to_datetime(values, utc=True).diff().dt.total_seconds() / 3600.0
    )
    new_segment = (
        ordered["NORAD_CAT_ID"].ne(ordered["NORAD_CAT_ID"].shift())
        | ordered["LABEL_NAME"].ne(ordered["LABEL_NAME"].shift())
        | gaps.gt(MAX_GAP_HOURS)
    )
    local = new_segment.groupby(ordered["NORAD_CAT_ID"], sort=False).cumsum().astype(int)
    ids = ordered["NORAD_CAT_ID"].astype(str) + ":feature-v2:" + local.astype(str)
    if times.isna().any():
        raise RuntimeError("Invalid target timestamp while constructing segment IDs")
    return ids.reindex(index.index)


def write_dataset_card(index: pd.DataFrame, catalog: pd.DataFrame, path: Path) -> None:
    state_counts = index["LABEL_NAME"].value_counts()
    fragment_counts = catalog["OBSERVABLE_STATE"].value_counts()
    lines = [
        "# TLE 可观测状态特征标签 v2 数据集",
        "",
        "## 定位",
        "",
        "该包复用原始 TLE `5→1` 窗口和 NORAD 隔离划分，只替换为",
        "`STARLINK_MOTION_PHASE_TEST_CASE.md` 定义的互斥、数据特征驱动状态标签。",
        "旧语义标签仅作为审计列保留，不参与 v2 状态判定。",
        "",
        "## 状态分布",
        "",
        "| 状态 | 片段数 | 窗口数 | 用途 |",
        "|---|---:|---:|---|",
    ]
    for state in STATE_ORDER:
        purpose = "主定量" if state in PRIMARY_STATES else "案例" if state in CASE_ONLY_STATES else "拒判"
        lines.append(
            f"| `{state}` | {int(fragment_counts.get(state, 0)):,} | "
            f"{int(state_counts.get(state, 0)):,} | {purpose} |"
        )
    lines.extend(
        [
            "",
            "## 评测协议",
            "",
            "- 聚类数固定为 3，对应三个主定量状态。",
            "- 无监督聚类器只在训练 NORAD 上拟合；簇到状态的 Hungarian 映射也只使用训练标签。",
            "- 验证和测试目标均不参与聚类器拟合或簇映射。",
            "- 测试结果属于标签重组后的回顾性评测，不是新的未见最终测试。",
            "- v1 冻结结果及访问标记保持不变。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate(index: pd.DataFrame, data: dict[str, np.ndarray]) -> None:
    count = len(index)
    if any(len(value) != count for value in data.values()):
        raise RuntimeError("Output arrays do not share the index row count")
    if not np.array_equal(data["sample_ids"].astype(str), index["SAMPLE_ID"].astype(str)):
        raise RuntimeError("SAMPLE_ID order changed")
    if not np.array_equal(data["labels"], index["LABEL_ID"].to_numpy(np.int16)):
        raise RuntimeError("Label tensor and index differ")
    if not np.array_equal(data["primary_mask"], index["IS_PRIMARY_LABEL"].to_numpy(bool)):
        raise RuntimeError("Primary mask tensor and index differ")
    split_sets = {
        split: set(index.loc[index["SPLIT"].eq(split), "NORAD_CAT_ID"].astype(int))
        for split in ["train", "validation", "test"]
    }
    if any(
        split_sets[left] & split_sets[right]
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise RuntimeError("NORAD leakage across splits")
    if not np.isfinite(data["contexts_raw"]).all() or not np.isfinite(data["next_raw"]).all():
        raise RuntimeError("Non-finite model input")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    points = load_points(source)
    catalog, fragments = build_catalog(points)
    assignment = sample_state_map(catalog, fragments)
    source_index = pd.read_csv(source / "dataset_index.csv")
    rename = {
        "LABEL_ID": "V1_LABEL_ID",
        "LABEL_NAME": "V1_LABEL_NAME",
        "SEGMENT_ID": "V1_SEGMENT_ID",
        "IS_PRIMARY_LABEL": "V1_IS_PRIMARY_LABEL",
        "IS_CASE_ONLY_LABEL": "V1_IS_CASE_ONLY_LABEL",
        "AUDIT_PASS": "V1_AUDIT_PASS",
        "PROXY_KIND": "V1_PROXY_KIND",
    }
    index = source_index.rename(columns={key: value for key, value in rename.items() if key in source_index})
    index = index.merge(assignment, on="SAMPLE_ID", how="left", validate="one_to_one", sort=False)
    if index["LABEL_NAME"].isna().any():
        raise RuntimeError("Some source samples have no feature-label assignment")
    index["LABEL_ID"] = index["LABEL_NAME"].map(STATE_TO_ID).astype(np.int16)
    index["IS_PRIMARY_LABEL"] = index["LABEL_NAME"].isin(PRIMARY_STATES)
    index["IS_CASE_ONLY_LABEL"] = index["LABEL_NAME"].isin(CASE_ONLY_STATES)
    index["PROXY_KIND"] = "TLE_OBSERVABLE_FEATURE_RULE_V2"
    index["SEGMENT_ID"] = recompute_segment_ids(index)

    with np.load(source / "tle_observable_regime_dataset.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data["labels"] = index["LABEL_ID"].to_numpy(np.int16)
    data["primary_mask"] = index["IS_PRIMARY_LABEL"].to_numpy(bool)
    validate(index, data)
    np.savez_compressed(output / "tle_observable_regime_dataset.npz", **data)
    index.to_csv(output / "dataset_index.csv", index=False)
    index.to_csv(output / "test_index.csv", index=False)
    catalog.to_csv(output / "fragment_catalog.csv", index=False)

    for filename in ["calibration_norad_ids.json", "evaluation_normalizer.json"]:
        shutil.copy2(source / filename, output / filename)
    schema = json.loads((source / "feature_schema.json").read_text(encoding="utf-8"))
    schema.update(
        {
            "label_version": "feature_rules_v2",
            "label_standard": "STARLINK_MOTION_PHASE_TEST_CASE.md",
            "fragment_maximum_gap_hours": MAX_GAP_HOURS,
            "label_semantics": "mutually exclusive TLE-observable fragment shape",
        }
    )
    (output / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    label_map = {
        "label_to_id": STATE_TO_ID,
        "id_to_label": {str(value): key for key, value in STATE_TO_ID.items()},
        "primary_labels": sorted(PRIMARY_STATES, key=STATE_TO_ID.get),
        "case_only_labels": sorted(CASE_ONLY_STATES, key=STATE_TO_ID.get),
        "unknown_label": "UNKNOWN_TRANSITION",
        "proxy_kind": "TLE_OBSERVABLE_FEATURE_RULE_V2",
        "decision_order": list(STATE_ORDER[4:5])
        + list(STATE_ORDER[3:4])
        + list(STATE_ORDER[1:3])
        + list(STATE_ORDER[0:1])
        + ["UNKNOWN_TRANSITION"],
        "standard": "STARLINK_MOTION_PHASE_TEST_CASE.md",
    }
    (output / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_dataset_card(index, catalog, output / "DATASET_CARD.md")
    summary = {
        "label_version": "feature_rules_v2",
        "sample_count": len(index),
        "fragment_count": len(catalog),
        "satellite_count": int(index["NORAD_CAT_ID"].nunique()),
        "label_sample_counts": index["LABEL_NAME"].value_counts().to_dict(),
        "label_fragment_counts": catalog["OBSERVABLE_STATE"].value_counts().to_dict(),
        "split_sample_counts": index["SPLIT"].value_counts().sort_index().to_dict(),
        "split_satellite_counts": {
            split: int(index.loc[index["SPLIT"].eq(split), "NORAD_CAT_ID"].nunique())
            for split in ["train", "validation", "test"]
        },
        "source_dataset": str(source),
        "source_dataset_sha256": sha256(source / "tle_observable_regime_dataset.npz"),
        "retrospective_test_only": True,
        "reason": "feature-label rules were organized after the original frozen test was consumed",
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "created_by": "scripts/build_tle_observable_feature_labels_v2.py",
                "source": str(source),
                "files": manifest_files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
