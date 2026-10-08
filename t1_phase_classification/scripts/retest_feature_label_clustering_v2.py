#!/usr/bin/env python3
"""Retrospectively score frozen unsupervised models with feature-label v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from run_frozen_observable_final_test import (
    batched_transformer,
    choose_device,
    infer_pelt,
)
from satlife_observable.data import (
    load_dataset,
    normalize,
    validate_dataset_index_alignment,
)
from satlife_observable.metrics import (
    UNMAPPED_ID,
    apply_mapping,
    evaluate_predictions,
    fit_hungarian_mapping,
)
from satlife_observable.model import ObservableARTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "output/tle_observable_regime_feature_labels_v2"
DEFAULT_BASELINES = PROJECT_ROOT / "reports/observable_regime_demo/baselines"
DEFAULT_TRANSFORMERS = PROJECT_ROOT / "reports/observable_regime_demo/scale/fraction_1.00"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/observable_regime_feature_labels_v2/retest"
PRIMARY_LABEL_IDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--transformer-root", type=Path, default=DEFAULT_TRANSFORMERS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_reuse_inputs(dataset_dir: Path, data: dict[str, np.ndarray]) -> None:
    source_path = (
        PROJECT_ROOT
        / "output/tle_observable_regime_testset/tle_observable_regime_dataset.npz"
    )
    with np.load(source_path, allow_pickle=False) as source:
        for key in ["contexts_raw", "next_raw", "sample_ids", "split_codes", "norad_ids"]:
            if not np.array_equal(data[key], source[key]):
                raise ValueError(
                    f"v2 changed model input array {key}; frozen models cannot be reused"
                )
    schema = json.loads((dataset_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("label_version") != "feature_rules_v2":
        raise ValueError("Expected feature_rules_v2 dataset")


def save_evaluation(
    output_dir: Path,
    index: pd.DataFrame,
    mapped_labels: np.ndarray,
    id_to_label: dict[int, str],
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "SAMPLE_ID": index["SAMPLE_ID"].astype(str),
            "predicted_label_id": mapped_labels.astype(int),
        }
    )
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    rows: list[dict[str, object]] = []
    for split in ["validation", "test"]:
        split_frame = index[index["SPLIT"].eq(split)].copy()
        split_positions = split_frame.index.to_numpy()
        split_frame["PREDICTED_LABEL_ID"] = mapped_labels[split_positions]
        metrics, per_class, confusion, boundary_matches, segment_matches = evaluate_predictions(
            split_frame,
            "PREDICTED_LABEL_ID",
            PRIMARY_LABEL_IDS,
        )
        metrics.update(
            {
                "evaluation_split": split,
                "mapping_split": "train",
                "label_version": "feature_rules_v2",
                "retrospective_test": split == "test",
            }
        )
        split_dir = output_dir / f"evaluation_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        per_class["LABEL_NAME"] = per_class["LABEL_ID"].map(id_to_label)
        per_class.to_csv(split_dir / "per_class.csv", index=False)
        confusion.to_csv(split_dir / "confusion_matrix.csv")
        boundary_matches.to_csv(split_dir / "boundary_matches.csv", index=False)
        segment_matches.to_csv(split_dir / "segment_matches.csv", index=False)
        rows.append(
            {
                "split": split,
                "sample_count": metrics["sample_count"],
                "satellite_count": metrics["satellite_count"],
                "adjusted_rand_index": metrics["adjusted_rand_index"],
                "normalized_mutual_information": metrics["normalized_mutual_information"],
                "macro_f1": metrics["macro_f1"],
                "symmetric_segment_iou": metrics["segments"][
                    "symmetric_segment_iou_with_false_positive_penalty"
                ],
                "segment_macro_duration_iou": metrics["segments"]["segment_macro_duration_iou"],
                "boundary_f1_24h": metrics["boundary"]["24h"]["f1"],
                "true_boundaries_24h": metrics["boundary"]["24h"]["true_boundaries"],
                "predicted_boundaries_24h": metrics["boundary"]["24h"]["predicted_boundaries"],
            }
        )
    return rows


def map_from_train(
    index: pd.DataFrame, clusters: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    primary = index["IS_PRIMARY_LABEL"]
    if primary.dtype != bool:
        primary = primary.astype(str).str.lower().isin({"1", "true", "yes"})
    mapping_mask = index["SPLIT"].eq("train") & primary
    mapping = fit_hungarian_mapping(
        index.loc[mapping_mask, "LABEL_ID"].to_numpy(np.int64),
        clusters[mapping_mask.to_numpy()],
    )
    mapped = apply_mapping(clusters, mapping)
    if np.any(mapped == UNMAPPED_ID):
        raise RuntimeError(f"Unmapped clusters after train mapping: {mapping}")
    return mapped, mapping


def transformer_clusters(
    data: dict[str, np.ndarray],
    run: Path,
    device: torch.device,
    batch_size: int,
    feature_names: list[str],
) -> np.ndarray:
    normalizer = json.loads((run / "normalizer.json").read_text(encoding="utf-8"))
    contexts = normalize(data["contexts_raw"], normalizer)
    checkpoint = torch.load(run / "model.pt", map_location=device, weights_only=False)
    if list(checkpoint.get("feature_names", [])) != feature_names:
        raise ValueError(f"Transformer feature order mismatch: {run}")
    model = ObservableARTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    embeddings, _ = batched_transformer(model, contexts, device, batch_size)
    objects = joblib.load(run / "clusterer.joblib")
    return objects["clusterer"].predict(
        objects["pca"].transform(objects["scaler"].transform(embeddings))
    )


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    dataset_dir = args.dataset_dir.resolve()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_dir / "tle_observable_regime_dataset.npz")
    index = pd.read_csv(dataset_dir / "dataset_index.csv")
    validate_dataset_index_alignment(data, index)
    validate_reuse_inputs(dataset_dir, data)
    schema = json.loads((dataset_dir / "feature_schema.json").read_text(encoding="utf-8"))
    feature_names = list(schema["feature_names"])
    label_map = json.loads((dataset_dir / "label_map.json").read_text(encoding="utf-8"))
    id_to_label = {int(key): value for key, value in label_map["id_to_label"].items()}
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    metric_rows: list[dict[str, object]] = []

    constant = np.zeros(len(index), dtype=np.int64)
    for row in save_evaluation(output / "constant_stable", index, constant, id_to_label):
        metric_rows.append({"system": "constant_stable", "family": "control", "seed": -1, **row})

    for seed in args.seeds:
        baseline_run = args.baseline_root / f"seed_{seed}"
        transformer_run = args.transformer_root / f"seed_{seed}"
        required = [
            baseline_run / "normalizer.json",
            baseline_run / "raw_window_clusterer.joblib",
            baseline_run / "pelt_segment_clusterer.joblib",
            baseline_run / "run_summary.json",
            transformer_run / "normalizer.json",
            transformer_run / "model.pt",
            transformer_run / "clusterer.joblib",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing frozen model artifacts:\n" + "\n".join(missing))

        baseline_normalizer = json.loads((baseline_run / "normalizer.json").read_text(encoding="utf-8"))
        contexts = normalize(data["contexts_raw"], baseline_normalizer)
        targets = normalize(data["next_raw"], baseline_normalizer)

        raw_objects = joblib.load(baseline_run / "raw_window_clusterer.joblib")
        flattened = contexts.reshape(len(contexts), -1)
        raw_clusters = raw_objects["clusterer"].predict(
            raw_objects["pca"].transform(raw_objects["scaler"].transform(flattened))
        )
        raw_mapped, raw_mapping = map_from_train(index, raw_clusters)
        raw_dir = output / "raw_pca_kmeans" / f"seed_{seed}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "cluster_label_mapping.json").write_text(
            json.dumps(raw_mapping, indent=2), encoding="utf-8"
        )
        for row in save_evaluation(raw_dir, index, raw_mapped, id_to_label):
            metric_rows.append({"system": "raw_pca_kmeans", "family": "traditional", "seed": seed, **row})

        pelt_objects = joblib.load(baseline_run / "pelt_segment_clusterer.joblib")
        baseline_summary = json.loads((baseline_run / "run_summary.json").read_text(encoding="utf-8"))
        pelt_clusters = infer_pelt(
            targets,
            index.reset_index(drop=True),
            pelt_objects,
            baseline_summary["configuration"],
            feature_names,
        )
        pelt_mapped, pelt_mapping = map_from_train(index, pelt_clusters)
        pelt_dir = output / "pelt_kmeans" / f"seed_{seed}"
        pelt_dir.mkdir(parents=True, exist_ok=True)
        (pelt_dir / "cluster_label_mapping.json").write_text(
            json.dumps(pelt_mapping, indent=2), encoding="utf-8"
        )
        for row in save_evaluation(pelt_dir, index, pelt_mapped, id_to_label):
            metric_rows.append({"system": "pelt_kmeans", "family": "traditional", "seed": seed, **row})

        learned_clusters = transformer_clusters(
            data,
            transformer_run,
            device,
            args.eval_batch_size,
            feature_names,
        )
        learned_mapped, learned_mapping = map_from_train(index, learned_clusters)
        learned_dir = output / "transformer_embedding_kmeans" / f"seed_{seed}"
        learned_dir.mkdir(parents=True, exist_ok=True)
        (learned_dir / "cluster_label_mapping.json").write_text(
            json.dumps(learned_mapping, indent=2), encoding="utf-8"
        )
        for row in save_evaluation(learned_dir, index, learned_mapped, id_to_label):
            metric_rows.append(
                {
                    "system": "transformer_embedding_kmeans",
                    "family": "self_supervised",
                    "seed": seed,
                    **row,
                }
            )
        print(f"Completed retrospective inference for seed {seed}", flush=True)

    metrics = pd.DataFrame.from_records(metric_rows)
    metrics.to_csv(output / "metrics_long.csv", index=False)
    manifest = {
        "created_by": "scripts/retest_feature_label_clustering_v2.py",
        "label_version": "feature_rules_v2",
        "evaluation_kind": "retrospective_test",
        "not_a_new_unseen_final_test": True,
        "reuse_rationale": (
            "TLE inputs, self-supervised next-TLE target, split, and cluster count are unchanged; "
            "only the proxy labels and train-fitted cluster-to-state mapping changed."
        ),
        "dataset": str(dataset_dir),
        "dataset_sha256": sha256(dataset_dir / "tle_observable_regime_dataset.npz"),
        "index_sha256": sha256(dataset_dir / "dataset_index.csv"),
        "seeds": args.seeds,
        "device": str(device),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
