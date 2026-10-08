#!/usr/bin/env python3
"""Run the frozen test exactly once after all validation decisions are locked."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import ruptures as rpt
import torch

from satlife_observable.data import denormalize, load_dataset, normalize, validate_dataset_index_alignment
from satlife_observable.metrics import UNMAPPED_ID, apply_mapping
from satlife_observable.model import ObservableARTransformer
from satlife_observable.final_test import (
    FINAL_TEST_MARKER,
    FINAL_TEST_TOKEN_ENV,
    require_active_final_test_access,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCESS_MARKER = FINAL_TEST_MARKER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument(
        "--transformer-run",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Frozen transformer run; repeat for each selected seed.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "output/tle_observable_regime_testset/tle_observable_regime_dataset.npz",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "output/tle_observable_regime_testset/dataset_index.csv",
    )
    parser.add_argument(
        "--calibration-norad-ids",
        type=Path,
        default=PROJECT_ROOT / "output/tle_observable_regime_testset/calibration_norad_ids.json",
    )
    parser.add_argument(
        "--evaluation-normalizer",
        type=Path,
        default=PROJECT_ROOT / "output/tle_observable_regime_testset/evaluation_normalizer.json",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def reserve_final_test(args: argparse.Namespace) -> str:
    ACCESS_MARKER.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    payload = {
        "status": "STARTED",
        "started_at_utc": utc_now(),
        "out": str(args.out),
        "baseline_run": str(args.baseline_run),
        "transformer_runs": args.transformer_run,
        "access_token": token,
        "policy": "Any existing marker blocks another final-test invocation, including after failure.",
    }
    descriptor = os.open(ACCESS_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, json.dumps(payload, indent=2).encode("utf-8"))
    finally:
        os.close(descriptor)
    return token


def preflight(args: argparse.Namespace) -> None:
    transformer_runs = parse_transformer_runs(args.transformer_run)
    if ACCESS_MARKER.exists():
        raise FileExistsError(
            f"Frozen test has already been reserved or consumed: {ACCESS_MARKER}"
        )
    required_baseline = [
        "normalizer.json",
        "raw_window_clusterer.joblib",
        "pelt_segment_clusterer.joblib",
        "ridge_delta.joblib",
        "run_summary.json",
        "evaluation_raw_window/cluster_label_mapping.json",
        "evaluation_pelt/cluster_label_mapping.json",
    ]
    required_transformer = [
        "normalizer.json",
        "model.pt",
        "clusterer.joblib",
        "evaluation_validation/cluster_label_mapping.json",
        "run_manifest.json",
        "run_summary.json",
    ]
    missing = [str(args.baseline_run / relative) for relative in required_baseline if not (args.baseline_run / relative).is_file()]
    for run in transformer_runs.values():
        missing.extend(str(run / relative) for relative in required_transformer if not (run / relative).is_file())
    if not args.dataset.is_file():
        missing.append(str(args.dataset))
    if not args.index.is_file():
        missing.append(str(args.index))
    if not args.calibration_norad_ids.is_file():
        missing.append(str(args.calibration_norad_ids))
    if not args.evaluation_normalizer.is_file():
        missing.append(str(args.evaluation_normalizer))
    if missing:
        raise FileNotFoundError("Final-test preflight is missing required frozen artifacts:\n" + "\n".join(missing))
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty final-test directory: {args.out}")
    dataset_hash = sha256(args.dataset)
    index_hash = sha256(args.index)
    calibration_hash = sha256(args.calibration_norad_ids)
    evaluation_normalizer_hash = sha256(args.evaluation_normalizer)
    feature_schema_path = args.index.parent / "feature_schema.json"
    if not feature_schema_path.is_file():
        raise FileNotFoundError(f"Missing feature schema: {feature_schema_path}")
    feature_schema_hash = sha256(feature_schema_path)
    baseline_summary = json.loads((args.baseline_run / "run_summary.json").read_text(encoding="utf-8"))
    expected_baseline_hashes = {
        "dataset_sha256": dataset_hash,
        "index_sha256": index_hash,
        "feature_schema_sha256": feature_schema_hash,
        "calibration_ids_sha256": calibration_hash,
        "evaluation_normalizer_sha256": evaluation_normalizer_hash,
    }
    for key, expected in expected_baseline_hashes.items():
        if baseline_summary.get(key) != expected:
            raise ValueError(f"Baseline {key} does not match frozen input")
    if int(baseline_summary["configuration"]["n_clusters"]) != 3:
        raise ValueError("Frozen baseline must use the three quantitative primary clusters")

    schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    feature_names = list(schema["feature_names"])
    allowed_configuration_differences = {
        "fraction", "seed", "out", "dataset", "index",
        "calibration_norad_ids", "evaluation_normalizer",
    }
    reference_configuration: dict[str, object] | None = None
    fraction_seeds: dict[float, set[int]] = {0.10: set(), 1.00: set()}
    fraction_run_counts: dict[float, int] = {0.10: 0, 1.00: 0}
    resolved_runs: set[Path] = set()
    for name, run in transformer_runs.items():
        resolved = run.resolve()
        if resolved in resolved_runs:
            raise ValueError(f"Transformer run path is duplicated: {run}")
        resolved_runs.add(resolved)
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
        expected_hashes = {
            "dataset_sha256": dataset_hash,
            "index_sha256": index_hash,
            "feature_schema_sha256": feature_schema_hash,
            "calibration_ids_sha256": calibration_hash,
            "evaluation_normalizer_sha256": evaluation_normalizer_hash,
        }
        for key, expected in expected_hashes.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Transformer {name} {key} does not match frozen input")
        configuration = manifest["configuration"]
        if reference_configuration is None:
            reference_configuration = configuration
        else:
            all_keys = set(configuration) | set(reference_configuration)
            unequal = sorted(
                key for key in all_keys - allowed_configuration_differences
                if configuration.get(key) != reference_configuration.get(key)
            )
            if unequal:
                raise ValueError(f"Transformer {name} changes controlled settings: {unequal}")
        fraction = float(summary["fraction"])
        seed = int(summary["seed"])
        matched_fraction = next((value for value in fraction_seeds if np.isclose(fraction, value)), None)
        if matched_fraction is None:
            raise ValueError(f"Final test only accepts frozen 10% and 100% runs, got {fraction}")
        fraction_seeds[matched_fraction].add(seed)
        fraction_run_counts[matched_fraction] += 1
        checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        if int(checkpoint["model_config"]["input_dim"]) != int(schema["feature_count"]):
            raise ValueError(f"Transformer {name} input dimension does not match schema")
        if list(checkpoint.get("feature_names", [])) != feature_names:
            raise ValueError(f"Transformer {name} feature order does not match schema")
        if int(configuration["n_clusters"]) != 3:
            raise ValueError(f"Transformer {name} must use three clusters")
    expected_final_seeds = {42, 43, 44}
    if any(
        fraction_seeds[fraction] != expected_final_seeds
        or fraction_run_counts[fraction] != len(expected_final_seeds)
        for fraction in fraction_seeds
    ):
        raise ValueError(
            "Final test requires exactly seeds 42/43/44 once each for both frozen 10% and 100% transformer runs"
        )


def parse_transformer_runs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH for --transformer-run, got {value!r}")
        name, path = value.split("=", 1)
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name)
            or name in {"constant_stable", "raw_window", "pelt", "persistence", "ridge_delta"}
            or name in result
        ):
            raise ValueError(f"Invalid or duplicate transformer name: {name!r}")
        result[name] = Path(path)
    if not result:
        raise ValueError("At least one --transformer-run NAME=PATH is required")
    return result


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if requested == "cuda" or torch.cuda.is_available() else "cpu")


def batched_transformer(
    model: ObservableARTransformer,
    contexts: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings, deltas = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            batch = torch.from_numpy(contexts[start : start + batch_size]).to(device).float()
            embedding, delta = model(batch)
            embeddings.append(embedding.cpu().numpy())
            deltas.append(delta.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(deltas)


def contiguous_chunks(frame: pd.DataFrame, maximum_gap_hours: float) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    for _, rows in frame.groupby("NORAD_CAT_ID", sort=True).groups.items():
        ordered = frame.loc[rows].sort_values("TARGET_EPOCH_UTC")
        positions = ordered.index.to_numpy()
        times = pd.to_datetime(ordered["TARGET_EPOCH_UTC"], utc=True)
        gaps = times.diff().dt.total_seconds().div(3600.0).fillna(0.0).to_numpy()
        start = 0
        for position in range(1, len(positions) + 1):
            if position == len(positions) or gaps[position] > maximum_gap_hours:
                chunks.append(positions[start:position])
                start = position
    return chunks


def infer_pelt(
    targets: np.ndarray,
    frame: pd.DataFrame,
    objects: dict[str, object],
    configuration: dict[str, object],
    feature_names: list[str],
) -> np.ndarray:
    selected = [feature_names.index(name) for name in objects["feature_names"]]
    signal_values = targets[:, selected]
    row_segment = np.full(len(frame), -1, dtype=np.int64)
    summaries: list[np.ndarray] = []
    segment_id = 0
    maximum_gap = float(configuration["maximum_gap_hours"])
    minimum_size = int(configuration["pelt_min_size"])
    for positions in contiguous_chunks(frame, maximum_gap):
        signal = signal_values[positions]
        if len(signal) < max(4, minimum_size * 2):
            breakpoints = [len(signal)]
        else:
            breakpoints = rpt.Pelt(
                model=str(configuration["pelt_model"]),
                min_size=minimum_size,
                jump=int(configuration["pelt_jump"]),
            ).fit(signal).predict(pen=float(configuration["pelt_penalty"]))
        start = 0
        for end in breakpoints:
            block = signal[start:end]
            selected_rows = positions[start:end]
            if len(block) == 0:
                continue
            row_segment[selected_rows] = segment_id
            summaries.append(
                np.concatenate([np.mean(block, axis=0), np.std(block, axis=0), block[-1] - block[0]])
            )
            segment_id += 1
            start = end
    if np.any(row_segment < 0):
        raise RuntimeError("PELT final inference failed to assign every row")
    summary_values = np.stack(summaries)
    segment_clusters = objects["clusterer"].predict(objects["scaler"].transform(summary_values))
    return segment_clusters[row_segment]


def normalized_prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    smooth_l1 = np.where(absolute < 1.0, 0.5 * absolute**2, absolute - 0.5)
    return {
        "normalized_mae": float(absolute.mean()),
        "normalized_rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_smooth_l1": float(smooth_l1.mean()),
    }


def save_prediction_errors(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    target: np.ndarray,
    path: Path,
) -> None:
    error = prediction - target
    pd.DataFrame(
        {
            "SAMPLE_ID": frame["SAMPLE_ID"].astype(str),
            "NORAD_CAT_ID": frame["NORAD_CAT_ID"].astype(int),
            "normalized_mae": np.abs(error).mean(axis=1),
            "normalized_rmse": np.sqrt(np.mean(error**2, axis=1)),
        }
    ).to_csv(path, index=False)


def evaluate_system(
    name: str,
    mapped_labels: np.ndarray,
    frame: pd.DataFrame,
    args: argparse.Namespace,
    normalizer: Path,
) -> dict[str, object]:
    if np.any(mapped_labels == UNMAPPED_ID):
        raise RuntimeError(f"System {name} contains unmapped clusters")
    system_dir = args.out / name
    system_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = system_dir / "train_test_predictions.csv"
    pd.DataFrame(
        {"SAMPLE_ID": frame["SAMPLE_ID"].astype(str), "predicted_label_id": mapped_labels.astype(int)}
    ).to_csv(prediction_path, index=False)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluate_observable_regimes.py"),
            str(prediction_path),
            "--index",
            str(args.index),
            "--split",
            "test",
            "--mapping-split",
            "train",
            "--calibration-norad-ids",
            str(args.calibration_norad_ids),
            "--out",
            str(system_dir / "evaluation_test"),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/analyze_observable_predictions.py"),
            str(prediction_path),
            "--dataset",
            str(args.dataset),
            "--index",
            str(args.index),
            "--normalizer",
            str(normalizer),
            "--split",
            "test",
            "--out",
            str(system_dir / "intrinsic_test"),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
    )
    return json.loads((system_dir / "evaluation_test/metrics.json").read_text(encoding="utf-8"))


def execute(args: argparse.Namespace) -> dict[str, object]:
    require_active_final_test_access(args.out)
    transformer_runs = parse_transformer_runs(args.transformer_run)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty final-test directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.dataset)
    index = pd.read_csv(args.index)
    validate_dataset_index_alignment(data, index)
    selected_mask = data["split_codes"] != 1
    test_mask = data["split_codes"] == 2
    selected_index = index.loc[selected_mask].reset_index(drop=True)
    local_test_mask = test_mask[selected_mask]
    test_index = selected_index.loc[local_test_mask].reset_index(drop=True)
    schema = json.loads((args.index.parent / "feature_schema.json").read_text(encoding="utf-8"))
    feature_names = list(schema["feature_names"])
    evaluation_normalizer = json.loads(args.evaluation_normalizer.read_text(encoding="utf-8"))
    evaluation_targets = normalize(data["next_raw"][test_mask], evaluation_normalizer)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    baseline_normalizer_path = args.baseline_run / "normalizer.json"
    baseline_normalizer = json.loads(baseline_normalizer_path.read_text(encoding="utf-8"))
    baseline_contexts = normalize(data["contexts_raw"], baseline_normalizer)[selected_mask]
    baseline_targets = normalize(data["next_raw"], baseline_normalizer)[selected_mask]
    system_metrics: dict[str, object] = {}
    prediction_metrics: dict[str, object] = {}

    system_metrics["constant_stable"] = evaluate_system(
        "constant_stable",
        np.zeros(len(selected_index), dtype=np.int64),
        selected_index,
        args,
        args.evaluation_normalizer,
    )

    raw_objects = joblib.load(args.baseline_run / "raw_window_clusterer.joblib")
    flattened = baseline_contexts.reshape(len(baseline_contexts), -1)
    raw_clusters = raw_objects["clusterer"].predict(
        raw_objects["pca"].transform(raw_objects["scaler"].transform(flattened))
    )
    raw_mapping = json.loads(
        (args.baseline_run / "evaluation_raw_window/cluster_label_mapping.json").read_text(encoding="utf-8")
    )
    system_metrics["raw_window"] = evaluate_system(
        "raw_window",
        apply_mapping(raw_clusters, raw_mapping),
        selected_index,
        args,
        args.evaluation_normalizer,
    )

    pelt_objects = joblib.load(args.baseline_run / "pelt_segment_clusterer.joblib")
    baseline_summary = json.loads((args.baseline_run / "run_summary.json").read_text(encoding="utf-8"))
    pelt_clusters = infer_pelt(
        baseline_targets,
        selected_index,
        pelt_objects,
        baseline_summary["configuration"],
        feature_names,
    )
    pelt_mapping = json.loads(
        (args.baseline_run / "evaluation_pelt/cluster_label_mapping.json").read_text(encoding="utf-8")
    )
    system_metrics["pelt"] = evaluate_system(
        "pelt",
        apply_mapping(pelt_clusters, pelt_mapping),
        selected_index,
        args,
        args.evaluation_normalizer,
    )

    persistence = baseline_contexts[local_test_mask, -1]
    persistence_evaluation = normalize(
        denormalize(persistence, baseline_normalizer), evaluation_normalizer
    )
    prediction_metrics["persistence"] = normalized_prediction_metrics(
        persistence_evaluation, evaluation_targets
    )
    save_prediction_errors(
        test_index,
        persistence_evaluation,
        evaluation_targets,
        args.out / "persistence_test_prediction_errors.csv",
    )
    ridge = joblib.load(args.baseline_run / "ridge_delta.joblib")
    ridge_prediction = persistence + ridge.predict(flattened[local_test_mask])
    ridge_evaluation = normalize(
        denormalize(ridge_prediction, baseline_normalizer), evaluation_normalizer
    )
    prediction_metrics["ridge_delta"] = normalized_prediction_metrics(
        ridge_evaluation, evaluation_targets
    )
    save_prediction_errors(
        test_index,
        ridge_evaluation,
        evaluation_targets,
        args.out / "ridge_test_prediction_errors.csv",
    )

    for name, run in transformer_runs.items():
        normalizer_path = run / "normalizer.json"
        normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
        contexts = normalize(data["contexts_raw"], normalizer)[selected_mask]
        targets = normalize(data["next_raw"], normalizer)[selected_mask]
        checkpoint = torch.load(run / "model.pt", map_location=device, weights_only=False)
        model = ObservableARTransformer(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        embeddings, deltas = batched_transformer(model, contexts, device, args.eval_batch_size)
        objects = joblib.load(run / "clusterer.joblib")
        clusters = objects["clusterer"].predict(
            objects["pca"].transform(objects["scaler"].transform(embeddings))
        )
        mapping = json.loads(
            (run / "evaluation_validation/cluster_label_mapping.json").read_text(encoding="utf-8")
        )
        system_metrics[name] = evaluate_system(
            name,
            apply_mapping(clusters, mapping),
            selected_index,
            args,
            args.evaluation_normalizer,
        )
        next_prediction = contexts[:, -1] + deltas
        evaluation_prediction = normalize(
            denormalize(next_prediction[local_test_mask], normalizer), evaluation_normalizer
        )
        prediction_metrics[name] = normalized_prediction_metrics(
            evaluation_prediction, evaluation_targets
        )
        save_prediction_errors(
            test_index,
            evaluation_prediction,
            evaluation_targets,
            args.out / f"{name}_test_prediction_errors.csv",
        )

    summary = {
        "evaluation_split": "test",
        "test_access_count": 1,
        "evaluation_normalizer": str(args.evaluation_normalizer),
        "scope": "TLE-observable proxy regimes; not operational intent or lifecycle ground truth",
        "baseline_run": str(args.baseline_run),
        "transformer_runs": {name: str(path) for name, path in transformer_runs.items()},
        "segmentation": system_metrics,
        "prediction": prediction_metrics,
    }
    (args.out / "final_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    # Configuration/path mistakes are rejected before the one-shot access is
    # reserved.  Once reservation succeeds, any failure remains recorded and
    # cannot be silently retried.
    preflight(args)
    token = reserve_final_test(args)
    os.environ[FINAL_TEST_TOKEN_ENV] = token
    try:
        summary = execute(args)
    except Exception as error:
        ACCESS_MARKER.write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "finished_at_utc": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "policy": "Final-test rerun remains blocked; inspect artifacts and obtain an explicit protocol decision.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    ACCESS_MARKER.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "finished_at_utc": utc_now(),
                "out": str(args.out),
                "systems": list(summary["segmentation"]),
                "test_access_count": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
