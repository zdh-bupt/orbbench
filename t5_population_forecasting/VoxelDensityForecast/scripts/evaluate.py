from __future__ import annotations

"""Evaluate learned forecasts without retaining every shard in CPU memory."""

import argparse
import gc
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starlink_forecast.config import load_config, resolve_project_path
from starlink_forecast.evaluation import (
    binomial_interval_coverage,
    evaluate_predictions,
    hotspot_ece,
    joint_tv,
    neighbor_tolerant_hotspot_f1,
    normalize_joint,
    regional_relative_mae,
)
from starlink_forecast.training import (
    ShardSelection,
    checkpoint_run_config,
    dataset_for_selection,
    model_from_config,
    serialize_ranges,
    shard_manifest,
    shard_selections_for_range,
    split_ranges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the learned model against physical and persistence baselines")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", help="Defaults to <artifact_dir>/model-best.pt")
    parser.add_argument("--split", choices=("validation", "calibration", "test"), default="test")
    parser.add_argument(
        "--max-shards",
        type=int,
        help="Debug only: cap selected shards after full chronological splitting",
    )
    parser.add_argument("--max-batches", type=int, help="Debug only: cap total evaluation batches")
    parser.add_argument("--max-calibration-batches", type=int, help="Debug only: cap total calibration batches")
    parser.add_argument("--batch-size", type=int, help="Runtime override; does not alter the shard digest")
    parser.add_argument("--num-workers", type=int, help="Runtime override; start with zero for large shards")
    parser.add_argument("--progress", action="store_true", help="Emit per-shard and periodic batch progress")
    parser.add_argument("--progress-every", type=int, default=50, help="Batch interval for --progress")
    parser.add_argument(
        "--artifact-dir",
        help="Directory for deployment_decision.json; defaults to the checkpoint's producing directory",
    )
    parser.add_argument(
        "--output",
        help="Evaluation JSON path; defaults to reports/evaluation.json for formal checkpoints",
    )
    return parser.parse_args()


def _progress(enabled: bool, event: str, **values: object) -> None:
    if enabled:
        print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def _loader_for_selection(
    selection: ShardSelection,
    config: dict[str, Any],
    time_range: Any,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, Any]:
    dataset = dataset_for_selection(selection, config, time_range, return_tensors=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    return loader, dataset


def _baseline_metrics(
    voxel_probability: np.ndarray,
    outside_probability: np.ndarray,
    total: int,
    target_count: np.ndarray,
    target_outside: np.ndarray,
    target_hard_count: np.ndarray,
) -> dict[str, float]:
    expected = voxel_probability * total
    outside_expected = outside_probability * total
    regional_error = (
        regional_relative_mae(expected, target_count)
        + float(np.abs(outside_expected - target_outside).sum() / max(target_count.sum() + target_outside.sum(), 1.0))
    )
    return {
        "tv": joint_tv(voxel_probability, outside_probability, target_count, target_outside),
        "regional_relative_mae": regional_error,
        "neighbor_hotspot_f1": neighbor_tolerant_hotspot_f1(expected, target_hard_count),
        "conservation_error": abs(expected.sum() + outside_expected.sum() - total) / max(total, 1),
    }


def fit_temperatures(
    model: torch.nn.Module,
    selections: list[ShardSelection],
    config: dict[str, Any],
    time_range: Any,
    device: torch.device,
    *,
    lead_count: int,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    progress: bool,
    progress_every: int,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Fit calibration temperatures by streaming one shard at a time."""

    candidates = np.geomspace(0.5, 2.0, 31).astype(np.float32)
    losses = np.zeros((lead_count, len(candidates)), dtype=np.float64)
    expected_samples = sum(selection.origin_count for selection in selections)
    batches = 0
    samples = 0
    shards = 0
    stopped = False
    model.eval()
    with torch.no_grad():
        for shard_index, selection in enumerate(selections, start=1):
            if max_batches is not None and batches >= max_batches:
                stopped = True
                break
            _progress(
                progress,
                "calibration_shard_start",
                shard_index=shard_index,
                shard_count=len(selections),
                shard=selection.path.name,
                shard_origins=selection.origin_count,
            )
            loader, dataset = _loader_for_selection(
                selection,
                config,
                time_range,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
            try:
                for batch in loader:
                    if max_batches is not None and batches >= max_batches:
                        stopped = True
                        break
                    device_batch = {
                        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    output = model(
                        device_batch["features"],
                        device_batch["physics_prior"],
                        device_batch["physics_prior_outside"],
                        device_batch["future_total"],
                    )
                    probability = torch.cat(
                        (
                            output.voxel_probability.flatten(start_dim=2),
                            output.outside_probability,
                        ),
                        dim=2,
                    ).clamp_min(1.0e-12)
                    target = torch.cat(
                        (
                            device_batch["target_counts"].flatten(start_dim=2),
                            device_batch["target_outside"],
                        ),
                        dim=2,
                    )
                    total = device_batch["future_total"].clamp_min(1.0)
                    log_probability = torch.log(probability)
                    for candidate_index, temperature in enumerate(candidates):
                        calibrated_log = torch.log_softmax(log_probability / float(temperature), dim=2)
                        nll = -(target * calibrated_log).sum(dim=2) / total
                        losses[:, candidate_index] += nll.sum(dim=0).cpu().numpy()
                    batch_samples = int(probability.shape[0])
                    batches += 1
                    samples += batch_samples
                    if progress and batches % progress_every == 0:
                        _progress(True, "calibration_batch", batches=batches, samples=samples, shard=selection.path.name)
                    if max_batches is not None and batches >= max_batches:
                        stopped = True
                        break
                shards += 1
            finally:
                del loader, dataset
                gc.collect()
            _progress(progress, "calibration_shard_done", shard_index=shard_index, batches=batches, samples=samples)
            if stopped:
                break
    if not samples:
        raise RuntimeError("Calibration stream produced no batches")
    metadata = {
        "batches": batches,
        "samples": samples,
        "expected_samples": expected_samples,
        "shards": shards,
        "complete": not stopped and samples == expected_samples,
    }
    return candidates[np.argmin(losses / samples, axis=1)], metadata


def apply_temperatures(
    voxel_probability: np.ndarray,
    outside_probability: np.ndarray,
    temperatures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    calibrated_voxel = np.empty_like(voxel_probability)
    calibrated_outside = np.empty_like(outside_probability)
    for sample_index in range(voxel_probability.shape[0]):
        for lead_index, temperature in enumerate(temperatures):
            categories = np.concatenate(
                (
                    voxel_probability[sample_index, lead_index].ravel(),
                    outside_probability[sample_index, lead_index],
                )
            )
            logits = np.log(np.maximum(categories, 1.0e-12)) / float(temperature)
            weights = np.exp(logits - logits.max())
            probability = weights / weights.sum()
            calibrated_voxel[sample_index, lead_index] = probability[:-2].reshape(voxel_probability.shape[-3:])
            calibrated_outside[sample_index, lead_index] = probability[-2:]
    return calibrated_voxel, calibrated_outside


def evaluate_stream(
    model: torch.nn.Module,
    selections: list[ShardSelection],
    config: dict[str, Any],
    time_range: Any,
    device: torch.device,
    horizons: tuple[int, ...],
    temperatures: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    progress: bool,
    progress_every: int,
) -> tuple[list[dict[str, object]], dict[str, int | bool]]:
    """Compute evaluation records while retaining only the active shard arrays."""

    records: list[dict[str, object]] = []
    expected_samples = sum(selection.origin_count for selection in selections)
    batches = 0
    samples = 0
    shards = 0
    stopped = False
    model.eval()
    with torch.no_grad():
        for shard_index, selection in enumerate(selections, start=1):
            if max_batches is not None and batches >= max_batches:
                stopped = True
                break
            _progress(
                progress,
                "evaluation_shard_start",
                shard_index=shard_index,
                shard_count=len(selections),
                shard=selection.path.name,
                shard_origins=selection.origin_count,
            )
            loader, dataset = _loader_for_selection(
                selection,
                config,
                time_range,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
            try:
                for batch in loader:
                    if max_batches is not None and batches >= max_batches:
                        stopped = True
                        break
                    device_batch = {
                        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    output = model(
                        device_batch["features"],
                        device_batch["physics_prior"],
                        device_batch["physics_prior_outside"],
                        device_batch["future_total"],
                    )
                    learned_voxel = output.voxel_probability.cpu().numpy()
                    learned_outside = output.outside_probability.cpu().numpy()
                    learned_voxel, learned_outside = apply_temperatures(
                        learned_voxel, learned_outside, temperatures
                    )
                    prior_voxel = batch["physics_prior"].numpy()
                    prior_outside = batch["physics_prior_outside"].numpy()
                    target = batch["target_counts"].numpy()
                    target_outside = batch["target_outside"].numpy()
                    target_hard = batch["target_hard_counts"].numpy()
                    target_hard_outside = batch["target_hard_outside"].numpy()
                    totals = batch["future_total"].numpy()
                    features = batch["features"].numpy()
                    origin_times = batch["origin_time_ns"].numpy()
                    for sample_index in range(len(origin_times)):
                        persistence_voxel = features[sample_index, -1, 0]
                        persistence_outside = np.asarray(
                            [
                                features[sample_index, -1, 10, 0, 0, 0],
                                features[sample_index, -1, 11, 0, 0, 0],
                            ]
                        )
                        persistence_voxel, persistence_outside = normalize_joint(
                            persistence_voxel, persistence_outside
                        )
                        for lead_index, horizon in enumerate(horizons):
                            total = int(round(float(totals[sample_index, lead_index])))
                            physics_voxel, physics_outside = normalize_joint(
                                prior_voxel[sample_index, lead_index], prior_outside[sample_index, lead_index]
                            )
                            learned = _baseline_metrics(
                                learned_voxel[sample_index, lead_index],
                                learned_outside[sample_index, lead_index],
                                total,
                                target[sample_index, lead_index],
                                target_outside[sample_index, lead_index],
                                target_hard[sample_index, lead_index],
                            )
                            learned["interval_90_coverage"] = binomial_interval_coverage(
                                learned_voxel[sample_index, lead_index],
                                learned_outside[sample_index, lead_index],
                                total,
                                target_hard[sample_index, lead_index],
                                target_hard_outside[sample_index, lead_index],
                            )
                            learned["hotspot_ece"] = hotspot_ece(
                                learned_voxel[sample_index, lead_index],
                                total,
                                target_hard[sample_index, lead_index],
                            )
                            records.append(
                                {
                                    "origin_time_ns": int(origin_times[sample_index]),
                                    "horizon": horizon,
                                    "learned": learned,
                                    "physics": _baseline_metrics(
                                        physics_voxel,
                                        physics_outside,
                                        total,
                                        target[sample_index, lead_index],
                                        target_outside[sample_index, lead_index],
                                        target_hard[sample_index, lead_index],
                                    ),
                                    "persistence": _baseline_metrics(
                                        persistence_voxel,
                                        persistence_outside,
                                        total,
                                        target[sample_index, lead_index],
                                        target_outside[sample_index, lead_index],
                                        target_hard[sample_index, lead_index],
                                    ),
                                }
                            )
                    batches += 1
                    samples += int(len(origin_times))
                    if progress and batches % progress_every == 0:
                        _progress(True, "evaluation_batch", batches=batches, samples=samples, shard=selection.path.name)
                    if max_batches is not None and batches >= max_batches:
                        stopped = True
                        break
                shards += 1
            finally:
                del loader, dataset
                gc.collect()
            _progress(progress, "evaluation_shard_done", shard_index=shard_index, batches=batches, samples=samples)
            if stopped:
                break
    if not records:
        raise RuntimeError("Evaluation stream produced no batches")
    return records, {
        "batches": batches,
        "samples": samples,
        "expected_samples": expected_samples,
        "shards": shards,
        "complete": not stopped and samples == expected_samples,
    }


def main() -> int:
    args = parse_args()
    for name, value in (
        ("--max-shards", args.max_shards),
        ("--max-batches", args.max_batches),
        ("--max-calibration-batches", args.max_calibration_batches),
        ("--batch-size", args.batch_size),
        ("--progress-every", args.progress_every),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{name} must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise SystemExit("--num-workers must be zero or positive")

    config = load_config(args.config)
    requested_artifact_dir = resolve_project_path(
        config,
        args.artifact_dir if args.artifact_dir is not None else config["paths"]["artifact_dir"],
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else requested_artifact_dir / "model-best.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_project_path(config, str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        run_config = checkpoint_run_config(checkpoint, config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    configured_artifact_dir = resolve_project_path(config, config["paths"]["artifact_dir"])
    recorded_artifact_dir = checkpoint.get("artifact_dir")
    if args.artifact_dir is not None:
        artifact_dir = resolve_project_path(config, args.artifact_dir)
    elif isinstance(recorded_artifact_dir, str) and recorded_artifact_dir:
        artifact_dir = Path(recorded_artifact_dir)
    else:
        artifact_dir = configured_artifact_dir

    manifest, all_shards = shard_manifest(config)
    ranges = split_ranges(config, all_shards)
    selections = shard_selections_for_range(all_shards, ranges[args.split])
    calibration_selections = (
        shard_selections_for_range(all_shards, ranges["calibration"]) if args.split == "test" else []
    )
    if args.max_shards is not None:
        selections = selections[: args.max_shards]
        calibration_selections = calibration_selections[: args.max_shards]
    if not selections:
        raise SystemExit(f"No samples are available in the {args.split} range")
    if args.split == "test" and not calibration_selections:
        raise SystemExit("No samples are available in the calibration range")

    batch_size = int(args.batch_size if args.batch_size is not None else config["training"]["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else config["training"]["num_workers"])
    if num_workers:
        print("warning: worker processes can multiply resident shard memory; start with --num-workers 0", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_config(run_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    horizons = tuple(int(value) for value in config["forecast"]["horizons"])
    _progress(
        args.progress,
        "runtime",
        device=str(device),
        batch_size=batch_size,
        num_workers=num_workers,
        evaluation_shards=len(selections),
        evaluation_origins=sum(selection.origin_count for selection in selections),
        calibration_shards=len(calibration_selections),
        calibration_origins=sum(selection.origin_count for selection in calibration_selections),
    )

    calibration_metadata: dict[str, int | bool]
    if args.split == "test":
        temperatures, calibration_metadata = fit_temperatures(
            model,
            calibration_selections,
            config,
            ranges["calibration"],
            device,
            lead_count=len(horizons),
            batch_size=batch_size,
            num_workers=num_workers,
            max_batches=args.max_calibration_batches,
            progress=args.progress,
            progress_every=args.progress_every,
        )
    else:
        temperatures = np.ones(len(horizons), dtype=np.float32)
        calibration_metadata = {"batches": 0, "samples": 0, "expected_samples": 0, "shards": 0, "complete": True}

    records, evaluation_metadata = evaluate_stream(
        model,
        selections,
        config,
        ranges[args.split],
        device,
        horizons,
        temperatures,
        batch_size=batch_size,
        num_workers=num_workers,
        max_batches=args.max_batches,
        progress=args.progress,
        progress_every=args.progress_every,
    )
    complete_formal_pass = (
        args.max_shards is None
        and args.max_batches is None
        and args.max_calibration_batches is None
        and bool(evaluation_metadata["complete"])
        and bool(calibration_metadata["complete"])
    )
    formally_eligible = (
        checkpoint.get("status") == "candidate_unvalidated_on_locked_test"
        and args.split == "test"
        and complete_formal_pass
    )
    result = evaluate_predictions(records, horizons, config, formally_eligible=formally_eligible)
    result.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_status": checkpoint.get("status"),
            "checkpoint_shard_config_digest": checkpoint.get(
                "shard_config_digest", checkpoint.get("config_digest")
            ),
            "checkpoint_run_config_digest": checkpoint.get(
                "run_config_digest", checkpoint.get("config_digest")
            ),
            "checkpoint_run_overrides": checkpoint.get("run_overrides", {}),
            "evaluated_split": args.split,
            "split_ranges": serialize_ranges(ranges),
            "shards": [str(selection.path) for selection in selections],
            "evaluation": evaluation_metadata,
            "data_manifest_config_digest": manifest.get("config_digest"),
            "calibration": {
                "method": "per-lead temperature scaling on the calibration split",
                "temperatures": {
                    str(horizon): float(temperatures[index])
                    for index, horizon in enumerate(horizons)
                },
                **calibration_metadata,
            },
        }
    )
    if args.output is not None:
        output = resolve_project_path(config, args.output)
    elif artifact_dir != configured_artifact_dir:
        output = artifact_dir / "evaluation.json"
    else:
        output = resolve_project_path(config, "reports/evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    decision = {
        "selected_model": result["selected_model"],
        "accepted": result["acceptance"]["accepted"],
        "evaluation_report": str(output),
        "checkpoint": str(checkpoint_path),
        "temperatures": result["calibration"]["temperatures"],
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "deployment_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
