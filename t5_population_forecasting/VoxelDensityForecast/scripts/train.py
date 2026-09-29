from __future__ import annotations

"""Train the closed-cohort physics-residual model with bounded shard memory."""

import argparse
import gc
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
# Allow direct execution without installing the package.
sys.path.insert(0, str(ROOT / "src"))

_RUNTIME_OPTION_KEYS = frozenset(
    {
        "batch_size",
        "num_workers",
        "shard_window_size",
        "max_shards_per_split",
        "max_batches",
        "validation_max_batches",
    }
)
_OVERRIDABLE_RUNTIME_OPTION_KEYS = frozenset(
    {
        "batch_size",
        "num_workers",
        "shard_window_size",
    }
)

from starlink_forecast.config import config_digest, load_config, resolve_project_path
from starlink_forecast.losses import joint_multinomial_nll, multiscale_distribution_l1
from starlink_forecast.training import (
    ShardSelection,
    checkpoint_run_config,
    config_with_runtime_overrides,
    dataset_for_selection,
    model_from_config,
    serialize_ranges,
    shard_manifest,
    shard_selections_for_range,
    split_ranges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the closed-cohort physics-residual LEO model")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Run a short, non-formal training pass")
    parser.add_argument(
        "--max-shards",
        type=int,
        help="Debug only: cap selected shards per split after full chronological splitting",
    )
    parser.add_argument("--max-batches", type=int, help="Debug only: cap total train batches per epoch")
    parser.add_argument("--validation-max-batches", type=int, help="Debug only: cap total validation batches")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Runtime override; does not alter the shard digest")
    parser.add_argument("--num-workers", type=int, help="Runtime override; start with zero for large shards")
    parser.add_argument(
        "--artifact-dir",
        help="Output directory for this run; required for model, optimizer, or loss ablations",
    )
    parser.add_argument("--learning-rate", type=float, help="Run-only AdamW learning-rate override")
    parser.add_argument("--weight-decay", type=float, help="Run-only AdamW weight-decay override")
    parser.add_argument("--base-channels", type=int, help="Run-only 3D U-Net width override")
    parser.add_argument("--hidden-channels", type=int, help="Run-only ConvGRU width override")
    parser.add_argument(
        "--multiscale-loss-weight",
        type=float,
        help="Run-only multiscale distribution loss weight override",
    )
    parser.add_argument(
        "--residual-l2-weight",
        type=float,
        help="Run-only residual-logit L2 weight override",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Run-only number of complete stale validation epochs before stopping",
    )
    parser.add_argument(
        "--shard-window-size",
        type=int,
        default=1,
        help="Maximum number of fully loaded shards at once (default: 1)",
    )
    parser.add_argument("--progress", action="store_true", help="Emit per-shard and periodic batch progress")
    parser.add_argument("--progress-every", type=int, default=50, help="Batch interval for --progress")
    parser.add_argument("--resume", help="Resume checkpoint-last.pt after a completed epoch")
    parser.add_argument(
        "--allow-runtime-override",
        action="store_true",
        help="With --resume, allow changing batch size, worker count, or shard window size",
    )
    return parser.parse_args()


def _run_overrides_from_args(args: argparse.Namespace) -> dict[str, dict[str, int | float]]:
    model = {
        name: int(value)
        for name, value in (
            ("base_channels", args.base_channels),
            ("hidden_channels", args.hidden_channels),
        )
        if value is not None
    }
    training = {
        name: value
        for name, value in (
            ("learning_rate", args.learning_rate),
            ("weight_decay", args.weight_decay),
            ("multiscale_loss_weight", args.multiscale_loss_weight),
            ("residual_l2_weight", args.residual_l2_weight),
            ("early_stopping_patience", args.early_stopping_patience),
        )
        if value is not None
    }
    return {"model": model, "training": training}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress(enabled: bool, event: str, **values: object) -> None:
    if enabled:
        print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def _runtime_option_changes(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_override: bool,
) -> dict[str, dict[str, Any]]:
    """Validate resume options and return the explicitly allowed changes."""
    previous_keys = set(previous)
    current_keys = set(current)
    unknown_previous = previous_keys - _RUNTIME_OPTION_KEYS
    unknown_current = current_keys - _RUNTIME_OPTION_KEYS
    missing_previous = _RUNTIME_OPTION_KEYS - previous_keys
    missing_current = _RUNTIME_OPTION_KEYS - current_keys
    if unknown_previous or unknown_current or missing_previous or missing_current:
        details = []
        if unknown_previous:
            details.append(f"checkpoint has unknown keys: {', '.join(sorted(unknown_previous))}")
        if unknown_current:
            details.append(f"runtime has unknown keys: {', '.join(sorted(unknown_current))}")
        if missing_previous:
            details.append(f"checkpoint is missing keys: {', '.join(sorted(missing_previous))}")
        if missing_current:
            details.append(f"runtime is missing keys: {', '.join(sorted(missing_current))}")
        raise SystemExit(f"Resume checkpoint has incompatible runtime data-loading options ({'; '.join(details)})")

    changes = {
        key: {"previous": previous[key], "current": current[key]}
        for key in sorted(_RUNTIME_OPTION_KEYS)
        if previous[key] != current[key]
    }
    if not changes:
        return changes
    if not allow_override:
        raise SystemExit("Resume checkpoint uses different runtime data-loading options")

    disallowed = sorted(set(changes) - _OVERRIDABLE_RUNTIME_OPTION_KEYS)
    if disallowed:
        raise SystemExit(
            "Resume checkpoint changes immutable runtime data-loading options: "
            + ", ".join(disallowed)
        )
    return changes


def _runtime_override_record(
    previous: dict[str, Any], current: dict[str, Any], resumed_after_epoch: int
) -> dict[str, Any]:
    return {
        "resumed_after_epoch": resumed_after_epoch,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "previous": {key: previous[key] for key in sorted(_OVERRIDABLE_RUNTIME_OPTION_KEYS)},
        "current": {key: current[key] for key in sorted(_OVERRIDABLE_RUNTIME_OPTION_KEYS)},
    }


def batch_loss(output: Any, batch: dict[str, torch.Tensor], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    nll_per_sample = joint_multinomial_nll(
        output.voxel_probability,
        output.outside_probability,
        batch["target_counts"],
        batch["target_outside"],
        reduction="none",
        include_constant=False,
    )
    normalized_nll = (nll_per_sample / batch["future_total"].clamp_min(1.0)).mean()
    multiscale = multiscale_distribution_l1(
        output.voxel_probability,
        output.outside_probability,
        batch["target_counts"],
        batch["target_outside"],
    )
    residual_l2 = 0.5 * (
        output.residual_logits.square().mean()
        + output.outside_residual_logits.square().mean()
    )
    training = config["training"]
    total = (
        normalized_nll
        + float(training["multiscale_loss_weight"]) * multiscale
        + float(training["residual_l2_weight"]) * residual_l2
    )
    return total, {
        "nll_per_member": float(normalized_nll.detach().cpu()),
        "multiscale_l1": float(multiscale.detach().cpu()),
        "residual_l2": float(residual_l2.detach().cpu()),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def distribution_tv(output: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    total = batch["future_total"].clamp_min(1.0)
    target_voxel = batch["target_counts"] / total[:, :, None, None, None]
    target_outside = batch["target_outside"] / total[:, :, None]
    return 0.5 * (
        torch.abs(output.voxel_probability - target_voxel).flatten(start_dim=2).sum(dim=2)
        + torch.abs(output.outside_probability - target_outside).sum(dim=2)
    ).mean()


def _windowed(values: list[ShardSelection], size: int) -> list[list[ShardSelection]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _loader_for_window(
    selections: list[ShardSelection],
    config: dict[str, Any],
    time_range: Any,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, Any, list[Any]]:
    datasets = [
        dataset_for_selection(selection, config, time_range, return_tensors=True)
        for selection in selections
    ]
    if not datasets:
        raise RuntimeError("A shard window must contain at least one selected shard")
    source: Any = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(
        source,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        generator=generator,
    )
    return loader, source, datasets


def validate(
    model: torch.nn.Module,
    selections: list[ShardSelection],
    shard_config: dict[str, Any],
    run_config: dict[str, Any],
    time_range: Any,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    progress: bool,
    progress_every: int,
) -> dict[str, float | int | bool]:
    """Validate one shard at a time and aggregate sample-weighted metrics."""

    model.eval()
    totals: dict[str, float | int] = {"loss": 0.0, "tv": 0.0, "batches": 0, "samples": 0, "shards": 0}
    expected_samples = sum(selection.origin_count for selection in selections)
    stopped = False
    with torch.no_grad():
        for shard_index, selection in enumerate(selections, start=1):
            if max_batches is not None and int(totals["batches"]) >= max_batches:
                stopped = True
                break
            _progress(
                progress,
                "validation_shard_start",
                shard_index=shard_index,
                shard_count=len(selections),
                shard=selection.path.name,
                shard_origins=selection.origin_count,
            )
            loader, source, datasets = _loader_for_window(
                [selection],
                shard_config,
                time_range,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
            try:
                for raw_batch in loader:
                    if max_batches is not None and int(totals["batches"]) >= max_batches:
                        stopped = True
                        break
                    batch = move_batch(raw_batch, device)
                    output = model(
                        batch["features"],
                        batch["physics_prior"],
                        batch["physics_prior_outside"],
                        batch["future_total"],
                    )
                    loss, _ = batch_loss(output, batch, run_config)
                    sample_count = int(batch["features"].shape[0])
                    totals["loss"] = float(totals["loss"]) + float(loss.cpu()) * sample_count
                    totals["tv"] = float(totals["tv"]) + float(distribution_tv(output, batch).cpu()) * sample_count
                    totals["batches"] = int(totals["batches"]) + 1
                    totals["samples"] = int(totals["samples"]) + sample_count
                    if progress and int(totals["batches"]) % progress_every == 0:
                        _progress(
                            True,
                            "validation_batch",
                            batches=int(totals["batches"]),
                            samples=int(totals["samples"]),
                            shard=selection.path.name,
                        )
                    if max_batches is not None and int(totals["batches"]) >= max_batches:
                        stopped = True
                        break
                totals["shards"] = int(totals["shards"]) + 1
            finally:
                del loader, source, datasets
                gc.collect()
            _progress(
                progress,
                "validation_shard_done",
                shard_index=shard_index,
                batches=int(totals["batches"]),
                samples=int(totals["samples"]),
            )
            if stopped:
                break
    if not int(totals["batches"]):
        raise RuntimeError("Validation stream produced no batches")
    complete = not stopped and int(totals["samples"]) == expected_samples
    return {
        "loss": float(totals["loss"]) / int(totals["samples"]),
        "tv": float(totals["tv"]) / int(totals["samples"]),
        "batches": int(totals["batches"]),
        "samples": int(totals["samples"]),
        "shards": int(totals["shards"]),
        "expected_samples": expected_samples,
        "complete": complete,
    }


def main() -> int:
    args = parse_args()
    for name, value in (
        ("--max-shards", args.max_shards),
        ("--max-batches", args.max_batches),
        ("--validation-max-batches", args.validation_max_batches),
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--shard-window-size", args.shard_window_size),
        ("--progress-every", args.progress_every),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{name} must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise SystemExit("--num-workers must be zero or positive")
    if args.allow_runtime_override and not args.resume:
        raise SystemExit("--allow-runtime-override requires --resume")
    for name, value in (
        ("--learning-rate", args.learning_rate),
        ("--base-channels", args.base_channels),
        ("--hidden-channels", args.hidden_channels),
        ("--early-stopping-patience", args.early_stopping_patience),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{name} must be positive")
    for name, value in (
        ("--weight-decay", args.weight_decay),
        ("--multiscale-loss-weight", args.multiscale_loss_weight),
        ("--residual-l2-weight", args.residual_l2_weight),
    ):
        if value is not None and value < 0:
            raise SystemExit(f"{name} cannot be negative")

    shard_config = load_config(args.config)
    if shard_config["forecast"]["population_mode"] != "origin_known_members":
        raise SystemExit("Formal training requires forecast.population_mode=origin_known_members")
    if shard_config["forecast"]["prior_mode"] != "causal_kepler_teme":
        raise SystemExit("Formal training requires forecast.prior_mode=causal_kepler_teme")
    run_overrides = _run_overrides_from_args(args)
    try:
        run_config = config_with_runtime_overrides(
            shard_config,
            model_overrides=run_overrides["model"],
            training_overrides=run_overrides["training"],
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    manifest, all_shards = shard_manifest(shard_config)
    # Split on the complete manifest before any debug limit is applied.
    ranges = split_ranges(shard_config, all_shards)
    train_selections = shard_selections_for_range(all_shards, ranges["train"])
    validation_selections = shard_selections_for_range(all_shards, ranges["validation"])
    if args.max_shards is not None:
        train_selections = train_selections[: args.max_shards]
        validation_selections = validation_selections[: args.max_shards]
    if not train_selections or not validation_selections:
        raise SystemExit("Chronological train/validation ranges contain no usable origins")

    training = run_config["training"]
    seed = int(training["seed"])
    batch_size = int(args.batch_size if args.batch_size is not None else training["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else training["num_workers"])
    configured_validation_max = training.get("validation_max_batches")
    validation_max = (
        args.validation_max_batches
        if args.validation_max_batches is not None
        else (int(configured_validation_max) if configured_validation_max is not None else None)
    )
    if args.smoke and validation_max is None:
        validation_max = 2
    max_batches = args.max_batches if args.max_batches is not None else (2 if args.smoke else None)
    epochs = int(args.epochs or (1 if args.smoke else training["epochs"]))
    runtime_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shard_window_size": int(args.shard_window_size),
        "max_shards_per_split": args.max_shards,
        "max_batches": max_batches,
        "validation_max_batches": validation_max,
    }
    debug_run = any(
        (
            args.smoke,
            args.max_shards is not None,
            max_batches is not None,
            validation_max is not None,
        )
    )
    status = "smoke_pipeline" if debug_run else "candidate_unvalidated_on_locked_test"
    has_run_overrides = any(run_overrides.values())
    if has_run_overrides and not debug_run and args.artifact_dir is None:
        raise SystemExit(
            "Run-only model, optimizer, or loss overrides require --artifact-dir so formal artifacts are not overwritten"
        )
    artifact_dir = resolve_project_path(
        shard_config,
        args.artifact_dir if args.artifact_dir is not None else shard_config["paths"]["artifact_dir"],
    )
    if not debug_run:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    shard_digest = config_digest(shard_config)
    run_digest = config_digest(run_config)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(training.get("mixed_precision", True) and device.type == "cuda")
    model = model_from_config(run_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if num_workers:
        print("warning: worker processes can multiply resident shard memory; start with --num-workers 0", flush=True)
    _progress(
        args.progress,
        "runtime",
        device=str(device),
        amp=use_amp,
        train_shards=len(train_selections),
        train_origins=sum(selection.origin_count for selection in train_selections),
        validation_shards=len(validation_selections),
        validation_origins=sum(selection.origin_count for selection in validation_selections),
        runtime_options=runtime_options,
        shard_config_digest=shard_digest,
        run_config_digest=run_digest,
        run_overrides=run_overrides,
        artifact_dir=str(artifact_dir) if not debug_run else None,
    )

    best_checkpoint = artifact_dir / "model-best.pt"
    best_checkpoint_available = bool(not debug_run and args.resume and best_checkpoint.is_file())
    history: list[dict[str, Any]] = []
    runtime_change_history: list[dict[str, Any]] = []
    start_epoch = 1
    best_validation = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        try:
            checkpoint_config = checkpoint_run_config(checkpoint, shard_config)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if config_digest(checkpoint_config) != run_digest:
            raise SystemExit("Resume checkpoint uses a different model, optimizer, or loss configuration")
        previous_runtime = checkpoint.get("runtime_options")
        runtime_change_history = list(checkpoint.get("runtime_change_history", []))
        if previous_runtime is not None:
            if not isinstance(previous_runtime, dict):
                raise SystemExit("Resume checkpoint has invalid runtime data-loading options")
            changes = _runtime_option_changes(
                previous_runtime,
                runtime_options,
                allow_override=args.allow_runtime_override,
            )
            if changes:
                override_record = _runtime_override_record(
                    previous_runtime,
                    runtime_options,
                    resumed_after_epoch=int(checkpoint["epoch"]),
                )
                runtime_change_history.append(override_record)
                _progress(args.progress, "resume_runtime_override", **override_record)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        best_validation = float(checkpoint.get("best_validation", best_validation))

    patience = int(training["early_stopping_patience"])
    stale_epochs = 0
    expected_train_samples = sum(selection.origin_count for selection in train_selections)
    data_scope = {
        "train": {
            "shards": [str(selection.path) for selection in train_selections],
            "origin_count": expected_train_samples,
        },
        "validation": {
            "shards": [str(selection.path) for selection in validation_selections],
            "origin_count": sum(selection.origin_count for selection in validation_selections),
        },
    }

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        sums: dict[str, float | int] = {
            "loss": 0.0,
            "nll": 0.0,
            "multiscale": 0.0,
            "residual": 0.0,
            "batches": 0,
            "samples": 0,
            "shards": 0,
        }
        train_order = list(train_selections)
        random.Random(seed + epoch).shuffle(train_order)
        stopped = False
        for window_index, window in enumerate(_windowed(train_order, int(args.shard_window_size)), start=1):
            if max_batches is not None and int(sums["batches"]) >= max_batches:
                stopped = True
                break
            _progress(
                args.progress,
                "train_window_start",
                epoch=epoch,
                window_index=window_index,
                window_count=(len(train_order) + int(args.shard_window_size) - 1) // int(args.shard_window_size),
                shards=[selection.path.name for selection in window],
                origins=sum(selection.origin_count for selection in window),
            )
            generator = torch.Generator()
            generator.manual_seed(seed + epoch * 1_000_003 + window_index)
            loader, source, datasets = _loader_for_window(
                window,
                shard_config,
                ranges["train"],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                pin_memory=device.type == "cuda",
                generator=generator,
            )
            try:
                for raw_batch in loader:
                    if max_batches is not None and int(sums["batches"]) >= max_batches:
                        stopped = True
                        break
                    batch = move_batch(raw_batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        output = model(
                            batch["features"],
                            batch["physics_prior"],
                            batch["physics_prior_outside"],
                            batch["future_total"],
                        )
                        loss, components = batch_loss(output, batch, run_config)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                    sample_count = int(batch["features"].shape[0])
                    sums["loss"] = float(sums["loss"]) + float(loss.detach().cpu()) * sample_count
                    sums["nll"] = float(sums["nll"]) + components["nll_per_member"] * sample_count
                    sums["multiscale"] = float(sums["multiscale"]) + components["multiscale_l1"] * sample_count
                    sums["residual"] = float(sums["residual"]) + components["residual_l2"] * sample_count
                    sums["batches"] = int(sums["batches"]) + 1
                    sums["samples"] = int(sums["samples"]) + sample_count
                    if args.progress and int(sums["batches"]) % args.progress_every == 0:
                        _progress(
                            True,
                            "train_batch",
                            epoch=epoch,
                            batches=int(sums["batches"]),
                            samples=int(sums["samples"]),
                            window_index=window_index,
                        )
                    if max_batches is not None and int(sums["batches"]) >= max_batches:
                        stopped = True
                        break
                sums["shards"] = int(sums["shards"]) + len(window)
            finally:
                del loader, source, datasets
                gc.collect()
            _progress(
                args.progress,
                "train_window_done",
                epoch=epoch,
                window_index=window_index,
                batches=int(sums["batches"]),
                samples=int(sums["samples"]),
            )
            if stopped:
                break

        if not int(sums["batches"]):
            raise SystemExit("Training stream produced no batches")
        if max_batches is None and int(sums["samples"]) != expected_train_samples:
            raise RuntimeError(
                f"Training coverage mismatch: expected {expected_train_samples} origins, got {sums['samples']}"
            )
        validation = validate(
            model,
            validation_selections,
            shard_config,
            run_config,
            ranges["validation"],
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_batches=validation_max,
            progress=args.progress,
            progress_every=args.progress_every,
        )
        if validation_max is None and not bool(validation["complete"]):
            raise RuntimeError(
                f"Validation coverage mismatch: expected {validation['expected_samples']} origins, "
                f"got {validation['samples']}"
            )
        sample_count = int(sums["samples"])
        record = {
            "epoch": epoch,
            "train_loss": float(sums["loss"]) / sample_count,
            "train_nll_per_member": float(sums["nll"]) / sample_count,
            "train_multiscale_l1": float(sums["multiscale"]) / sample_count,
            "train_residual_l2": float(sums["residual"]) / sample_count,
            "train_batches": int(sums["batches"]),
            "train_samples": sample_count,
            "train_shards": int(sums["shards"]),
            "validation_loss": validation["loss"],
            "validation_tv": validation["tv"],
            "validation_batches": validation["batches"],
            "validation_samples": validation["samples"],
            "validation_expected_samples": validation["expected_samples"],
            "validation_shards": validation["shards"],
            "validation_complete": validation["complete"],
        }
        history.append(record)

        # A truncated validation pass is useful for smoke tests, but must not
        # drive model selection or early stopping.
        improved = (
            not debug_run
            and validation_max is None
            and bool(validation["complete"])
            and float(validation["loss"]) < best_validation
        )
        if improved:
            best_validation = float(validation["loss"])
            stale_epochs = 0
        elif not debug_run and validation_max is None and bool(validation["complete"]):
            stale_epochs += 1

        if not debug_run:
            payload = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": run_config,
                "config_digest": run_digest,
                "shard_config": shard_config,
                "shard_config_digest": shard_digest,
                "run_config": run_config,
                "run_config_digest": run_digest,
                "run_overrides": run_overrides,
                "artifact_dir": str(artifact_dir),
                "runtime_options": runtime_options,
                "data_manifest": manifest,
                "data_scope": data_scope,
                "split_ranges": serialize_ranges(ranges),
                "horizons": tuple(int(value) for value in shard_config["forecast"]["horizons"]),
                "status": status,
                "epoch": epoch,
                "history": history,
                "runtime_change_history": runtime_change_history,
                "best_validation": best_validation,
            }
            torch.save(payload, artifact_dir / "checkpoint-last.pt")
            if improved:
                torch.save(payload, artifact_dir / "model-best.pt")
                best_checkpoint_available = True
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if not debug_run and stale_epochs >= patience:
            print(f"early stopping after {stale_epochs} stale validation epochs", flush=True)
            break

    run_record = {
        "status": status,
        "device": str(device),
        "config_digest": run_digest,
        "shard_config_digest": shard_digest,
        "run_config_digest": run_digest,
        "run_overrides": run_overrides,
        "artifact_dir": str(artifact_dir),
        "effective_model_config": run_config["model"],
        "effective_training_config": run_config["training"],
        "seed": seed,
        "runtime_options": runtime_options,
        "runtime_change_history": runtime_change_history,
        "split_ranges": serialize_ranges(ranges),
        "data_scope": data_scope,
        "history": history,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint_available else None,
    }
    if not debug_run:
        (artifact_dir / "training_run.json").write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if debug_run:
        print("debug run: no checkpoint or training_run.json written", flush=True)
    elif best_checkpoint_available:
        print(f"best checkpoint: {best_checkpoint}", flush=True)
    else:
        print("no model-best.pt written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
