from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from .common import GROUP_IDS, atomic_json, read_json, sha256, validate_protocol
from .data import ready_entries, select_entries
from .models import TRAINED_MODELS, build_model, model_config
from .pilot_representation import PILOT_REPRESENTATIONS, PilotWindowDataset
from .train import resolve_device, save_checkpoint_atomic, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic comparison of endpoint and full-trajectory supervision."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=TRAINED_MODELS)
    parser.add_argument("--group", required=True, choices=GROUP_IDS)
    parser.add_argument("--objects", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--horizon-step", required=True, type=int)
    parser.add_argument("--supervision", required=True, choices=("endpoint", "trajectory"))
    parser.add_argument(
        "--representation", choices=PILOT_REPRESENTATIONS, default="origin_rtn_relative"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-windows-per-epoch", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def supervision_steps(horizon_step: int, mode: str) -> list[int]:
    if horizon_step <= 0:
        raise ValueError("horizon-step must be positive")
    if mode == "endpoint":
        return [horizon_step]
    if mode == "trajectory":
        return list(range(1, horizon_step + 1))
    raise ValueError(f"unsupported supervision mode: {mode}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    gradient_clip_norm: float,
    max_batches: int | None,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    element_count = 0
    window_count = 0
    for batch_index, (contexts, targets, _origins) in enumerate(loader, start=1):
        contexts = contexts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            prediction = model(contexts)
            loss = torch.mean(torch.square(prediction - targets))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        elements = prediction.numel()
        loss_sum += float(loss.detach().item()) * elements
        element_count += elements
        window_count += len(contexts)
        if batch_index % 250 == 0:
            print(
                f"batch={batch_index}/{len(loader)} train_mse={loss_sum / element_count:.8g}",
                flush=True,
            )
        if max_batches is not None and batch_index >= max_batches:
            break
    if window_count == 0:
        raise RuntimeError("training produced no windows")
    return loss_sum / element_count, window_count


def validate_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    position_sum = 0.0
    velocity_sum = 0.0
    endpoint_position_sum = 0.0
    endpoint_velocity_sum = 0.0
    window_count = 0
    element_count = 0
    with torch.inference_mode():
        for contexts, targets, physical_scales in loader:
            contexts = contexts.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = model(contexts)
            difference = prediction - targets
            mse_sum += float(torch.square(difference).sum().item())
            element_count += difference.numel()
            physical_scales = physical_scales.to(device, non_blocking=True)
            physical_difference = difference * physical_scales[:, None, :]
            position_errors = torch.linalg.vector_norm(
                physical_difference[:, :, :3], dim=2
            )
            velocity_errors = torch.linalg.vector_norm(
                physical_difference[:, :, 3:], dim=2
            )
            position_sum += float(position_errors.sum().item())
            velocity_sum += float(velocity_errors.sum().item())
            endpoint_position_sum += float(position_errors[:, -1].sum().item())
            endpoint_velocity_sum += float(velocity_errors[:, -1].sum().item())
            window_count += len(contexts)
    if window_count == 0 or element_count == 0:
        raise RuntimeError("validation produced no windows")
    predicted_steps = int(loader.dataset.target_steps.size)
    return {
        "normalized_mse": mse_sum / element_count,
        "trajectory_position_error_mean_m": position_sum / (window_count * predicted_steps),
        "trajectory_velocity_error_mean_m_s": velocity_sum / (window_count * predicted_steps),
        "endpoint_position_error_mean_m": endpoint_position_sum / window_count,
        "endpoint_velocity_error_mean_m_s": endpoint_velocity_sum / window_count,
    }


def main() -> int:
    args = parse_args()
    if args.objects <= 0 or args.num_workers < 0:
        raise ValueError("objects must be positive and num-workers cannot be negative")
    for output in (args.checkpoint, args.report):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

    protocol_path = args.protocol.resolve()
    manifest_path = args.manifest.resolve()
    protocol = validate_protocol(read_json(protocol_path))
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else protocol["training"]["learning_rate"]
    )
    if (
        args.max_epochs <= 0
        or args.patience < 0
        or args.max_windows_per_epoch <= 0
        or learning_rate <= 0.0
    ):
        raise ValueError("invalid training limits")
    if args.horizon_step not in [int(value) for value in protocol["target_steps"]]:
        raise ValueError("horizon-step must be one of the formal endpoint horizons")
    allowed_seeds = [int(value) for value in protocol["training"]["seeds"]]
    if args.seed not in allowed_seeds:
        raise ValueError(f"seed {args.seed} is not locked in the protocol")
    target_steps = supervision_steps(args.horizon_step, args.supervision)

    seed_everything(args.seed)
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
    device = resolve_device(args.device)
    input_steps = int(protocol["input_steps"])
    train_entries = select_entries(ready_entries(manifest_path, "train", args.group), args.objects)
    val_entries = select_entries(ready_entries(manifest_path, "val", args.group), args.objects)
    if [item["norad_id"] for item in train_entries] != [item["norad_id"] for item in val_entries]:
        raise ValueError("train and validation must follow the same fixed cohort")
    train_dataset = PilotWindowDataset(
        train_entries,
        input_steps,
        target_steps,
        int(protocol["train_stride_steps"]),
        args.representation,
    )
    val_dataset = PilotWindowDataset(
        val_entries,
        input_steps,
        target_steps,
        int(protocol["validation_origin_stride_steps"]),
        args.representation,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.max_windows_per_epoch < len(train_dataset):
        sampler = RandomSampler(
            train_dataset,
            replacement=False,
            num_samples=args.max_windows_per_epoch,
            generator=generator,
        )
    batch_size = int(protocol["training"]["batch_size"])
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    config = model_config(args.model, input_steps, target_steps)
    model = build_model(args.model, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    manifest_digest = sha256(manifest_path)
    protocol_digest = sha256(protocol_path)
    best_position = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    print(
        {
            "diagnostic_only": True,
            "model": args.model,
            "group": args.group,
            "objects": args.objects,
            "horizon_step": args.horizon_step,
            "supervision": args.supervision,
            "representation": args.representation,
            "predicted_steps": len(target_steps),
            "device": str(device),
            "train_windows_available": len(train_dataset),
            "train_windows_per_epoch": min(len(train_dataset), args.max_windows_per_epoch),
            "val_windows": len(val_dataset),
        },
        flush=True,
    )
    for epoch in range(1, args.max_epochs + 1):
        train_loss, examples = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            float(protocol["training"]["gradient_clip_norm"]),
            args.max_batches,
        )
        validation = validate_epoch(model, val_loader, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "examples": examples,
            "learning_rate": learning_rate,
            "train_normalized_mse": train_loss,
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch}/{args.max_epochs} train_mse={train_loss:.8g} "
            f"val_trajectory_pos_m={validation['trajectory_position_error_mean_m']:.3f} "
            f"val_endpoint_pos_m={validation['endpoint_position_error_mean_m']:.3f}",
            flush=True,
        )
        selection_key = (
            "trajectory_position_error_mean_m"
            if args.supervision == "trajectory"
            else "endpoint_position_error_mean_m"
        )
        if validation[selection_key] < best_position:
            best_position = validation[selection_key]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint_atomic(
                args.checkpoint.resolve(),
                {
                    "schema_version": 1,
                    "diagnostic_only": True,
                    "formal_eligible": False,
                    "benchmark_id": protocol["benchmark_id"],
                    "protocol_sha256": protocol_digest,
                    "manifest_sha256": manifest_digest,
                    "model": args.model,
                    "group": args.group,
                    "seed": args.seed,
                    "objects": args.objects,
                    "horizon_step": args.horizon_step,
                    "supervision": args.supervision,
                    "representation": args.representation,
                    "target_steps": target_steps,
                    "model_config": config,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    if best_epoch == 0 or not args.checkpoint.is_file():
        raise RuntimeError("pilot training did not produce a checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    complete = args.max_batches is None
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "formal_eligible": False,
        "complete": complete,
        "reason": "Tests whether sparse endpoint supervision caused the cross-window error.",
        "benchmark_id": protocol["benchmark_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_digest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "model": args.model,
        "group": args.group,
        "seed": args.seed,
        "objects": args.objects,
        "horizon_step": args.horizon_step,
        "supervision": args.supervision,
        "representation": args.representation,
        "target_steps": target_steps,
        "model_config": config,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": float(protocol["training"]["weight_decay"]),
            "scheduler": "cosine_annealing",
            "batch_size": batch_size,
        },
        "limits": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "max_windows_per_epoch": args.max_windows_per_epoch,
            "max_batches": args.max_batches,
        },
        "best_epoch": best_epoch,
        "selection_metric": (
            "trajectory_position_error_mean_m"
            if args.supervision == "trajectory"
            else "endpoint_position_error_mean_m"
        ),
        "best_validation": checkpoint["validation"],
        "history": history,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
    }
    atomic_json(args.report.resolve(), report)
    print(
        {
            "status": "complete" if complete else "debug_incomplete",
            "formal_eligible": False,
            "best_epoch": best_epoch,
            "best_validation": checkpoint["validation"],
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
