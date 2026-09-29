from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from .common import GROUP_IDS, atomic_json, read_json, sha256, validate_protocol
from .data import WindowDataset, ready_entries, select_entries
from .models import TRAINED_MODELS, build_model, model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one group/seed orbit forecasting model.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=TRAINED_MODELS)
    parser.add_argument("--group", required=True, choices=GROUP_IDS)
    parser.add_argument("--objects", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--run-kind", choices=("study", "formal"), default="formal")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-windows-per-epoch", type=int)
    parser.add_argument("--max-batches", type=int, help="Debug only; makes the report incomplete.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    one_hour_index: int,
) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    position_sum = 0.0
    velocity_sum = 0.0
    one_hour_position_sum = 0.0
    window_count = 0
    element_count = 0
    with torch.inference_mode():
        for contexts, targets, origins in loader:
            contexts = contexts.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = model(contexts)
            difference = prediction - targets
            mse_sum += float(torch.square(difference).sum().item())
            element_count += difference.numel()
            radii = torch.linalg.vector_norm(origins[:, :3], dim=1).to(device)
            speeds = torch.linalg.vector_norm(origins[:, 3:], dim=1).to(device)
            position_errors = torch.linalg.vector_norm(difference[:, :, :3], dim=2) * radii[:, None]
            velocity_errors = torch.linalg.vector_norm(difference[:, :, 3:], dim=2) * speeds[:, None]
            position_sum += float(position_errors.sum().item())
            velocity_sum += float(velocity_errors.sum().item())
            one_hour_position_sum += float(position_errors[:, one_hour_index].sum().item())
            window_count += len(contexts)
    if window_count == 0 or element_count == 0:
        raise RuntimeError("validation produced no windows")
    horizons = loader.dataset.target_steps.size
    return {
        "normalized_mse": mse_sum / element_count,
        "position_error_mean_m": position_sum / (window_count * horizons),
        "velocity_error_mean_m_s": velocity_sum / (window_count * horizons),
        "one_hour_position_error_mean_m": one_hour_position_sum / window_count,
    }


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
    manifest_digest = sha256(manifest_path)
    allowed_seeds = [int(value) for value in protocol["training"]["seeds"]]
    if args.seed not in allowed_seeds:
        raise ValueError(f"seed {args.seed} is not locked in the protocol")
    spec = protocol["training_size_study"] if args.run_kind == "study" else protocol["training"]
    max_epochs = int(args.max_epochs or spec["max_epochs"])
    patience = int(args.patience if args.patience is not None else spec["patience"])
    window_cap = int(
        args.max_windows_per_epoch
        if args.max_windows_per_epoch is not None
        else spec["max_windows_per_epoch"]
    )
    if max_epochs <= 0 or patience < 0 or window_cap <= 0:
        raise ValueError("invalid training limits")
    seed_everything(args.seed)
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
    device = resolve_device(args.device)
    input_steps = int(protocol["input_steps"])
    target_steps = [int(value) for value in protocol["target_steps"]]
    one_hour_index = target_steps.index(60)
    train_entries = select_entries(ready_entries(manifest_path, "train", args.group), args.objects)
    val_entries = select_entries(ready_entries(manifest_path, "val", args.group), args.objects)
    if [entry["norad_id"] for entry in train_entries] != [entry["norad_id"] for entry in val_entries]:
        raise ValueError("train and validation must follow the same fixed cohort")
    train_dataset = WindowDataset(
        train_entries, input_steps, target_steps, int(protocol["train_stride_steps"])
    )
    val_dataset = WindowDataset(
        val_entries,
        input_steps,
        target_steps,
        int(protocol["validation_origin_stride_steps"]),
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if window_cap < len(train_dataset):
        sampler = RandomSampler(
            train_dataset,
            replacement=False,
            num_samples=window_cap,
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
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_position = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = args.checkpoint.resolve()
    print(
        {
            "model": args.model,
            "group": args.group,
            "seed": args.seed,
            "run_kind": args.run_kind,
            "device": str(device),
            "objects": args.objects,
            "train_windows_available": len(train_dataset),
            "train_windows_per_epoch": min(len(train_dataset), window_cap),
            "val_windows": len(val_dataset),
        },
        flush=True,
    )
    for epoch in range(1, max_epochs + 1):
        train_loss, examples = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            float(protocol["training"]["gradient_clip_norm"]),
            args.max_batches,
        )
        validation = validate_epoch(model, val_loader, device, one_hour_index)
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
            f"epoch={epoch}/{max_epochs} train_mse={train_loss:.8g} "
            f"val_pos_m={validation['position_error_mean_m']:.3f} "
            f"val_1h_pos_m={validation['one_hour_position_error_mean_m']:.3f}",
            flush=True,
        )
        if validation["position_error_mean_m"] < best_position:
            best_position = validation["position_error_mean_m"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint_atomic(
                checkpoint_path,
                {
                    "schema_version": 2,
                    "benchmark_id": protocol["benchmark_id"],
                    "protocol_sha256": sha256(protocol_path),
                    "manifest_sha256": manifest_digest,
                    "representation": protocol["representation"],
                    "model": args.model,
                    "group": args.group,
                    "seed": args.seed,
                    "model_config": config,
                    "state_dict": model.state_dict(),
                    "objects": args.objects,
                    "epoch": epoch,
                    "validation": validation,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_epoch == 0 or not checkpoint_path.is_file():
        raise RuntimeError("training did not produce a checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_validation = payload["validation"]
    threshold = float(protocol["quality_gates"]["formal_validation_one_hour_position_error_m"])
    quality_gate_passed = bool(
        math.isfinite(float(best_validation["one_hour_position_error_mean_m"]))
        and float(best_validation["one_hour_position_error_mean_m"]) <= threshold
    )
    complete = args.max_batches is None
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": protocol["benchmark_id"],
        "protocol_sha256": sha256(protocol_path),
        "manifest_sha256": manifest_digest,
        "representation": protocol["representation"],
        "complete": complete,
        "quality_gate_passed": quality_gate_passed,
        "quality_gate_threshold_one_hour_position_m": threshold,
        "run_kind": args.run_kind,
        "model": args.model,
        "group": args.group,
        "seed": args.seed,
        "model_config": config,
        "objects": args.objects,
        "train_norad_ids": [entry["norad_id"] for entry in train_entries],
        "validation_norad_ids": [entry["norad_id"] for entry in val_entries],
        "train_windows_available": len(train_dataset),
        "train_windows_per_epoch": min(len(train_dataset), window_cap),
        "validation_windows": len(val_dataset),
        "max_epochs": max_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "history": history,
        "debug_max_batches": args.max_batches,
    }
    atomic_json(args.report.resolve(), report)
    print(
        {
            "status": "complete" if complete else "debug_incomplete",
            "quality_gate_passed": quality_gate_passed,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
