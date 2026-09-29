from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
import torch

from .common import GROUP_IDS, atomic_json, read_json, sha256, validate_protocol
from .data import WindowDataset, ready_entries
from .models import TRAINED_MODELS, build_model, model_config
from .representation import decode_states, encode_states, encode_window
from .train import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mandatory real-window overfit gate.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=TRAINED_MODELS)
    parser.add_argument("--group", required=True, choices=GROUP_IDS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=3000)
    return parser.parse_args()


def physical_errors(
    prediction: torch.Tensor, target: torch.Tensor, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    difference = (prediction - target).detach().cpu().numpy()[0].astype(np.float64)
    radius = float(np.linalg.norm(origin[:3]))
    speed = float(np.linalg.norm(origin[3:]))
    return (
        np.linalg.norm(difference[:, :3], axis=1) * radius,
        np.linalg.norm(difference[:, 3:], axis=1) * speed,
    )


def float32_vector_tolerance(values: np.ndarray, minimum: float) -> float:
    maximum = float(np.max(np.abs(np.asarray(values, dtype=np.float32))))
    ulp = float(abs(np.spacing(np.float32(maximum))))
    return max(float(minimum), 2.0 * math.sqrt(3.0) * ulp)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    protocol_path = args.protocol.resolve()
    manifest_path = args.manifest.resolve()
    protocol = validate_protocol(read_json(protocol_path))
    input_steps = int(protocol["input_steps"])
    target_steps = [int(value) for value in protocol["target_steps"]]
    entry = ready_entries(manifest_path, "train", args.group)[0]
    array = WindowDataset._open(entry["array_path"])
    origin_index = input_steps
    context = np.asarray(array[:origin_index], dtype=np.float32).copy()
    targets = np.asarray(array[origin_index + np.asarray(target_steps) - 1], dtype=np.float32).copy()
    origin = context[-1].copy()
    encoded_context, encoded_target = encode_window(context, targets)
    reconstructed_context = decode_states(encoded_context, origin)
    reconstructed_target = decode_states(encoded_target, origin)
    roundtrip_position_m = max(
        float(np.max(np.linalg.norm(reconstructed_context[:, :3] - context[:, :3], axis=1))),
        float(np.max(np.linalg.norm(reconstructed_target[:, :3] - targets[:, :3], axis=1))),
    )
    roundtrip_velocity_m_s = max(
        float(np.max(np.linalg.norm(reconstructed_context[:, 3:] - context[:, 3:], axis=1))),
        float(np.max(np.linalg.norm(reconstructed_target[:, 3:] - targets[:, 3:], axis=1))),
    )
    position_roundtrip_tolerance_m = float32_vector_tolerance(
        np.concatenate((context[:, :3], targets[:, :3])), 2.0
    )
    velocity_roundtrip_tolerance_m_s = float32_vector_tolerance(
        np.concatenate((context[:, 3:], targets[:, 3:])), 0.002
    )
    if (
        roundtrip_position_m > position_roundtrip_tolerance_m
        or roundtrip_velocity_m_s > velocity_roundtrip_tolerance_m_s
    ):
        raise RuntimeError(
            "RTN representation round trip is outside its float32 precision bound: "
            f"position={roundtrip_position_m:.6g}/{position_roundtrip_tolerance_m:.6g} m, "
            f"velocity={roundtrip_velocity_m_s:.6g}/{velocity_roundtrip_tolerance_m_s:.6g} m/s"
        )
    seed = int(protocol["training"]["seeds"][0])
    seed_everything(seed)
    device = resolve_device(args.device)
    sanity_model_config = model_config(args.model, input_steps, target_steps)
    if "dropout" in sanity_model_config:
        sanity_model_config["dropout"] = 0.0
    model = build_model(args.model, sanity_model_config).to(device)
    x = torch.from_numpy(encoded_context).unsqueeze(0).to(device)
    y = torch.from_numpy(encoded_target).unsqueeze(0).to(device)
    # This gate tests representational and optimization capacity on one fixed
    # example. Disabling dropout removes stochastic regularization that is only
    # appropriate for formal multi-window training.
    learning_rate = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=1.0e-6
    )
    position_threshold = float(protocol["quality_gates"]["single_window_max_position_error_m"])
    velocity_threshold = float(protocol["quality_gates"]["single_window_max_velocity_error_m_s"])
    passed = False
    final_position = np.full(len(target_steps), math.inf)
    final_velocity = np.full(len(target_steps), math.inf)
    final_loss = math.inf
    steps_run = 0
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.mean(torch.square(prediction - y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        steps_run = step
        if step == 1 or step % 25 == 0 or step == args.max_steps:
            model.eval()
            with torch.inference_mode():
                prediction = model(x)
            final_loss = float(torch.mean(torch.square(prediction - y)).item())
            final_position, final_velocity = physical_errors(prediction, y, origin)
            if step == 1 or step % 250 == 0:
                print(
                    f"step={step}/{args.max_steps} loss={final_loss:.8g} "
                    f"max_position_m={final_position.max():.3f} "
                    f"max_velocity_m_s={final_velocity.max():.6f}",
                    flush=True,
                )
            if final_position.max() <= position_threshold and final_velocity.max() <= velocity_threshold:
                passed = True
                break
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": protocol["benchmark_id"],
        "protocol_sha256": sha256(protocol_path),
        "manifest_sha256": sha256(manifest_path),
        "model": args.model,
        "group": args.group,
        "norad_id": entry["norad_id"],
        "seed": seed,
        "steps_run": steps_run,
        "max_steps": args.max_steps,
        "final_normalized_mse": final_loss,
        "position_error_m_by_horizon": final_position.tolist(),
        "velocity_error_m_s_by_horizon": final_velocity.tolist(),
        "roundtrip_position_error_m": roundtrip_position_m,
        "roundtrip_velocity_error_m_s": roundtrip_velocity_m_s,
        "roundtrip_position_tolerance_m": position_roundtrip_tolerance_m,
        "roundtrip_velocity_tolerance_m_s": velocity_roundtrip_tolerance_m_s,
        "position_threshold_m": position_threshold,
        "velocity_threshold_m_s": velocity_threshold,
        "model_config": sanity_model_config,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "scheduler": "cosine_annealing",
            "eta_min": 1.0e-6,
            "dropout_active": False,
        },
        "quality_gate_passed": passed,
    }
    atomic_json(args.output.resolve(), report)
    print(report, flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
