from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import GROUP_IDS, atomic_json, read_json, sha256, validate_protocol
from .data import iter_evaluation_batches, ready_entries, select_entries
from .models import build_model
from .representation import encode_states
from .train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a failed checkpoint without formalizing it.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group", required=True, choices=GROUP_IDS)
    parser.add_argument("--objects", required=True, type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-origins-per-object", type=int, default=8)
    return parser.parse_args()


class Moments:
    def __init__(self, horizons: int, channels: int) -> None:
        self.count = 0
        self.target_sum = np.zeros((horizons, channels), dtype=np.float64)
        self.target_sum_sq = np.zeros((horizons, channels), dtype=np.float64)
        self.prediction_sum = np.zeros((horizons, channels), dtype=np.float64)
        self.prediction_sum_sq = np.zeros((horizons, channels), dtype=np.float64)
        self.cross_sum = np.zeros((horizons, channels), dtype=np.float64)
        self.position_error_sum = np.zeros(horizons, dtype=np.float64)
        self.velocity_error_sum = np.zeros(horizons, dtype=np.float64)

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        radii: np.ndarray,
        speeds: np.ndarray,
    ) -> None:
        difference = prediction - target
        self.count += len(target)
        self.target_sum += target.sum(axis=0, dtype=np.float64)
        self.target_sum_sq += np.square(target, dtype=np.float64).sum(axis=0)
        self.prediction_sum += prediction.sum(axis=0, dtype=np.float64)
        self.prediction_sum_sq += np.square(prediction, dtype=np.float64).sum(axis=0)
        self.cross_sum += np.multiply(prediction, target, dtype=np.float64).sum(axis=0)
        self.position_error_sum += (
            np.linalg.norm(difference[:, :, :3], axis=2) * radii[:, None]
        ).sum(axis=0)
        self.velocity_error_sum += (
            np.linalg.norm(difference[:, :, 3:], axis=2) * speeds[:, None]
        ).sum(axis=0)

    @staticmethod
    def _mean_std(
        total: np.ndarray, total_sq: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        return mean, np.sqrt(variance)

    def summary(self, target_steps: list[int]) -> dict[str, Any]:
        if self.count <= 0:
            raise RuntimeError("diagnostic collected no samples")
        target_mean, target_std = self._mean_std(self.target_sum, self.target_sum_sq, self.count)
        prediction_mean, prediction_std = self._mean_std(
            self.prediction_sum, self.prediction_sum_sq, self.count
        )
        covariance = self.cross_sum / self.count - target_mean * prediction_mean
        denominator = target_std * prediction_std
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0.0,
        )
        rows = []
        for index, step in enumerate(target_steps):
            rows.append(
                {
                    "horizon_step": step,
                    "position_error_mean_m": float(self.position_error_sum[index] / self.count),
                    "velocity_error_mean_m_s": float(self.velocity_error_sum[index] / self.count),
                    "target_mean": target_mean[index].tolist(),
                    "target_std": target_std[index].tolist(),
                    "prediction_mean": prediction_mean[index].tolist(),
                    "prediction_std": prediction_std[index].tolist(),
                    "prediction_target_correlation": correlation[index].tolist(),
                }
            )
        return {"n_origins": self.count, "horizons": rows}


def evaluate_split(
    model: torch.nn.Module,
    entries: list[dict[str, Any]],
    input_steps: int,
    target_steps: list[int],
    stride: int,
    batch_size: int,
    max_origins: int,
    device: torch.device,
) -> dict[str, Any]:
    moments = Moments(len(target_steps), 6)
    model.eval()
    with torch.inference_mode():
        for contexts, truths in iter_evaluation_batches(
            entries,
            input_steps,
            target_steps,
            stride,
            batch_size,
            max_origins,
        ):
            origins = contexts[:, -1]
            encoded_contexts = encode_states(contexts, origins)
            encoded_truths = encode_states(truths, origins)
            predictions = model(
                torch.as_tensor(encoded_contexts, dtype=torch.float32, device=device)
            ).detach().cpu().numpy()
            radii = np.linalg.norm(origins[:, :3], axis=1)
            speeds = np.linalg.norm(origins[:, 3:], axis=1)
            moments.update(predictions, encoded_truths, radii, speeds)
    return moments.summary(target_steps)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if min(args.objects, args.batch_size, args.max_origins_per_object) <= 0:
        raise ValueError("objects, batch-size, and max-origins-per-object must be positive")
    protocol_path = args.protocol.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    protocol = validate_protocol(read_json(protocol_path))
    protocol_digest = sha256(protocol_path)
    manifest_digest = sha256(manifest_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "protocol_sha256": protocol_digest,
        "manifest_sha256": manifest_digest,
        "group": args.group,
        "objects": args.objects,
    }
    mismatches = {
        key: {"expected": value, "found": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint provenance mismatch: {mismatches}")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no model configuration")
    model_name = str(payload["model"])
    device = resolve_device(args.device)
    model = build_model(model_name, config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    input_steps = int(protocol["input_steps"])
    target_steps = [int(value) for value in protocol["target_steps"]]
    split_reports: dict[str, Any] = {}
    cohort_ids: dict[str, list[str]] = {}
    for split in ("train", "val"):
        entries = select_entries(ready_entries(manifest_path, split, args.group), args.objects)
        cohort_ids[split] = [str(item["norad_id"]) for item in entries]
        split_reports[split] = evaluate_split(
            model,
            entries,
            input_steps,
            target_steps,
            int(protocol["validation_origin_stride_steps"]),
            args.batch_size,
            args.max_origins_per_object,
            device,
        )
    if cohort_ids["train"] != cohort_ids["val"]:
        raise ValueError("training and validation cohorts differ")
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "formal_eligible": False,
        "benchmark_id": protocol["benchmark_id"],
        "protocol_sha256": protocol_digest,
        "manifest_sha256": manifest_digest,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "model": model_name,
        "group": args.group,
        "objects": args.objects,
        "cohort_norad_ids": cohort_ids["train"],
        "sampling": {
            "stride_steps": int(protocol["validation_origin_stride_steps"]),
            "max_origins_per_object": args.max_origins_per_object,
        },
        "splits": split_reports,
    }
    atomic_json(args.output.resolve(), report)
    print(
        {
            "status": "complete",
            "formal_eligible": False,
            "model": model_name,
            "group": args.group,
            "checkpoint_epoch": int(payload["epoch"]),
            "one_hour_position_m": {
                split: values["horizons"][1]["position_error_mean_m"]
                for split, values in split_reports.items()
            },
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
