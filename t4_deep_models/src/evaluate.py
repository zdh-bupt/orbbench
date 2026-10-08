from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from .common import (
    GROUP_IDS,
    MODEL_LABELS,
    RESULT_FIELDS,
    atomic_json,
    read_json,
    sha256,
    validate_protocol,
)
from .data import iter_evaluation_batches, ready_entries, select_entries
from .models import TRAINED_MODELS, build_model
from .representation import decode_states, encode_states
from .train import resolve_device, seed_everything


class Adapter(Protocol):
    metadata: dict[str, Any]

    def predict_targets(self, contexts: np.ndarray, target_steps: list[int]) -> np.ndarray:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one method on the locked test cohort.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_LABELS))
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance-output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-files-per-group", type=int, help="Debug only; makes results incomplete.")
    parser.add_argument("--max-origins-per-file", type=int, help="Debug only; makes results incomplete.")
    return parser.parse_args()


def torch_load(path: Path, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class TrainedAdapter:
    def __init__(
        self,
        model_name: str,
        group: str,
        seed: int,
        objects: int,
        checkpoint: Path,
        report_path: Path,
        protocol_digest: str,
        manifest_digest: str,
        device: torch.device,
    ) -> None:
        report = read_json(report_path)
        checks = (
            report.get("complete") is True,
            report.get("quality_gate_passed") is True,
            report.get("protocol_sha256") == protocol_digest,
            report.get("manifest_sha256") == manifest_digest,
            report.get("model") == model_name,
            report.get("group") == group,
            int(report.get("seed", -1)) == seed,
            int(report.get("objects", -1)) == objects,
            report.get("checkpoint_sha256") == sha256(checkpoint),
        )
        if not all(checks):
            raise ValueError(f"checkpoint report failed provenance or quality checks: {report_path}")
        payload = torch_load(checkpoint, device)
        if (
            payload.get("protocol_sha256") != protocol_digest
            or payload.get("manifest_sha256") != manifest_digest
            or payload.get("model") != model_name
            or payload.get("group") != group
            or int(payload.get("seed", -1)) != seed
            or int(payload.get("objects", -1)) != objects
        ):
            raise ValueError(f"checkpoint metadata mismatch: {checkpoint}")
        config = payload.get("model_config")
        if not isinstance(config, dict):
            raise ValueError(f"checkpoint has no model configuration: {checkpoint}")
        self.model = build_model(model_name, config).to(device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.device = device
        self.metadata = {
            "backend": "project2_pytorch_rtn",
            "group": group,
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "training_report": str(report_path.resolve()),
            "training_report_sha256": sha256(report_path),
        }

    def predict_targets(self, contexts: np.ndarray, target_steps: list[int]) -> np.ndarray:
        origins = contexts[:, -1]
        encoded = encode_states(contexts, origins)
        with torch.inference_mode():
            prediction = self.model(
                torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
            ).detach().cpu().numpy()
        if prediction.shape != (len(contexts), len(target_steps), 6):
            raise ValueError(f"trained model produced unexpected shape {prediction.shape}")
        return decode_states(prediction, origins)


def exact_cadence(target_step: int, maximum_prediction_length: int = 60) -> int:
    first = max(1, math.ceil(target_step / maximum_prediction_length))
    for cadence in range(first, target_step + 1):
        if target_step % cadence == 0:
            return cadence
    raise AssertionError("every positive integer is divisible by itself")


class ChronosAdapter:
    def __init__(
        self,
        model_name: str,
        model_id: str,
        batch_size: int,
        device_map: str,
        num_samples: int,
        quantile: float,
    ) -> None:
        try:
            from chronos import Chronos2Pipeline, ChronosPipeline
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("install chronos-forecasting==2.2.2") from exc
        pipeline_class = ChronosPipeline if model_name == "chronos" else Chronos2Pipeline
        self.pipeline = pipeline_class.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.quantile = quantile
        config = getattr(getattr(self.pipeline, "inner_model", None), "config", None)
        self.metadata = {
            "backend": model_name,
            "model_id": model_id,
            "resolved_revision": getattr(config, "_commit_hash", None),
            "point_forecast": "sample_median" if model_name == "chronos" else f"quantile_{quantile:g}",
            "horizon_adaptation": "exact-endpoint direct forecast with largest cadence giving <=60 steps",
            "context_normalization": "per_origin_per_channel_zscore",
            "inference_dtype": "bfloat16",
            "num_samples": num_samples if model_name == "chronos" else None,
        }

    def _chronos_endpoint(self, history: np.ndarray, prediction_length: int) -> np.ndarray:
        batch, channels, _ = history.shape
        flat = history.reshape(batch * channels, -1)
        samples = self.pipeline.predict(
            torch.as_tensor(flat, dtype=torch.float32),
            prediction_length=prediction_length,
            num_samples=self.num_samples,
        )
        if isinstance(samples, list):
            samples = torch.stack(samples)
        point = samples.median(dim=1).values[:, -1].detach().cpu().numpy()
        return point.reshape(batch, channels)

    def _chronos2_endpoint(self, history: np.ndarray, prediction_length: int) -> np.ndarray:
        predictions = self.pipeline.predict(
            history,
            prediction_length=prediction_length,
            context_length=history.shape[-1],
            batch_size=self.batch_size,
            cross_learning=False,
            unrolled_quantiles=[self.quantile],
        )
        supported = [float(value) for value in self.pipeline.quantiles]
        quantile_index = min(
            range(len(supported)), key=lambda index: abs(supported[index] - self.quantile)
        )
        rows: list[np.ndarray] = []
        for prediction in predictions:
            array = prediction.detach().cpu().numpy()
            if array.ndim != 3 or array.shape[0] != 6:
                raise ValueError(f"Chronos2 prediction has unexpected shape {array.shape}")
            rows.append(array[:, quantile_index, -1])
        return np.stack(rows)

    def predict_targets(self, contexts: np.ndarray, target_steps: list[int]) -> np.ndarray:
        origins = contexts[:, -1]
        encoded = encode_states(contexts, origins).transpose(0, 2, 1)
        mean = encoded.mean(axis=2, keepdims=True)
        std = encoded.std(axis=2, keepdims=True)
        std = np.where(std < 1.0e-8, 1.0, std)
        normalized = (encoded - mean) / std
        endpoints: list[np.ndarray] = []
        cadences: dict[int, int] = {}
        for target_step in target_steps:
            cadence = exact_cadence(target_step)
            cadences[target_step] = cadence
            history = normalized[:, :, ::-cadence][:, :, ::-1]
            prediction_length = target_step // cadence
            if self.model_name == "chronos":
                endpoint = self._chronos_endpoint(history, prediction_length)
            else:
                endpoint = self._chronos2_endpoint(history, prediction_length)
            if not np.isfinite(endpoint).all():
                raise FloatingPointError("foundation model produced non-finite predictions")
            endpoints.append(endpoint)
        self.metadata["cadence_minutes_by_target_step"] = cadences
        prediction = np.stack(endpoints, axis=1)
        prediction = prediction * std.transpose(0, 2, 1) + mean.transpose(0, 2, 1)
        return decode_states(prediction, origins)


def expected_origins(entries: list[dict[str, Any]], protocol: dict[str, Any]) -> int:
    from .data import origins_for_rows

    return sum(
        len(
            origins_for_rows(
                int(entry["rows"]),
                int(protocol["input_steps"]),
                int(protocol["target_steps"][-1]),
                int(protocol["test_origin_stride_steps"]),
            )
        )
        for entry in entries
    )


def evaluate_adapter(
    adapter: Adapter,
    entries: list[dict[str, Any]],
    protocol: dict[str, Any],
    batch_size: int,
    max_origins_per_file: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    target_steps = [int(value) for value in protocol["target_steps"]]
    position_sums = np.zeros(len(target_steps), dtype=np.float64)
    velocity_sums = np.zeros(len(target_steps), dtype=np.float64)
    count = 0
    for batch_index, (contexts, truth) in enumerate(
        iter_evaluation_batches(
            entries,
            int(protocol["input_steps"]),
            target_steps,
            int(protocol["test_origin_stride_steps"]),
            batch_size,
            max_origins_per_file,
        ),
        start=1,
    ):
        prediction = adapter.predict_targets(contexts, target_steps)
        if prediction.shape != truth.shape:
            raise ValueError(f"prediction has unexpected shape {prediction.shape}")
        difference = prediction.astype(np.float64) - truth.astype(np.float64)
        position_sums += np.linalg.norm(difference[:, :, :3], axis=2).sum(axis=0)
        velocity_sums += np.linalg.norm(difference[:, :, 3:], axis=2).sum(axis=0)
        count += len(contexts)
        if batch_index % 25 == 0:
            print(f"evaluation_batch={batch_index} origins={count}", flush=True)
    if count == 0:
        raise RuntimeError("evaluation produced no origins")
    return position_sums / count, velocity_sums / count, count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    for output in (args.output, args.provenance_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(read_json(protocol_path))
    protocol_digest = sha256(protocol_path)
    manifest_path = args.manifest.resolve()
    manifest_digest = sha256(manifest_path)
    selection = read_json(args.selection.resolve())
    if (
        not selection.get("complete")
        or selection.get("protocol_sha256") != protocol_digest
        or selection.get("manifest_sha256") != manifest_digest
    ):
        raise ValueError("training-size selection is incomplete or belongs to another protocol")
    selected_size = int(selection["selected_objects_per_group"])
    batch_size = int(args.batch_size or protocol["evaluation"]["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    seed_everything(int(protocol["training"]["seeds"][0]))
    device = resolve_device(args.device)
    model_spec = next(item for item in protocol["models"] if item["id"] == args.model)
    complete = args.max_files_per_group is None and args.max_origins_per_file is None
    group_labels = {item["id"]: item["label"] for item in protocol["groups"]}
    rows: list[dict[str, Any]] = []
    adapters_metadata: dict[str, Any] = {}
    group_counts: dict[str, int] = {}
    for group in GROUP_IDS:
        entries = select_entries(ready_entries(manifest_path, "test", group), selected_size)
        if args.max_files_per_group is not None:
            if args.max_files_per_group <= 0:
                raise ValueError("max-files-per-group must be positive")
            entries = entries[: args.max_files_per_group]
        position_runs: list[np.ndarray] = []
        velocity_runs: list[np.ndarray] = []
        counts: list[int] = []
        if args.model in TRAINED_MODELS:
            metadata: list[dict[str, Any]] = []
            for seed in (int(value) for value in protocol["training"]["seeds"]):
                root = args.artifacts_root.resolve() / "trained" / group / args.model / f"seed_{seed}"
                adapter: Adapter = TrainedAdapter(
                    args.model,
                    group,
                    seed,
                    selected_size,
                    root.with_suffix(".pt"),
                    root.with_suffix(".json"),
                    protocol_digest,
                    manifest_digest,
                    device,
                )
                position, velocity, count = evaluate_adapter(
                    adapter, entries, protocol, batch_size, args.max_origins_per_file
                )
                position_runs.append(position)
                velocity_runs.append(velocity)
                counts.append(count)
                metadata.append(adapter.metadata)
            adapters_metadata[group] = metadata
            n_seeds = len(position_runs)
        else:
            adapter = ChronosAdapter(
                args.model,
                str(model_spec["model_id"]),
                batch_size,
                args.device_map,
                int(protocol["evaluation"]["chronos_num_samples"]),
                float(protocol["evaluation"]["chronos2_quantile"]),
            )
            position, velocity, count = evaluate_adapter(
                adapter, entries, protocol, batch_size, args.max_origins_per_file
            )
            position_runs.append(position)
            velocity_runs.append(velocity)
            counts.append(count)
            adapters_metadata[group] = adapter.metadata
            n_seeds = 1
        if len(set(counts)) != 1:
            raise RuntimeError(f"seed origin counts differ for {args.model}/{group}: {counts}")
        count = counts[0]
        if complete and count != expected_origins(entries, protocol):
            raise RuntimeError(f"origin count mismatch for {args.model}/{group}")
        group_counts[group] = count
        position_mean = np.stack(position_runs).mean(axis=0)
        velocity_mean = np.stack(velocity_runs).mean(axis=0)
        if not np.isfinite(position_mean).all() or not np.isfinite(velocity_mean).all():
            raise FloatingPointError("aggregate metrics are non-finite")
        for index, hour in enumerate(protocol["horizons_hours"]):
            rows.append(
                {
                    "schema_version": 2,
                    "benchmark_id": protocol["benchmark_id"],
                    "protocol_sha256": protocol_digest,
                    "model": args.model,
                    "model_label": MODEL_LABELS[args.model],
                    "group": group,
                    "group_label": group_labels[group],
                    "horizon_hours": f"{float(hour):g}",
                    "horizon_steps": int(protocol["target_steps"][index]),
                    "sample_seconds": int(protocol["sample_seconds"]),
                    "context_steps": int(protocol["input_steps"]),
                    "complete": str(complete).lower(),
                    "quality_gate_passed": "true",
                    "n_files": len(entries),
                    "n_origins": count,
                    "n_seeds": n_seeds,
                    "position_error_mean_m": f"{position_mean[index]:.12g}",
                    "velocity_error_mean_m_s": f"{velocity_mean[index]:.12g}",
                }
            )
        print({"model": args.model, "group": group, "origins": count, "seeds": n_seeds}, flush=True)
    write_csv(args.output.resolve(), rows)
    provenance = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": protocol["benchmark_id"],
        "protocol_sha256": protocol_digest,
        "manifest_sha256": manifest_digest,
        "selection_sha256": sha256(args.selection.resolve()),
        "complete": complete,
        "quality_gate_passed": True,
        "model": args.model,
        "adapters": adapters_metadata,
        "inference_batch_size": batch_size,
        "selected_objects_per_group": selected_size,
        "group_origin_counts": group_counts,
        "result_csv": str(args.output.resolve()),
        "result_csv_sha256": sha256(args.output.resolve()),
        "debug_limits": {
            "max_files_per_group": args.max_files_per_group,
            "max_origins_per_file": args.max_origins_per_file,
        },
    }
    atomic_json(args.provenance_output.resolve(), provenance)
    print({"status": "complete" if complete else "debug_incomplete", "output": str(args.output)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
