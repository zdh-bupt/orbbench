from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

from .common import GROUP_IDS, atomic_json, read_json, sha256, validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze training cohort size from validation curves.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reports-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def choose_size(
    reports: dict[int, dict[str, float]],
    probe_keys: list[str],
    aggregate_tolerance: float,
    per_probe_regret: float,
) -> tuple[int, list[dict[str, Any]], bool]:
    if not reports or not probe_keys:
        raise ValueError("learning-curve reports and probe keys are required")
    if aggregate_tolerance < 0 or per_probe_regret < 0:
        raise ValueError("selection tolerances cannot be negative")
    best = {key: min(float(values[key]) for values in reports.values()) for key in probe_keys}
    rows: list[dict[str, Any]] = []
    for size in sorted(reports):
        relative = {key: float(reports[size][key]) / best[key] for key in probe_keys}
        if any(not math.isfinite(value) or value < 1.0 - 1.0e-9 for value in relative.values()):
            raise ValueError(f"invalid relative validation score at size {size}")
        rows.append(
            {
                "objects_per_group": size,
                "validation_position_error_mean_m": reports[size],
                "relative_to_probe_best": relative,
                "aggregate_relative_score": sum(relative.values()) / len(relative),
            }
        )
    best_aggregate = min(float(row["aggregate_relative_score"]) for row in rows)
    eligible = [
        row
        for row in rows
        if float(row["aggregate_relative_score"]) <= best_aggregate * (1.0 + aggregate_tolerance)
        and max(float(value) for value in row["relative_to_probe_best"].values())
        <= 1.0 + per_probe_regret
    ]
    fallback = not eligible
    if eligible:
        selected = min(int(row["objects_per_group"]) for row in eligible)
    else:
        selected = int(
            min(
                rows,
                key=lambda row: (
                    float(row["aggregate_relative_score"]), int(row["objects_per_group"])
                ),
            )["objects_per_group"]
        )
    return selected, rows, fallback


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing selection: {args.output}")
    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(read_json(protocol_path))
    protocol_digest = sha256(protocol_path)
    manifest_digest = sha256(args.manifest.resolve())
    study = protocol["training_size_study"]
    sizes = [int(value) for value in study["candidate_objects_per_group"]]
    models = [str(value) for value in study["probe_models"]]
    seed = int(protocol["training"]["seeds"][0])
    probe_keys = [f"{model}/{group}" for model in models for group in GROUP_IDS]
    values: dict[int, dict[str, float]] = {}
    sources: list[dict[str, Any]] = []
    for size in sizes:
        values[size] = {}
        for model in models:
            for group in GROUP_IDS:
                path = (args.reports_root / str(size) / group / f"{model}.json").resolve()
                report = read_json(path)
                checks = (
                    report.get("complete") is True,
                    report.get("run_kind") == "study",
                    report.get("protocol_sha256") == protocol_digest,
                    report.get("manifest_sha256") == manifest_digest,
                    report.get("model") == model,
                    report.get("group") == group,
                    int(report.get("objects", -1)) == size,
                    int(report.get("seed", -1)) == seed,
                )
                if not all(checks):
                    raise ValueError(f"invalid learning-curve report: {path}")
                score = float(report["best_validation"]["position_error_mean_m"])
                if not math.isfinite(score) or score <= 0:
                    raise ValueError(f"invalid validation position error in {path}")
                values[size][f"{model}/{group}"] = score
                sources.append({"path": str(path), "sha256": sha256(path)})
    selected, rows, fallback = choose_size(
        values,
        probe_keys,
        float(study["aggregate_relative_tolerance"]),
        float(study["per_model_max_relative_regret"]),
    )
    output = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": protocol["benchmark_id"],
        "protocol_sha256": protocol_digest,
        "manifest_sha256": manifest_digest,
        "complete": True,
        "probe_models": models,
        "probe_groups": list(GROUP_IDS),
        "study_seed": seed,
        "candidate_objects_per_group": sizes,
        "selected_objects_per_group": selected,
        "selection_rule": {
            "description": (
                "Choose the smallest size within the aggregate relative tolerance of the best "
                "validation position-error score while limiting every model/group regret."
            ),
            "aggregate_relative_tolerance": float(study["aggregate_relative_tolerance"]),
            "per_probe_max_relative_regret": float(study["per_model_max_relative_regret"]),
            "fallback_to_minimum_aggregate": fallback,
        },
        "curves": rows,
        "source_reports": sources,
    }
    atomic_json(args.output.resolve(), output)
    print({"status": "complete", "selected_objects_per_group": selected}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
