from __future__ import annotations

"""Extract the +1 h slice from a formal density evaluation report.

The script deliberately keeps report completeness, checkpoint eligibility, and
deployment acceptance separate from the metric values.  It never upgrades a
debug, validation-only, or incomplete report to a formal benchmark result.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HORIZON_HOURS = 1
METHODS = (
    ("learned", "PhysicsResidualUNetConvGRU3D"),
    ("physics", "Causal two-body Kepler + CIC"),
    ("persistence", "Last-frame persistence"),
)
METRICS = (
    "tv",
    "density_mass_overlap_percent",
    "regional_relative_mae",
    "neighbor_hotspot_f1",
    "conservation_error",
    "interval_90_coverage",
    "hotspot_ece",
)
OUTPUT_NAMES = (
    "one_hour_evaluation.csv",
    "one_hour_evaluation.json",
    "one_hour_evaluation.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a provenance-preserving +1 h density benchmark table."
    )
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--heatmap-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/benchmark/one_hour"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path, description: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain a JSON object: {resolved}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def validate_decision(
    decision: dict[str, Any] | None,
    evaluation: dict[str, Any],
) -> tuple[bool | None, str | None, list[str]]:
    notes: list[str] = []
    if decision is None:
        notes.append("deployment_decision.json was not supplied")
        return None, None, notes

    accepted = decision.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError("deployment decision field 'accepted' must be boolean")
    selected_model = decision.get("selected_model")
    if not isinstance(selected_model, str) or not selected_model:
        raise ValueError("deployment decision field 'selected_model' must be a nonempty string")

    report_accepted = evaluation.get("acceptance", {}).get("accepted")
    report_selected = evaluation.get("selected_model")
    if report_accepted is not None and bool(report_accepted) != accepted:
        raise ValueError("evaluation and deployment decision disagree on acceptance")
    if report_selected is not None and str(report_selected) != selected_model:
        raise ValueError("evaluation and deployment decision disagree on selected_model")
    return accepted, selected_model, notes


def extract_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("evaluation report is missing a metrics object")

    rows: list[dict[str, Any]] = []
    for method_key, method_label in METHODS:
        method_metrics = metrics.get(method_key)
        if not isinstance(method_metrics, dict):
            raise ValueError(f"evaluation metrics are missing method {method_key!r}")
        horizon_metrics = method_metrics.get(str(HORIZON_HOURS))
        if not isinstance(horizon_metrics, dict):
            raise ValueError(f"evaluation metrics are missing the {HORIZON_HOURS} h slice for {method_key}")

        row: dict[str, Any] = {
            "method_key": method_key,
            "method": method_label,
            "horizon_hours": HORIZON_HOURS,
        }
        for key in METRICS:
            if key == "density_mass_overlap_percent":
                tv = finite_number(horizon_metrics.get("tv"), f"{method_key}.1.tv")
                row[key] = 100.0 * (1.0 - tv)
            elif key in horizon_metrics:
                row[key] = finite_number(
                    horizon_metrics[key], f"{method_key}.1.{key}"
                )
            else:
                row[key] = None
        rows.append(row)
    return rows


def completeness_status(evaluation: dict[str, Any]) -> dict[str, Any]:
    evaluated_split = evaluation.get("evaluated_split")
    evaluation_complete = evaluation.get("evaluation", {}).get("complete") is True
    calibration_complete = evaluation.get("calibration", {}).get("complete") is True
    formal_checkpoint = evaluation.get("acceptance", {}).get("formal_checkpoint") is True
    formally_reportable = bool(
        evaluated_split == "test"
        and evaluation_complete
        and calibration_complete
        and formal_checkpoint
    )
    reasons: list[str] = []
    if evaluated_split != "test":
        reasons.append(f"evaluated_split is {evaluated_split!r}, not 'test'")
    if not evaluation_complete:
        reasons.append("evaluation coverage is incomplete or unrecorded")
    if not calibration_complete:
        reasons.append("calibration coverage is incomplete or unrecorded")
    if not formal_checkpoint:
        reasons.append("checkpoint is not formally eligible")
    return {
        "evaluated_split": evaluated_split,
        "evaluation_complete": evaluation_complete,
        "calibration_complete": calibration_complete,
        "formal_checkpoint": formal_checkpoint,
        "formally_reportable": formally_reportable,
        "nonformal_reasons": reasons,
    }


def validate_heatmap_report(
    heatmap: dict[str, Any] | None,
    selected_model: str | None,
) -> dict[str, Any] | None:
    if heatmap is None:
        return None
    selection = heatmap.get("selection")
    population = heatmap.get("population")
    if not isinstance(selection, dict) or not isinstance(population, dict):
        raise ValueError("heatmap report is missing selection/population provenance")
    if int(selection.get("horizon_hours", -1)) != HORIZON_HOURS:
        raise ValueError("heatmap report is not a +1 h result")
    if population.get("matched") is not True:
        raise ValueError("heatmap report does not confirm a matched closed cohort")
    heatmap_model = str(selection.get("selected_model", ""))
    return {
        "status": heatmap.get("status"),
        "origin_time_utc": selection.get("origin_time_utc"),
        "target_time_utc": selection.get("target_time_utc"),
        "selected_model": heatmap_model,
        "same_as_deployment_selection": (
            selected_model is not None and heatmap_model == selected_model
        ),
        "satellite_count": population.get("satellite_count"),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_path = args.evaluation.resolve()
    evaluation = read_json(evaluation_path, "evaluation report")
    decision_path = args.decision.resolve() if args.decision else None
    decision = read_json(decision_path, "deployment decision") if decision_path else None
    heatmap_path = args.heatmap_report.resolve() if args.heatmap_report else None
    heatmap = read_json(heatmap_path, "heatmap report") if heatmap_path else None

    rows = extract_rows(evaluation)
    status = completeness_status(evaluation)
    accepted, selected_model, notes = validate_decision(decision, evaluation)
    heatmap_summary = validate_heatmap_report(heatmap, selected_model)

    if accepted is None:
        status["accepted"] = evaluation.get("acceptance", {}).get("accepted")
        status["selected_model"] = evaluation.get("selected_model")
    else:
        status["accepted"] = accepted
        status["selected_model"] = selected_model

    sources: dict[str, Any] = {
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": sha256(evaluation_path),
        }
    }
    if decision_path is not None:
        sources["decision"] = {
            "path": str(decision_path),
            "sha256": sha256(decision_path),
        }
    if heatmap_path is not None:
        sources["heatmap_report"] = {
            "path": str(heatmap_path),
            "sha256": sha256(heatmap_path),
        }

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_hours": HORIZON_HOURS,
        "status": status,
        "rows": rows,
        "heatmap": heatmap_summary,
        "sources": sources,
        "notes": notes,
        "interpretation_boundary": (
            "Metrics are reportable as a formal locked-test benchmark only when "
            "status.formally_reportable is true. Density overlap is 100*(1-TV) "
            "for CIC mass, not per-satellite exact-grid accuracy."
        ),
    }


def format_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown(report: dict[str, Any]) -> str:
    status = report["status"]
    lines = [
        "# One-hour voxel-density evaluation",
        "",
        f"- Formal locked-test reportable: `{str(status['formally_reportable']).lower()}`",
        f"- Accepted: `{str(status.get('accepted')).lower()}`",
        f"- Selected model: `{status.get('selected_model')}`",
        "",
        "| Method | TV | CIC mass overlap (%) | Regional relative MAE | Neighbor hotspot F1 | Conservation error | 90% coverage | Hotspot ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {method} | {tv} | {overlap} | {regional} | {f1} | {mass} | {coverage} | {ece} |".format(
                method=row["method"],
                tv=format_value(row["tv"]),
                overlap=format_value(row["density_mass_overlap_percent"], 3),
                regional=format_value(row["regional_relative_mae"]),
                f1=format_value(row["neighbor_hotspot_f1"]),
                mass=format_value(row["conservation_error"], 9),
                coverage=format_value(row["interval_90_coverage"]),
                ece=format_value(row["hotspot_ece"]),
            )
        )
    lines.extend(
        [
            "",
            "> CIC mass overlap is `100 * (1 - TV)`. It is not per-satellite exact-grid accuracy.",
            "",
        ]
    )
    if status["nonformal_reasons"]:
        lines.append("## Nonformal reasons")
        lines.append("")
        lines.extend(f"- {reason}" for reason in status["nonformal_reasons"])
        lines.append("")
    return "\n".join(lines)


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    paths = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace them: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def write_outputs(report: dict[str, Any], paths: dict[str, Path]) -> None:
    csv_path = paths["one_hour_evaluation.csv"]
    fieldnames = ["method_key", "method", "horizon_hours", *METRICS]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])
    paths["one_hour_evaluation.json"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["one_hour_evaluation.md"].write_text(markdown(report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    paths = prepare_outputs(args.output_dir.resolve(), args.overwrite)
    write_outputs(report, paths)
    print(json.dumps({"status": report["status"], "outputs": {k: str(v) for k, v in paths.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
