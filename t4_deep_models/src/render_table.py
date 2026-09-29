from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .common import (
    MODEL_LABELS,
    RESULT_FIELDS,
    atomic_json,
    horizon_steps,
    read_json,
    sha256,
    validate_protocol,
)


METRICS = (
    ("position_error_mean_m", "Position (m)"),
    ("velocity_error_mean_m_s", "Velocity (m/s)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the complete formal eight-model result table.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_complete(value: str) -> bool:
    if value.strip().lower() == "true":
        return True
    if value.strip().lower() == "false":
        return False
    raise ValueError(f"invalid complete value: {value!r}")


def validate_result_provenance(
    results_dir: Path, protocol: dict[str, Any], protocol_digest: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model_spec in protocol["models"]:
        model = model_spec["id"]
        result_path = results_dir / f"{model}.csv"
        provenance_path = results_dir / f"{model}.provenance.json"
        if not result_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(f"result or provenance is missing for {model}")
        provenance = read_json(provenance_path)
        checks = (
            provenance.get("complete") is True,
            provenance.get("quality_gate_passed") is True,
            provenance.get("protocol_sha256") == protocol_digest,
            provenance.get("model") == model,
            provenance.get("result_csv_sha256") == sha256(result_path),
            isinstance(provenance.get("manifest_sha256"), str),
            isinstance(provenance.get("selection_sha256"), str),
        )
        if not all(checks):
            raise ValueError(f"result provenance failed validation: {provenance_path}")
        records.append(
            {
                "model": model,
                "result_csv": str(result_path.resolve()),
                "result_csv_sha256": sha256(result_path),
                "provenance": str(provenance_path.resolve()),
                "provenance_sha256": sha256(provenance_path),
                "manifest_sha256": provenance["manifest_sha256"],
                "selection_sha256": provenance["selection_sha256"],
            }
        )
    manifest_digests = {record["manifest_sha256"] for record in records}
    selection_digests = {record["selection_sha256"] for record in records}
    if len(manifest_digests) != 1 or len(selection_digests) != 1:
        raise ValueError("formal model results do not share one manifest and size selection")
    return records


def load_results(
    results_dir: Path, protocol: dict[str, Any], protocol_digest: str
) -> dict[tuple[str, str, float], dict[str, Any]]:
    validate_result_provenance(results_dir, protocol, protocol_digest)
    expected_models = [item["id"] for item in protocol["models"]]
    expected_groups = [item["id"] for item in protocol["groups"]]
    expected_hours = [float(value) for value in protocol["horizons_hours"]]
    expected_steps = dict(
        zip(expected_hours, horizon_steps(expected_hours, int(protocol["sample_seconds"])))
    )
    rows: dict[tuple[str, str, float], dict[str, Any]] = {}
    for model in expected_models:
        path = results_dir / f"{model}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"formal model result is missing: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != set(RESULT_FIELDS):
                raise ValueError(f"unexpected formal result schema: {path}")
            for line_number, raw in enumerate(reader, start=2):
                if int(raw["schema_version"]) != 2:
                    raise ValueError(f"result schema version mismatch at {path}:{line_number}")
                if raw["benchmark_id"] != protocol["benchmark_id"]:
                    raise ValueError(f"benchmark mismatch at {path}:{line_number}")
                if raw["protocol_sha256"] != protocol_digest:
                    raise ValueError(f"protocol mismatch at {path}:{line_number}")
                if raw["model"] != model or raw["model_label"] != MODEL_LABELS[model]:
                    raise ValueError(f"model mismatch at {path}:{line_number}")
                if raw["group"] not in expected_groups:
                    raise ValueError(f"unknown group at {path}:{line_number}")
                expected_group_label = next(
                    item["label"] for item in protocol["groups"] if item["id"] == raw["group"]
                )
                if raw["group_label"] != expected_group_label:
                    raise ValueError(f"group label mismatch at {path}:{line_number}")
                hour = float(raw["horizon_hours"])
                if hour not in expected_hours or int(raw["horizon_steps"]) != expected_steps[hour]:
                    raise ValueError(f"horizon mismatch at {path}:{line_number}")
                if not parse_complete(raw["complete"]):
                    raise ValueError(f"incomplete row cannot enter the formal table: {path}:{line_number}")
                if not parse_complete(raw["quality_gate_passed"]):
                    raise ValueError(f"failed quality gate cannot enter the formal table: {path}:{line_number}")
                if int(raw["sample_seconds"]) != int(protocol["sample_seconds"]):
                    raise ValueError(f"sample interval mismatch at {path}:{line_number}")
                if int(raw["context_steps"]) != int(protocol["input_steps"]):
                    raise ValueError(f"context mismatch at {path}:{line_number}")
                if int(raw["n_files"]) <= 0 or int(raw["n_origins"]) <= 0:
                    raise ValueError(f"empty test row at {path}:{line_number}")
                expected_seeds = 3 if model not in {"chronos", "chronos2"} else 1
                if int(raw["n_seeds"]) != expected_seeds:
                    raise ValueError(f"seed count mismatch at {path}:{line_number}")
                for key, _ in METRICS:
                    value = float(raw[key])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid {key} at {path}:{line_number}")
                    raw[key] = value
                key = (model, raw["group"], hour)
                if key in rows:
                    raise ValueError(f"duplicate formal result row: {key}")
                rows[key] = raw
    expected_keys = {
        (model, group, hour)
        for model in expected_models
        for group in expected_groups
        for hour in expected_hours
    }
    missing = sorted(expected_keys - set(rows))
    extra = sorted(set(rows) - expected_keys)
    if missing or extra:
        raise ValueError(f"formal table coverage mismatch: missing={missing}, extra={extra}")
    for model in expected_models:
        for group in expected_groups:
            origin_counts = {
                int(rows[(model, group, hour)]["n_origins"]) for hour in expected_hours
            }
            if len(origin_counts) != 1:
                raise ValueError(f"inconsistent origin counts for {model}/{group}")
    file_counts = {int(row["n_files"]) for row in rows.values()}
    if len(file_counts) != 1:
        raise ValueError(f"formal rows disagree on cohort size: {sorted(file_counts)}")
    return rows


def format_value(value: float, metric: str) -> str:
    if metric == "position_error_mean_m":
        if value >= 1.0e7:
            return f"{value:.3e}"
        if value >= 1000:
            return f"{value:,.0f}"
        if value >= 10:
            return f"{value:.1f}"
        return f"{value:.3f}"
    if value >= 1000:
        return f"{value:,.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def ranking(
    rows: dict[tuple[str, str, float], dict[str, Any]], protocol: dict[str, Any]
) -> dict[tuple[str, str, float, str], int]:
    result: dict[tuple[str, str, float, str], int] = {}
    for group in (item["id"] for item in protocol["groups"]):
        for hour in (float(value) for value in protocol["horizons_hours"]):
            for metric, _ in METRICS:
                ordered = sorted(
                    (
                        (float(rows[(model["id"], group, hour)][metric]), model["id"])
                        for model in protocol["models"]
                    ),
                    key=lambda item: (item[0], item[1]),
                )
                for rank, (_, model) in enumerate(ordered[:2], start=1):
                    result[(model, group, hour, metric)] = rank
    return result


def matrix_rows(
    rows: dict[tuple[str, str, float], dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model in protocol["models"]:
        for metric, metric_label in METRICS:
            row: dict[str, Any] = {"Model": model["label"], "Metric": metric_label}
            for hour in protocol["horizons_hours"]:
                for group in protocol["groups"]:
                    key = f"{float(hour):g} h | {group['label']}"
                    row[key] = float(rows[(model["id"], group["id"], float(hour))][metric])
            result.append(row)
    return result


def write_csv(path: Path, matrix: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader()
        for row in matrix:
            writer.writerow(
                {
                    key: (f"{value:.12g}" if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )


def write_markdown(path: Path, matrix: list[dict[str, Any]]) -> None:
    headers = list(matrix[0])
    lines = [
        "# Formal Eight-Model Orbit Forecasting Results",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in matrix:
        metric_key = "position_error_mean_m" if row["Metric"] == "Position (m)" else "velocity_error_mean_m_s"
        values = [
            str(row[key]) if key in {"Model", "Metric"} else format_value(float(row[key]), metric_key)
            for key in headers
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Mean endpoint 3-D Euclidean error on the locked 2025 H2 test intervals. Lower is better.",
            "Supervised rows are means over three seeds; Chronos and Chronos2 are zero-shot.",
            "LEO: 160-2000 km and e < 0.1; MEO: >=2000 km and e < 0.1; HEO: e >= 0.1.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_table(
    path_png: Path,
    path_pdf: Path,
    matrix: list[dict[str, Any]],
    rows: dict[tuple[str, str, float], dict[str, Any]],
    protocol: dict[str, Any],
) -> None:
    headers = list(matrix[0])
    cell_text: list[list[str]] = []
    ranks = ranking(rows, protocol)
    metric_ids: list[str] = []
    row_models: list[str] = []
    for model in protocol["models"]:
        for metric, metric_label in METRICS:
            source = next(row for row in matrix if row["Model"] == model["label"] and row["Metric"] == metric_label)
            cell_text.append(
                [
                    str(source[key]) if key in {"Model", "Metric"} else format_value(float(source[key]), metric)
                    for key in headers
                ]
            )
            metric_ids.append(metric)
            row_models.append(model["id"])

    figure, axis = plt.subplots(figsize=(29, 10.5))
    figure.patch.set_facecolor("white")
    axis.axis("off")
    axis.set_title("Formal Eight-Model Orbit Forecasting Results", fontsize=15, fontweight="bold", pad=18)
    column_widths = [0.065, 0.075] + [0.0478] * (len(headers) - 2)
    table = axis.table(
        cellText=cell_text,
        colLabels=[value.replace(" | ", "\n") for value in headers],
        cellLoc="center",
        colLoc="center",
        colWidths=column_widths,
        bbox=[0.0, 0.09, 1.0, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.1)
    table.scale(1.0, 1.55)
    for column in range(len(headers)):
        cell = table[(0, column)]
        cell.set_facecolor("#263238")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    groups = [item["id"] for item in protocol["groups"]]
    hours = [float(value) for value in protocol["horizons_hours"]]
    for row_index, (model, metric) in enumerate(zip(row_models, metric_ids), start=1):
        shade = "#F4F6F7" if (row_index - 1) // 2 % 2 == 0 else "white"
        for column in range(len(headers)):
            cell = table[(row_index, column)]
            cell.set_facecolor(shade)
            cell.set_edgecolor("#90A4AE")
            cell.set_linewidth(0.45)
        table[(row_index, 0)].get_text().set_fontweight("bold")
        for hour_index, hour in enumerate(hours):
            for group_index, group in enumerate(groups):
                column = 2 + hour_index * len(groups) + group_index
                rank = ranks.get((model, group, hour, metric))
                if rank == 1:
                    table[(row_index, column)].set_facecolor("#DDEEDF")
                    table[(row_index, column)].get_text().set_fontweight("bold")
                elif rank == 2:
                    table[(row_index, column)].set_facecolor("#FFF1CC")
                    table[(row_index, column)].get_text().set_fontstyle("italic")
    axis.text(
        0.0,
        0.045,
        "Mean endpoint 3-D Euclidean error on locked 2025 H2 test intervals. Supervised rows average three seeds.",
        transform=axis.transAxes,
        fontsize=8,
        va="center",
    )
    axis.text(
        0.0,
        0.015,
        "LEO: 160-2000 km, e<0.1; MEO: >=2000 km, e<0.1; HEO: e>=0.1. Chronos/Chronos2: zero-shot.",
        transform=axis.transAxes,
        fontsize=8,
        va="center",
    )
    temp_png = path_png.with_name(path_png.stem + ".tmp.png")
    temp_pdf = path_pdf.with_name(path_pdf.stem + ".tmp.pdf")
    figure.savefig(temp_png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(temp_pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    os.replace(temp_png, path_png)
    os.replace(temp_pdf, path_pdf)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    outputs = {
        "png": output_dir / "formal_eight_model_table.png",
        "pdf": output_dir / "formal_eight_model_table.pdf",
        "csv": output_dir / "formal_eight_model_table.csv",
        "markdown": output_dir / "formal_eight_model_table.md",
        "manifest": output_dir / "formal_eight_model_table.manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite formal outputs: {existing}")
    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(read_json(protocol_path))
    protocol_digest = sha256(protocol_path)
    results_dir = args.results_dir.resolve()
    rows = load_results(results_dir, protocol, protocol_digest)
    provenance_records = validate_result_provenance(results_dir, protocol, protocol_digest)
    matrix = matrix_rows(rows, protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(outputs["csv"], matrix)
    write_markdown(outputs["markdown"], matrix)
    draw_table(outputs["png"], outputs["pdf"], matrix, rows, protocol)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": protocol["benchmark_id"],
        "protocol_sha256": protocol_digest,
        "manifest_sha256": provenance_records[0]["manifest_sha256"],
        "selection_sha256": provenance_records[0]["selection_sha256"],
        "status": "complete",
        "accepted_result_rows": len(rows),
        "expected_result_rows": len(protocol["models"]) * len(protocol["groups"]) * len(protocol["horizons_hours"]),
        "outputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "source_results": provenance_records,
    }
    atomic_json(outputs["manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
