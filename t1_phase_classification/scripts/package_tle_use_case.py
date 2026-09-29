#!/usr/bin/env python3
"""Package the TLE observable-state paper use case without data or model weights."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "tle_observable_state_use_case_20260908"

ROOT_FILES = [
    "STARLINK_MOTION_PHASE_TEST_CASE.md",
    "pyproject.toml",
]
PIPELINE_FILES = [
    "scripts/build_tle_observable_feature_labels_v2.py",
    "scripts/retest_feature_label_clustering_v2.py",
    "scripts/analyze_feature_label_clustering_v2.py",
    "scripts/plot_tle_observable_state_division.py",
    "scripts/run_frozen_observable_final_test.py",
    "scripts/package_tle_use_case.py",
    "satlife_observable/__init__.py",
    "satlife_observable/data.py",
    "satlife_observable/metrics.py",
    "satlife_observable/model.py",
    "satlife_observable/final_test.py",
    "tests/test_observable_metrics.py",
    "tests/test_tle_observable_state_division.py",
]


README = """# TLE Observable-State Use Case Package

This directory collects the publication figures, retrospective clustering
results, and the code involved in the TLE observable-state use case.

## Contents

- `figures/tle_state_use_case/`: three final figures in PDF, SVG, and PNG,
  their plotting scripts, the Markdown/LaTeX result tables, and small figure
  configuration CSV files.
- `reports/observable_regime_feature_labels_v2/retest/`: aggregate and
  per-seed evaluation outputs. Row-level `predictions.csv` files are excluded.
- `reports/observable_regime_feature_labels_v2/retest.log`: execution log.
- `scripts/`, `satlife_observable/`, and `tests/`: label-rule, retrospective
  evaluation, plotting, core metric/model, and test code.
- `STARLINK_MOTION_PHASE_TEST_CASE.md`: the observable-state test standard.

## Scope and interpretation

The state rules relabel TLE-observable trajectory shapes inside inherited
legacy candidate segments. They do not perform fully legacy-independent,
end-to-end segmentation. The reported test scores are a retrospective
evaluation after label reorganization, not a new unseen frozen final test.
The labels describe observable trajectory morphology and must not be
interpreted as operational intent or the cause of a maneuver.

As requested, this package contains no label dataset, raw/derived TLE dataset,
row-level prediction file, or model weight/artifact (`.pt`, `.joblib`, `.npz`,
`.npy`). Those external inputs are required to rerun label construction or
model inference. Figure 3 and the result tables can be regenerated from the
packaged aggregate result CSV files; Figures 1 and 2 require the omitted
feature-label dataset.

## Main result files

- `figures/tle_state_use_case/RESULTS_TABLE.md`
- `figures/tle_state_use_case/TABLE_clustering_results.tex`
- `reports/observable_regime_feature_labels_v2/retest/RESULTS.md`
- `reports/observable_regime_feature_labels_v2/retest/summary_by_system.csv`
- `reports/observable_regime_feature_labels_v2/retest/per_class_summary.csv`

The recorded retrospective run used Python 3.13.11, PyTorch 2.11.0+cu128,
NumPy 2.5.2, pandas 3.0.5, and scikit-learn 1.9.0. Plotting also requires
Matplotlib; evaluation imports SciPy, joblib, and ruptures.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=PROJECT_ROOT / "deliverables",
        help="Directory that will contain the package directory and ZIP.",
    )
    parser.add_argument("--name", default=DEFAULT_NAME)
    return parser.parse_args()


def copy_file(relative_path: str, package_dir: Path) -> None:
    source = PROJECT_ROOT / relative_path
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = package_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_figure_bundle(package_dir: Path) -> None:
    source_root = PROJECT_ROOT / "figures/tle_state_use_case"
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.suffix in {".pyc", ".pyo"}:
            continue
        if "__pycache__" in source.parts:
            continue
        relative = source.relative_to(PROJECT_ROOT)
        copy_file(relative.as_posix(), package_dir)


def copy_filtered_results(package_dir: Path) -> None:
    source_root = PROJECT_ROOT / "reports/observable_regime_feature_labels_v2/retest"
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.name == "predictions.csv":
            continue
        relative = source.relative_to(PROJECT_ROOT)
        copy_file(relative.as_posix(), package_dir)
    copy_file("reports/observable_regime_feature_labels_v2/retest.log", package_dir)


def assert_scope(package_dir: Path) -> None:
    forbidden_suffixes = {".pt", ".pth", ".joblib", ".npz", ".npy", ".pyc", ".pyo"}
    violations = []
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in forbidden_suffixes or path.name == "predictions.csv":
            violations.append(path.relative_to(package_dir).as_posix())
        if "output" in path.relative_to(package_dir).parts:
            violations.append(path.relative_to(package_dir).as_posix())
    if violations:
        raise RuntimeError("Excluded data/model files entered package: " + ", ".join(violations))


def write_manifest(package_dir: Path) -> None:
    files = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(package_dir).as_posix(),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "package": package_dir.name,
        "file_count_excluding_manifest": len(files),
        "total_bytes_excluding_manifest": sum(item["bytes"] for item in files),
        "exclusions": [
            "label and TLE datasets",
            "row-level predictions.csv",
            "model weights and fitted artifacts",
            "Python caches",
        ],
        "files": files,
    }
    (package_dir / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_zip(package_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source in sorted(package_dir.rglob("*")):
            if source.is_file():
                archive.write(source, Path(package_dir.name) / source.relative_to(package_dir))


def verify_zip(package_dir: Path, archive_path: Path) -> None:
    expected = {
        (Path(package_dir.name) / path.relative_to(package_dir)).as_posix(): path.stat().st_size
        for path in package_dir.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(archive_path) as archive:
        actual = {info.filename: info.file_size for info in archive.infolist() if not info.is_dir()}
        if actual != expected:
            raise RuntimeError("ZIP content or file sizes do not match the package directory")
        bad_entries = archive.testzip()
        if bad_entries is not None:
            raise RuntimeError(f"Corrupt ZIP entry: {bad_entries}")


def main() -> None:
    args = parse_args()
    destination_root = args.destination_root.resolve()
    package_dir = destination_root / args.name
    archive_path = destination_root / f"{args.name}.zip"
    if package_dir.exists() or archive_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing package path: {package_dir} or {archive_path}"
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir()
    try:
        for relative_path in ROOT_FILES + PIPELINE_FILES:
            copy_file(relative_path, package_dir)
        copy_figure_bundle(package_dir)
        copy_filtered_results(package_dir)
        (package_dir / "README.md").write_text(README, encoding="utf-8")
        assert_scope(package_dir)
        write_manifest(package_dir)
        write_zip(package_dir, archive_path)
        verify_zip(package_dir, archive_path)
    except Exception:
        if archive_path.exists():
            archive_path.unlink()
        shutil.rmtree(package_dir, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "package_dir": str(package_dir),
                "archive": str(archive_path),
                "file_count": sum(1 for path in package_dir.rglob("*") if path.is_file()),
                "package_bytes": sum(
                    path.stat().st_size for path in package_dir.rglob("*") if path.is_file()
                ),
                "archive_bytes": archive_path.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
