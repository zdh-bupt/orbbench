from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .common import GROUP_IDS, read_json, sha256, validate_protocol
from .models import TRAINED_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gated OrbBench-aligned pipeline.")
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=("sanity", "study", "train", "evaluate", "render", "all"),
        default="all",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def run(arguments: list[str]) -> None:
    print("RUN", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def verified_report(path: Path, digest: str, **expected: object) -> bool:
    if not path.exists():
        return False
    report = read_json(path)
    if report.get("protocol_sha256") != digest:
        raise RuntimeError(f"protocol mismatch in existing report: {path}")
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"{key} mismatch in existing report: {path}")
    if report.get("complete") is False:
        raise RuntimeError(f"incomplete output requires inspection: {path}")
    return True


def require_sanity(root: Path, protocol_digest: str, manifest_digest: str) -> None:
    failures: list[str] = []
    for group in GROUP_IDS:
        for model in TRAINED_MODELS:
            path = root / group / f"{model}.json"
            if not path.is_file():
                failures.append(f"missing {path}")
                continue
            report = read_json(path)
            if (
                report.get("protocol_sha256") != protocol_digest
                or report.get("manifest_sha256") != manifest_digest
                or report.get("model") != model
                or report.get("group") != group
                or report.get("quality_gate_passed") is not True
            ):
                failures.append(f"failed {path}")
    if failures:
        raise RuntimeError("sanity gate is not satisfied: " + "; ".join(failures))


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    work_root = args.work_root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = validate_protocol(read_json(protocol_path))
    protocol_digest = sha256(protocol_path)
    manifest = work_root / "data" / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"prepared data manifest is missing: {manifest}")
    manifest_digest = sha256(manifest)
    python = sys.executable
    artifacts = work_root / "artifacts"
    sanity_root = artifacts / "sanity"
    study_root = artifacts / "training_size_study"
    selection_path = artifacts / "training_size_selection.json"
    phases = (
        {args.phase}
        if args.phase != "all"
        else {"sanity", "study", "train", "evaluate", "render"}
    )
    if "sanity" in phases:
        for group in GROUP_IDS:
            for model in TRAINED_MODELS:
                report_path = sanity_root / group / f"{model}.json"
                if report_path.is_file():
                    report = read_json(report_path)
                    if (
                        report.get("protocol_sha256") == protocol_digest
                        and report.get("manifest_sha256") == manifest_digest
                        and report.get("model") == model
                        and report.get("group") == group
                        and report.get("quality_gate_passed") is True
                    ):
                        print(f"SKIP verified sanity gate {report_path}", flush=True)
                        continue
                    raise RuntimeError(f"existing sanity report failed: {report_path}")
                run(
                    [
                        python,
                        "-m",
                        "project2.sanity",
                        "--manifest",
                        str(manifest),
                        "--protocol",
                        str(protocol_path),
                        "--model",
                        model,
                        "--group",
                        group,
                        "--output",
                        str(report_path),
                        "--device",
                        args.device,
                    ]
                )
    if any(phase in phases for phase in ("study", "train", "evaluate", "render")):
        require_sanity(sanity_root, protocol_digest, manifest_digest)
    if "study" in phases:
        seed = int(protocol["training"]["seeds"][0])
        study = protocol["training_size_study"]
        for size_value in study["candidate_objects_per_group"]:
            size = int(size_value)
            for group in GROUP_IDS:
                for model in study["probe_models"]:
                    root = study_root / str(size) / group / model
                    checkpoint = root.with_suffix(".pt")
                    report = root.with_suffix(".json")
                    if verified_report(
                        report,
                        protocol_digest,
                        complete=True,
                        manifest_sha256=manifest_digest,
                        run_kind="study",
                        model=model,
                        group=group,
                        seed=seed,
                        objects=size,
                    ):
                        if sha256(checkpoint) != read_json(report)["checkpoint_sha256"]:
                            raise RuntimeError(f"study checkpoint hash mismatch: {checkpoint}")
                        print(f"SKIP verified study output {report}", flush=True)
                        continue
                    run(
                        [
                            python,
                            "-m",
                            "project2.train",
                            "--manifest",
                            str(manifest),
                            "--protocol",
                            str(protocol_path),
                            "--model",
                            model,
                            "--group",
                            group,
                            "--objects",
                            str(size),
                            "--seed",
                            str(seed),
                            "--checkpoint",
                            str(checkpoint),
                            "--report",
                            str(report),
                            "--run-kind",
                            "study",
                            "--device",
                            args.device,
                            "--num-workers",
                            str(args.num_workers),
                        ]
                    )
        if selection_path.is_file():
            selection = read_json(selection_path)
            if (
                selection.get("protocol_sha256") != protocol_digest
                or selection.get("manifest_sha256") != manifest_digest
                or not selection.get("complete")
            ):
                raise RuntimeError(f"invalid existing size selection: {selection_path}")
            print(f"SKIP verified size selection {selection_path}", flush=True)
        else:
            run(
                [
                    python,
                    "-m",
                    "project2.select_size",
                    "--protocol",
                    str(protocol_path),
                    "--manifest",
                    str(manifest),
                    "--reports-root",
                    str(study_root),
                    "--output",
                    str(selection_path),
                ]
            )
    if any(phase in phases for phase in ("train", "evaluate", "render")):
        if not selection_path.is_file():
            raise FileNotFoundError(f"training-size selection is missing: {selection_path}")
        selection = read_json(selection_path)
        if (
            selection.get("protocol_sha256") != protocol_digest
            or selection.get("manifest_sha256") != manifest_digest
            or not selection.get("complete")
        ):
            raise RuntimeError(f"invalid size selection: {selection_path}")
        selected_size = int(selection["selected_objects_per_group"])
    else:
        selected_size = -1
    if "train" in phases:
        for group in GROUP_IDS:
            for model in TRAINED_MODELS:
                for seed_value in protocol["training"]["seeds"]:
                    seed = int(seed_value)
                    root = artifacts / "trained" / group / model / f"seed_{seed}"
                    checkpoint = root.with_suffix(".pt")
                    report = root.with_suffix(".json")
                    if verified_report(
                        report,
                        protocol_digest,
                        complete=True,
                        manifest_sha256=manifest_digest,
                        quality_gate_passed=True,
                        run_kind="formal",
                        model=model,
                        group=group,
                        seed=seed,
                        objects=selected_size,
                    ):
                        if sha256(checkpoint) != read_json(report)["checkpoint_sha256"]:
                            raise RuntimeError(f"formal checkpoint hash mismatch: {checkpoint}")
                        print(f"SKIP verified formal training output {report}", flush=True)
                        continue
                    run(
                        [
                            python,
                            "-m",
                            "project2.train",
                            "--manifest",
                            str(manifest),
                            "--protocol",
                            str(protocol_path),
                            "--model",
                            model,
                            "--group",
                            group,
                            "--objects",
                            str(selected_size),
                            "--seed",
                            str(seed),
                            "--checkpoint",
                            str(checkpoint),
                            "--report",
                            str(report),
                            "--run-kind",
                            "formal",
                            "--device",
                            args.device,
                            "--num-workers",
                            str(args.num_workers),
                        ]
                    )
                    if read_json(report).get("quality_gate_passed") is not True:
                        raise RuntimeError(f"formal validation quality gate failed: {report}")
    if "evaluate" in phases:
        for model_spec in protocol["models"]:
            model = model_spec["id"]
            result = work_root / "results" / f"{model}.csv"
            provenance = work_root / "results" / f"{model}.provenance.json"
            if result.is_file() and provenance.is_file():
                metadata = read_json(provenance)
                if (
                    metadata.get("complete") is True
                    and metadata.get("quality_gate_passed") is True
                    and metadata.get("protocol_sha256") == protocol_digest
                    and metadata.get("manifest_sha256") == manifest_digest
                    and metadata.get("selection_sha256") == sha256(selection_path)
                    and metadata.get("result_csv_sha256") == sha256(result)
                ):
                    print(f"SKIP verified evaluation output {result}", flush=True)
                    continue
                raise RuntimeError(f"invalid existing evaluation output: {result}")
            if result.exists() or provenance.exists():
                raise RuntimeError(f"partial evaluation output requires inspection: {result}")
            run(
                [
                    python,
                    "-m",
                    "project2.evaluate",
                    "--manifest",
                    str(manifest),
                    "--protocol",
                    str(protocol_path),
                    "--selection",
                    str(selection_path),
                    "--model",
                    model,
                    "--artifacts-root",
                    str(artifacts),
                    "--output",
                    str(result),
                    "--provenance-output",
                    str(provenance),
                    "--device",
                    args.device,
                    "--device-map",
                    args.device_map,
                ]
            )
    if "render" in phases:
        run(
            [
                python,
                "-m",
                "project2.render_table",
                "--protocol",
                str(protocol_path),
                "--results-dir",
                str(work_root / "results"),
                "--output-dir",
                str(work_root / "formal"),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
