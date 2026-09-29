from __future__ import annotations

"""Generate a one-ID-per-line satellite selection file from pruned CSV names."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_REGEX = r"^(?P<satellite_id>\d+)_pruned\.csv$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build configs/satellite_ids.txt from <satellite_id>_pruned.csv files"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Pruned root or the Pruned_results directory containing range subdirectories",
    )
    parser.add_argument(
        "--output",
        default="configs/satellite_ids.txt",
        help="Output TXT selection file, one satellite ID per line",
    )
    parser.add_argument(
        "--manifest",
        default="configs/satellite_ids_manifest.json",
        help="Audit manifest JSON path; use empty string to disable",
    )
    parser.add_argument("--pattern", default="*_pruned.csv", help="Filename glob used for discovery")
    parser.add_argument(
        "--satellite-id-regex",
        default=DEFAULT_REGEX,
        help="Full filename regex with a named satellite_id group",
    )
    parser.add_argument(
        "--no-pruned-results-child",
        action="store_true",
        help="Do not automatically descend into a Pruned_results child directory",
    )
    parser.add_argument("--allow-duplicates", action="store_true", help="Keep one ID if duplicate files are found")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def resolve_search_root(data_root: str, use_pruned_results_child: bool) -> Path:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist or is not a directory: {root}")
    child = root / "Pruned_results"
    if use_pruned_results_child and child.is_dir():
        return child
    return root


def ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)


def directory_key(path: Path, search_root: Path) -> str:
    try:
        relative = path.relative_to(search_root)
    except ValueError:
        return "."
    return relative.parts[0] if len(relative.parts) > 1 else "."


def discover_ids(
    search_root: Path,
    *,
    pattern: str,
    satellite_id_regex: str,
) -> tuple[list[str], dict[str, Any]]:
    regex = re.compile(satellite_id_regex)
    id_to_paths: dict[str, list[str]] = defaultdict(list)
    nonmatching: list[str] = []
    directory_counts: dict[str, dict[str, int]] = {}
    files = sorted(path for path in search_root.rglob(pattern) if path.is_file())

    for path in files:
        key = directory_key(path, search_root)
        counts = directory_counts.setdefault(key, {"files": 0, "matched": 0, "nonmatching": 0})
        counts["files"] += 1
        match = regex.fullmatch(path.name)
        if match is None or "satellite_id" not in match.groupdict():
            counts["nonmatching"] += 1
            nonmatching.append(str(path))
            continue
        satellite_id = str(match.group("satellite_id")).strip()
        if not satellite_id:
            counts["nonmatching"] += 1
            nonmatching.append(str(path))
            continue
        counts["matched"] += 1
        id_to_paths[satellite_id].append(str(path))

    duplicates = {
        satellite_id: paths
        for satellite_id, paths in sorted(id_to_paths.items(), key=lambda item: int(item[0]))
        if len(paths) > 1
    }
    ids = sorted(id_to_paths, key=int)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_root": str(search_root),
        "pattern": pattern,
        "satellite_id_regex": satellite_id_regex,
        "discovered_file_count": len(files),
        "matched_file_count": sum(len(paths) for paths in id_to_paths.values()),
        "unique_satellite_count": len(ids),
        "duplicate_satellite_count": len(duplicates),
        "nonmatching_file_count": len(nonmatching),
        "first_satellite_id": ids[0] if ids else None,
        "last_satellite_id": ids[-1] if ids else None,
        "directories": dict(sorted(directory_counts.items())),
        "duplicates": duplicates,
        "nonmatching_files": nonmatching,
    }
    return ids, manifest


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("".join(f"{satellite_id}\n" for satellite_id in ids), encoding="ascii", newline="\n")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        search_root = resolve_search_root(args.data_root, not args.no_pruned_results_child)
        output_path = Path(args.output).resolve()
        manifest_path = Path(args.manifest).resolve() if args.manifest else None
        ensure_can_write(output_path, overwrite=args.overwrite)
        if manifest_path is not None:
            ensure_can_write(manifest_path, overwrite=args.overwrite)

        ids, manifest = discover_ids(
            search_root,
            pattern=args.pattern,
            satellite_id_regex=args.satellite_id_regex,
        )
        if not ids:
            raise ValueError(f"no satellite IDs found under {search_root}")
        if manifest["duplicate_satellite_count"] and not args.allow_duplicates:
            raise ValueError(
                f"found {manifest['duplicate_satellite_count']} duplicate satellite IDs; "
                "inspect the manifest or pass --allow-duplicates"
            )

        manifest["output"] = str(output_path)
        manifest["manifest"] = str(manifest_path) if manifest_path is not None else None
        write_ids(output_path, ids)
        if manifest_path is not None:
            write_manifest(manifest_path, manifest)

        summary = {
            "unique_satellite_count": manifest["unique_satellite_count"],
            "matched_file_count": manifest["matched_file_count"],
            "duplicate_satellite_count": manifest["duplicate_satellite_count"],
            "nonmatching_file_count": manifest["nonmatching_file_count"],
            "first_satellite_id": manifest["first_satellite_id"],
            "last_satellite_id": manifest["last_satellite_id"],
            "output": str(output_path),
            "manifest": str(manifest_path) if manifest_path is not None else None,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
