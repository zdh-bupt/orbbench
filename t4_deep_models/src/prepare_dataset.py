from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from .common import (
    CSV_COLUMNS,
    TARGET_COLUMNS,
    atomic_json,
    parse_utc_datetime,
    read_json,
    sha256,
    stable_rank,
    validate_protocol,
)
from .orbit import orbit_group, orbital_elements


STRICT_NAME = re.compile(r"^(?P<norad>[0-9]{5})_pruned\.csv$")
ONE_MINUTE_NS = 60_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and extract the locked Project 2 dataset.")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def parse_csv_line(line: bytes) -> list[str]:
    values = next(csv.reader([line.decode("utf-8").strip()]))
    if len(values) != len(CSV_COLUMNS):
        raise ValueError(f"expected {len(CSV_COLUMNS)} columns, found {len(values)}")
    return values


def tail_lines(stream: BinaryIO, block_size: int = 65536) -> list[bytes]:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - block_size))
    return stream.read().splitlines()


def read_state_at(path: Path, target: datetime) -> tuple[datetime, list[float]]:
    offset = find_start_offset(path, target)
    with path.open("rb") as stream:
        stream.seek(offset)
        values = parse_csv_line(stream.readline())
    timestamp = parse_utc_datetime(values[0])
    state = [float(value) for value in values[1:]]
    return timestamp, state


def classification_at(path: Path, target: datetime) -> dict[str, Any] | None:
    timestamp, state = read_state_at(path, target)
    if not target <= timestamp < target + timedelta(seconds=60):
        return None
    if not np.isfinite(state).all():
        raise ValueError(f"non-finite state at classification reference: {path}")
    altitude_km, eccentricity = orbital_elements(state)
    group = orbit_group(altitude_km, eccentricity)
    if group is None:
        return None
    return {
        "timestamp": timestamp.isoformat(),
        "semi_major_axis_altitude_km": altitude_km,
        "eccentricity": eccentricity,
        "group": group,
    }


def scan_candidate(
    path: Path,
    coverage_start: datetime,
    coverage_end: datetime,
    group_references: list[datetime],
) -> dict[str, Any] | None:
    match = STRICT_NAME.fullmatch(path.name)
    if match is None:
        return None
    with path.open("rb") as stream:
        header = tuple(stream.readline().decode("utf-8-sig").strip().split(","))
        if header != CSV_COLUMNS:
            raise ValueError(f"unexpected schema in {path}: {header}")
        first_line = stream.readline()
        second_line = stream.readline()
        lines = tail_lines(stream)
    if not first_line or not second_line:
        return None
    first = parse_csv_line(first_line)
    second = parse_csv_line(second_line)
    previous = parse_csv_line(lines[-2])
    last = parse_csv_line(lines[-1])
    first_time = parse_utc_datetime(first[0])
    second_time = parse_utc_datetime(second[0])
    previous_time = parse_utc_datetime(previous[0])
    last_time = parse_utc_datetime(last[0])
    if (second_time - first_time).total_seconds() != 60.0:
        raise ValueError(f"first records are not 60 s apart: {path}")
    if (last_time - previous_time).total_seconds() != 60.0:
        raise ValueError(f"last records are not 60 s apart: {path}")
    required_last = coverage_end - timedelta(seconds=60)
    if first_time > coverage_start or last_time < required_last:
        return None
    classifications = [classification_at(path, reference) for reference in group_references]
    if any(item is None for item in classifications):
        return None
    stable = [item for item in classifications if item is not None]
    if len({item["group"] for item in stable}) != 1:
        return None
    classification = stable[0]
    stat = path.stat()
    return {
        "norad_id": match.group("norad"),
        "source_path": str(path.resolve()),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "first_time": first_time.isoformat(),
        "last_time": last_time.isoformat(),
        "group_reference_utc": classification["timestamp"],
        "semi_major_axis_altitude_km": classification["semi_major_axis_altitude_km"],
        "eccentricity": classification["eccentricity"],
        "group": classification["group"],
        "split_classifications": {
            reference.isoformat(): item for reference, item in zip(group_references, stable)
        },
    }


def select_entries(
    raw_root: Path,
    protocol: dict[str, Any],
    work_root: Path,
    workers: int,
) -> dict[str, Any]:
    split_specs = protocol["splits"]
    coverage_start = min(parse_utc_datetime(spec["start_utc"]) for spec in split_specs.values())
    coverage_end = max(parse_utc_datetime(spec["end_utc"]) for spec in split_specs.values())
    group_reference = parse_utc_datetime(protocol["group_reference_utc"])
    if not coverage_start <= group_reference < coverage_end:
        raise ValueError("group_reference_utc must lie inside the benchmark coverage")
    group_references = [
        parse_utc_datetime(protocol["splits"][split]["start_utc"])
        for split in ("train", "val", "test")
    ]
    seed = int(protocol["training"]["seeds"][0])
    candidates: dict[str, list[dict[str, Any]]] = {item["id"]: [] for item in protocol["groups"]}
    paths = sorted(raw_root.rglob("*_pruned.csv"))
    scanned = len(paths)
    short_or_excluded = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                scan_candidate,
                path,
                coverage_start,
                coverage_end,
                group_references,
            ): path
            for path in paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                raise RuntimeError(f"candidate scan failed for {path}") from exc
            completed += 1
            if item is None:
                short_or_excluded += 1
            else:
                candidates[item["group"]].append(item)
            if completed % 250 == 0 or completed == scanned:
                counts = {group: len(values) for group, values in candidates.items()}
                print(f"candidate_scan={completed}/{scanned} eligible_so_far={counts}", flush=True)

    entries: list[dict[str, Any]] = []
    cohort_size = int(protocol["cohort"]["prepared_objects_per_group"])
    for group, values in candidates.items():
        values.sort(key=lambda item: stable_rank(seed, group, item["norad_id"]))
        if len(values) < cohort_size:
            raise RuntimeError(f"group {group} has {len(values)} stable objects; {cohort_size} required")

    for group, values in candidates.items():
        for rank, item in enumerate(values[:cohort_size]):
            for split, spec in split_specs.items():
                split_reference = parse_utc_datetime(spec["start_utc"])
                split_classification = item["split_classifications"][split_reference.isoformat()]
                relative = Path("data") / split / group / f"{item['norad_id']}.npy"
                shared = {key: value for key, value in item.items() if key != "split_classifications"}
                entries.append(
                    {
                        **shared,
                        "split": split,
                        "rank_in_group": rank,
                        "interval_start_utc": spec["start_utc"],
                        "interval_end_utc": spec["end_utc"],
                        "split_group_reference_utc": split_classification["timestamp"],
                        "split_semi_major_axis_altitude_km": split_classification[
                            "semi_major_axis_altitude_km"
                        ],
                        "split_eccentricity": split_classification["eccentricity"],
                        "array_path": str((work_root / relative).resolve()),
                        "status": "selected",
                    }
                )
    return {
        "schema_version": 3,
        "raw_root": str(raw_root.resolve()),
        "protocol_id": protocol["benchmark_id"],
        "protocol_sha256": sha256(Path(protocol["_path"])),
        "source_scan": {
            "strict_csv_files_scanned": scanned,
            "short_or_outside_group": short_or_excluded,
            "eligible_covering_full_period": {group: len(values) for group, values in candidates.items()},
            "stable_across_split_starts": {group: len(values) for group, values in candidates.items()},
        },
        "entries": entries,
    }


def line_at_offset(stream: BinaryIO, offset: int) -> tuple[int, int, datetime] | None:
    stream.seek(offset)
    if offset > 0:
        stream.readline()
    start = stream.tell()
    line = stream.readline()
    if not line:
        return None
    token = line.split(b",", 1)[0].decode("ascii", errors="strict")
    if token == "Epoch_UTC":
        start = stream.tell()
        line = stream.readline()
        if not line:
            return None
        token = line.split(b",", 1)[0].decode("ascii", errors="strict")
    return start, stream.tell(), parse_utc_datetime(token)


def find_start_offset(path: Path, target: datetime) -> int:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        low, high = 0, stream.tell()
        while high - low > 262_144:
            middle = (low + high) // 2
            item = line_at_offset(stream, middle)
            if item is None:
                high = middle
                continue
            start, end, timestamp = item
            if timestamp < target:
                low = max(low + 1, end)
            else:
                high = min(high - 1, start)
        stream.seek(max(0, low - 262_144))
        if stream.tell() > 0:
            stream.readline()
        while line := stream.readline():
            start = stream.tell() - len(line)
            token = line.split(b",", 1)[0].decode("ascii", errors="strict")
            if token == "Epoch_UTC":
                continue
            if parse_utc_datetime(token) >= target:
                return start
    raise RuntimeError(f"no row at or after {target.isoformat()} in {path}")


def extract_entry(entry: dict[str, Any]) -> dict[str, Any]:
    source = Path(entry["source_path"])
    destination = Path(entry["array_path"])
    start = parse_utc_datetime(entry["interval_start_utc"])
    end = parse_utc_datetime(entry["interval_end_utc"])
    expected_rows_raw = (end - start).total_seconds() / 60.0
    if not expected_rows_raw.is_integer():
        raise ValueError("split boundaries are not aligned to the 60 s sampling contract")
    expected_rows = int(expected_rows_raw)
    if destination.exists():
        actual_start, _ = read_state_at(source, start)
        if not start <= actual_start < start + timedelta(seconds=60):
            raise ValueError(f"existing array source cannot align to interval start: {source}")
        actual_end = actual_start + timedelta(seconds=60 * (expected_rows - 1))
        array = np.load(destination, mmap_mode="r", allow_pickle=False)
        if array.shape != (expected_rows, 6) or array.dtype != np.float32:
            raise ValueError(f"existing prepared array is invalid: {destination}")
        if not np.isfinite(array).all():
            raise ValueError(f"existing prepared array contains non-finite values: {destination}")
        values64 = np.asarray(array, dtype=np.float64)
        return {
            "status": "ready",
            "rows": len(array),
            "actual_start_utc": actual_start.isoformat(),
            "actual_end_utc": actual_end.isoformat(),
            "array_size_bytes": destination.stat().st_size,
            "array_sha256": sha256(destination),
            "state_sum": values64.sum(axis=0).tolist(),
            "state_sum_sq": np.square(values64).sum(axis=0).tolist(),
            "recovered_existing_array": True,
        }
    offset = find_start_offset(source, start)
    parts: list[np.ndarray] = []
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    previous_ns: int | None = None
    with source.open("rb") as stream:
        stream.seek(offset)
        reader = pd.read_csv(
            stream,
            header=None,
            names=CSV_COLUMNS,
            chunksize=200_000,
            dtype={column: np.float64 for column in TARGET_COLUMNS},
        )
        for frame in reader:
            times = pd.to_datetime(
                frame["Epoch_UTC"], format="ISO8601", errors="raise", utc=True
            ).dt.tz_localize(None)
            keep = (times >= pd.Timestamp(start)) & (times < pd.Timestamp(end))
            if keep.any():
                kept_times = times[keep]
                time_ns = kept_times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                if previous_ns is not None and int(time_ns[0]) - previous_ns != ONE_MINUTE_NS:
                    raise ValueError(f"time gap at chunk boundary in {source}")
                if len(time_ns) > 1 and not np.all(np.diff(time_ns) == ONE_MINUTE_NS):
                    bad_index = int(np.flatnonzero(np.diff(time_ns) != ONE_MINUTE_NS)[0])
                    raise ValueError(f"time gap near {kept_times.iloc[bad_index]} in {source}")
                previous_ns = int(time_ns[-1])
                first_timestamp = first_timestamp or kept_times.iloc[0].isoformat()
                last_timestamp = kept_times.iloc[-1].isoformat()
                values = frame.loc[keep, list(TARGET_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite state in selected interval: {source}")
                parts.append(values)
            if times.iloc[-1] >= pd.Timestamp(end):
                break
    if not parts:
        raise ValueError(f"selected interval is empty: {source}")
    array = np.concatenate(parts, axis=0)
    if first_timestamp is None:
        raise AssertionError("nonempty extraction has no first timestamp")
    aligned_start = parse_utc_datetime(first_timestamp)
    expected_last = (aligned_start + timedelta(seconds=60 * (expected_rows - 1))).isoformat()
    if not start <= aligned_start < start + timedelta(seconds=60) or last_timestamp != expected_last:
        raise ValueError(
            f"interval boundary mismatch in {source}: {first_timestamp} .. {last_timestamp}"
        )
    if len(array) != expected_rows:
        raise ValueError(
            f"interval row count mismatch in {source}: {len(array)} != {expected_rows}"
        )
    minimum = 1440 + 1440
    if len(array) < minimum:
        raise ValueError(f"selected interval has {len(array)} rows; at least {minimum} required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npy")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, destination)
    total = array.astype(np.float64).sum(axis=0)
    total_sq = np.square(array.astype(np.float64)).sum(axis=0)
    return {
        "status": "ready",
        "rows": len(array),
        "actual_start_utc": first_timestamp,
        "actual_end_utc": last_timestamp,
        "array_size_bytes": destination.stat().st_size,
        "array_sha256": sha256(destination),
        "state_sum": total.tolist(),
        "state_sum_sq": total_sq.tolist(),
    }


def prepare(manifest_path: Path, manifest: dict[str, Any], workers: int) -> None:
    pending = [entry for entry in manifest["entries"] if entry.get("status") != "ready"]
    if not pending:
        print("all selected arrays are already ready", flush=True)
        return
    by_key = {(entry["split"], entry["group"], entry["norad_id"]): entry for entry in manifest["entries"]}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_entry, entry): entry for entry in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            original = futures[future]
            key = (original["split"], original["group"], original["norad_id"])
            target = by_key[key]
            try:
                target.update(future.result())
                target.pop("error", None)
                print(f"[{index}/{len(pending)}] ready {key}", flush=True)
            except Exception as exc:
                target["status"] = "error"
                target["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{index}/{len(pending)}] error {key}: {target['error']}", flush=True)
            atomic_json(manifest_path, manifest)
    errors = [entry for entry in manifest["entries"] if entry.get("status") != "ready"]
    if errors:
        raise RuntimeError(f"{len(errors)} selected arrays failed; inspect {manifest_path}")


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    raw_root = args.raw_root.resolve()
    work_root = args.work_root.resolve()
    protocol_path = args.protocol.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    if int(protocol["sample_seconds"]) != 60:
        raise ValueError("the Pruned extraction implementation requires 60 s samples")
    protocol["_path"] = str(protocol_path)
    try:
        work_root.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("work-root must not be inside the immutable raw-root")
    manifest_path = work_root / "data" / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("protocol_sha256") != sha256(protocol_path):
            raise ValueError("existing manifest was created from a different protocol")
    else:
        work_root.mkdir(parents=True, exist_ok=True)
        manifest = select_entries(raw_root, protocol, work_root, args.workers)
        atomic_json(manifest_path, manifest)
    counts: dict[str, int] = {}
    for entry in manifest["entries"]:
        key = f"{entry['split']}/{entry['group']}"
        counts[key] = counts.get(key, 0) + 1
    print({"manifest": str(manifest_path), "selected": counts}, flush=True)
    if not args.select_only:
        prepare(manifest_path, manifest, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
