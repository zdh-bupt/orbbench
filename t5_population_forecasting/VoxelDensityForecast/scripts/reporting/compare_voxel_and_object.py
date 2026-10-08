from __future__ import annotations

"""Create paired voxel-vs-object density comparison figures.

The object route is a per-satellite causal two-body propagation followed by
the same CIC aggregation used by the voxel model.  This keeps the comparison
on one closed population, one target, and one density grid.
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from starlink_forecast.closed_cohort import ClosedCohortDataset
from starlink_forecast.config import config_digest, load_config
from starlink_forecast.evaluation import normalize_joint
from starlink_forecast.training import (
    checkpoint_run_config,
    model_from_config,
    shard_manifest,
    shard_selections_for_range,
    split_ranges,
)


HORIZON_HOURS = 1
NS_PER_HOUR = 3_600_000_000_000
POOL_FACTORS = (1, 2, 3, 4, 6, 12)
METHODS = (
    ("voxel", "Voxel residual network (candidate)"),
    ("object", "Per-satellite Kepler + CIC"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render four paired +1 h voxel-vs-object density comparison figures."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument(
        "--object-prediction",
        required=True,
        help="Formal per-satellite propagation prediction exported as CIC expected counts.",
    )
    parser.add_argument("--truth-shard", required=True)
    parser.add_argument("--origin-time", required=True, help="UTC ISO timestamp for Figures 1-3.")
    parser.add_argument("--sample-origins", type=int, default=96)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate representative inputs and run one inference without writing outputs.",
    )
    return parser.parse_args()


def require_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def utc_ns(value: str) -> int:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("--origin-time must include a UTC offset or trailing Z")
    parsed = parsed.astimezone(timezone.utc)
    return int(np.datetime64(parsed.replace(tzinfo=None), "ns").astype(np.int64))


def utc_string(time_ns: int) -> str:
    return np.datetime_as_string(np.datetime64(int(time_ns), "ns"), unit="s") + "Z"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_decision(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    temperatures = value.get("temperatures")
    if not isinstance(temperatures, dict) or str(HORIZON_HOURS) not in temperatures:
        raise ValueError("deployment decision does not contain the +1 h calibration temperature")
    return value


def load_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_config = checkpoint_run_config(checkpoint, config)
    model = model_from_config(run_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def find_origin_item(dataset: ClosedCohortDataset, origin_time_ns: int) -> int:
    times = np.asarray(dataset.times_ns[dataset.origins], dtype=np.int64)
    matches = np.flatnonzero(times == int(origin_time_ns))
    if len(matches) != 1:
        raise ValueError(
            f"truth shard must contain exactly one requested origin; found {len(matches)}"
        )
    return int(matches[0])


def origin_population_ids(shard_path: Path, origin_time_ns: int) -> tuple[str, ...]:
    with np.load(shard_path, allow_pickle=False) as archive:
        if "satellite_ids" not in archive.files or "valid" not in archive.files:
            raise ValueError("truth shard is missing satellite_ids or valid")
        times = np.asarray(archive["time_ns"], dtype=np.int64)
        origins = np.asarray(archive["origin_indices"], dtype=np.int64)
        matches = np.flatnonzero(times[origins] == int(origin_time_ns))
        if len(matches) != 1:
            raise ValueError("truth shard does not uniquely identify the requested origin")
        raw_origin = int(origins[int(matches[0])])
        satellite_ids = np.asarray(archive["satellite_ids"]).astype(str)
        valid = np.asarray(archive["valid"][raw_origin], dtype=bool)
    return tuple(satellite_ids[valid].tolist())


def lead_index(horizons: Iterable[int]) -> int:
    values = [int(value) for value in horizons]
    if values.count(HORIZON_HOURS) != 1:
        raise ValueError(f"horizons must contain exactly one {HORIZON_HOURS} h lead")
    return values.index(HORIZON_HOURS)


def load_object_prediction(
    path: Path,
    origin_time_ns: int,
    truth_ids: tuple[str, ...],
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "selected_model",
            "origin_time_ns",
            "horizons_hours",
            "satellite_ids",
            "expected_count",
            "outside_expected_count",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"object prediction is missing arrays: {sorted(missing)}")
        selected_model = str(np.asarray(archive["selected_model"]).reshape(-1)[0])
        if selected_model != "causal_physics_prior":
            raise ValueError(
                "object prediction must be the causal per-satellite physics prior, "
                f"not {selected_model!r}"
            )
        prediction_origin = int(np.asarray(archive["origin_time_ns"]).reshape(-1)[0])
        if prediction_origin != int(origin_time_ns):
            raise ValueError("object prediction and requested origin do not match")
        index = lead_index(np.asarray(archive["horizons_hours"]).tolist())
        ids = tuple(np.asarray(archive["satellite_ids"]).astype(str).tolist())
        counts = np.asarray(archive["expected_count"][index], dtype=np.float64)
        outside = np.asarray(archive["outside_expected_count"][index], dtype=np.float64)
    if set(ids) != set(truth_ids) or len(ids) != len(truth_ids):
        raise ValueError("object prediction and truth do not use the same origin-known population")
    return {
        "counts": counts,
        "outside": outside,
        "satellite_ids": ids,
        "selected_model": selected_model,
    }


def calibrated_probabilities(
    voxel_probability: np.ndarray,
    outside_probability: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    categories = np.concatenate(
        (
            np.asarray(voxel_probability, dtype=np.float64).ravel(),
            np.asarray(outside_probability, dtype=np.float64).ravel(),
        )
    )
    logits = np.log(np.maximum(categories, 1.0e-12)) / float(temperature)
    weights = np.exp(logits - logits.max())
    probability = weights / weights.sum()
    return probability[:-2].reshape(voxel_probability.shape), probability[-2:]


def infer_voxel_candidate(
    model: torch.nn.Module,
    sample: dict[str, Any],
    lead: int,
    temperature: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    def tensor(name: str) -> torch.Tensor:
        return torch.from_numpy(np.asarray(sample[name], dtype=np.float32)[None]).to(device)

    with torch.inference_mode():
        output = model(
            tensor("features"),
            tensor("physics_prior"),
            tensor("physics_prior_outside"),
            tensor("future_total"),
        )
    voxel_probability = output.voxel_probability[0, lead].detach().cpu().numpy()
    outside_probability = output.outside_probability[0, lead].detach().cpu().numpy()
    voxel_probability, outside_probability = calibrated_probabilities(
        voxel_probability, outside_probability, temperature
    )
    total = float(np.asarray(sample["future_total"])[lead])
    return voxel_probability * total, outside_probability * total


def normalized_counts(
    counts: np.ndarray,
    outside: np.ndarray,
    total: float,
) -> tuple[np.ndarray, np.ndarray]:
    voxel_probability, outside_probability = normalize_joint(counts, outside)
    return voxel_probability * total, outside_probability * total


def pool_counts(field: np.ndarray, factor: int) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 3 or any(size % factor for size in values.shape):
        raise ValueError(f"pool factor {factor} does not divide grid shape {values.shape}")
    d, h, w = values.shape
    return values.reshape(
        d // factor,
        factor,
        h // factor,
        factor,
        w // factor,
        factor,
    ).sum(axis=(1, 3, 5))


def density_metrics(
    prediction: np.ndarray,
    prediction_outside: np.ndarray,
    truth: np.ndarray,
    truth_outside: np.ndarray,
    *,
    factor: int,
) -> dict[str, float]:
    pooled_prediction = pool_counts(prediction, factor)
    pooled_truth = pool_counts(truth, factor)
    predicted_probability, predicted_outside_probability = normalize_joint(
        pooled_prediction, prediction_outside
    )
    truth_probability, truth_outside_probability = normalize_joint(pooled_truth, truth_outside)
    tv = 0.5 * float(
        np.abs(predicted_probability - truth_probability).sum()
        + np.abs(predicted_outside_probability - truth_outside_probability).sum()
    )
    truth_in_grid = float(pooled_truth.sum())
    truth_total = float(truth_in_grid + np.asarray(truth_outside).sum())
    relative_mae = float(
        np.abs(pooled_prediction - pooled_truth).sum() / max(truth_in_grid, 1.0)
        + np.abs(np.asarray(prediction_outside) - np.asarray(truth_outside)).sum()
        / max(truth_total, 1.0)
    )
    return {
        "tv": tv,
        "density_mass_overlap_percent": 100.0 * (1.0 - tv),
        "regional_relative_mae": relative_mae,
    }


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def style_map_axis(axis: plt.Axes, *, xlabel: bool, ylabel: bool) -> None:
    axis.set_xticks(np.arange(-180, 181, 60))
    axis.set_yticks(np.arange(-90, 91, 30))
    axis.set_xlabel("Longitude (deg)" if xlabel else "")
    axis.set_ylabel("Latitude (deg)" if ylabel else "")


def render_density_six_panel(
    truth: np.ndarray,
    voxel: np.ndarray,
    object_counts: np.ndarray,
    origin_time_ns: int,
    output_dir: Path,
) -> tuple[list[Path], list[dict[str, float]]]:
    maps = {
        "truth": truth.sum(axis=0),
        "voxel": voxel.sum(axis=0),
        "object": object_counts.sum(axis=0),
    }
    maps["voxel_error"] = np.abs(maps["voxel"] - maps["truth"])
    maps["object_error"] = np.abs(maps["object"] - maps["truth"])
    maps["advantage"] = maps["object_error"] - maps["voxel_error"]
    mass_max = max(float(maps[name].max()) for name in ("truth", "voxel", "object"))
    error_max = max(float(maps[name].max()) for name in ("voxel_error", "object_error"))
    advantage_max = max(float(np.quantile(np.abs(maps["advantage"]), 0.99)), 1.0e-12)
    mass_norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=max(mass_max, 1.0e-12))
    error_norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=max(error_max, 1.0e-12))
    advantage_norm = mcolors.TwoSlopeNorm(
        vmin=-advantage_max, vcenter=0.0, vmax=advantage_max
    )
    extent = (-180.0, 180.0, -90.0, 90.0)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), constrained_layout=True)
    panels = (
        (0, 0, "truth", "Ground truth", "viridis", mass_norm),
        (0, 1, "voxel", "Voxel residual network (candidate)", "viridis", mass_norm),
        (0, 2, "object", "Per-satellite Kepler + CIC", "viridis", mass_norm),
        (1, 0, "voxel_error", "Voxel absolute error", "magma", error_norm),
        (1, 1, "object_error", "Object-route absolute error", "magma", error_norm),
        (
            1,
            2,
            "advantage",
            "Advantage: object error - voxel error",
            "RdBu_r",
            advantage_norm,
        ),
    )
    images: dict[str, Any] = {}
    for row, col, key, title, cmap, norm in panels:
        image = axes[row, col].imshow(
            maps[key], origin="lower", extent=extent, aspect="auto", cmap=cmap, norm=norm
        )
        images[key] = image
        axes[row, col].set_title(title)
        style_map_axis(axes[row, col], xlabel=row == 1, ylabel=col == 0)
    fig.colorbar(
        images["truth"],
        ax=axes[0, :].tolist(),
        location="right",
        shrink=0.86,
        label="CIC expected count per 5 deg x 5 deg column",
    )
    fig.colorbar(
        images["voxel_error"],
        ax=axes[1, :2].tolist(),
        location="bottom",
        shrink=0.86,
        label="Absolute CIC count error per column",
    )
    fig.colorbar(
        images["advantage"],
        ax=axes[1, 2],
        location="bottom",
        shrink=0.86,
        extend="both",
        label="Positive: voxel candidate has lower error",
    )
    target_time_ns = origin_time_ns + HORIZON_HOURS * NS_PER_HOUR
    fig.suptitle(
        "+1 h density comparison | origin "
        f"{utc_string(origin_time_ns)} | target {utc_string(target_time_ns)}",
        fontweight="normal",
    )
    rows: list[dict[str, float]] = []
    lat_centers = np.arange(-87.5, 90.0, 5.0)
    lon_centers = np.arange(-177.5, 180.0, 5.0)
    for lat_index, latitude in enumerate(lat_centers):
        for lon_index, longitude in enumerate(lon_centers):
            rows.append(
                {
                    "latitude_deg": float(latitude),
                    "longitude_deg": float(longitude),
                    "truth_count": float(maps["truth"][lat_index, lon_index]),
                    "voxel_candidate_count": float(maps["voxel"][lat_index, lon_index]),
                    "object_kepler_cic_count": float(maps["object"][lat_index, lon_index]),
                    "voxel_absolute_error": float(maps["voxel_error"][lat_index, lon_index]),
                    "object_absolute_error": float(maps["object_error"][lat_index, lon_index]),
                    "voxel_advantage": float(maps["advantage"][lat_index, lon_index]),
                }
            )
    return save_figure(fig, output_dir, "01_density_six_panel"), rows


def multiscale_rows(
    truth: np.ndarray,
    truth_outside: np.ndarray,
    voxel: np.ndarray,
    voxel_outside: np.ndarray,
    object_counts: np.ndarray,
    object_outside: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictions = {
        "voxel": (voxel, voxel_outside),
        "object": (object_counts, object_outside),
    }
    for factor in POOL_FACTORS:
        for key, label in METHODS:
            prediction, outside = predictions[key]
            metrics = density_metrics(
                prediction,
                outside,
                truth,
                truth_outside,
                factor=factor,
            )
            rows.append(
                {
                    "method_key": key,
                    "method": label,
                    "pool_factor": factor,
                    "height_bin_km": 50.0 * factor,
                    "latitude_bin_deg": 5.0 * factor,
                    "longitude_bin_deg": 5.0 * factor,
                    "tv": metrics["tv"],
                    "density_mass_overlap_percent": metrics[
                        "density_mass_overlap_percent"
                    ],
                    "pooled_relative_count_error": metrics["regional_relative_mae"],
                    "is_formal_regional_relative_mae_scale": factor == 2,
                }
            )
    return rows


def render_multiscale(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    styles = {
        "voxel": ("#0072B2", "o", "-"),
        "object": ("#D55E00", "s", "--"),
    }
    for key, label in METHODS:
        selected = [row for row in rows if row["method_key"] == key]
        factors = [int(row["pool_factor"]) for row in selected]
        color, marker, line = styles[key]
        axes[0].plot(
            factors,
            [float(row["density_mass_overlap_percent"]) for row in selected],
            color=color,
            marker=marker,
            linestyle=line,
            label=label,
        )
        axes[1].plot(
            factors,
            [float(row["pooled_relative_count_error"]) for row in selected],
            color=color,
            marker=marker,
            linestyle=line,
            label=label,
        )
    for axis in axes:
        axis.set_xticks(POOL_FACTORS)
        axis.set_xlabel("Pooling factor on each grid axis")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("CIC density mass overlap (%)")
    axes[0].set_title("Overlap after spatial pooling")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Pooled relative count error")
    axes[1].set_title("Count error after spatial pooling (factor 2 = formal RMAE)")
    fig.suptitle(
        "Multi-scale +1 h density agreement | original grid: 50 km x 5 deg x 5 deg",
        fontweight="normal",
    )
    return save_figure(fig, output_dir, "02_multiscale_overlap")


def altitude_mass_rows(
    truth: np.ndarray,
    truth_outside: np.ndarray,
    voxel: np.ndarray,
    voxel_outside: np.ndarray,
    object_counts: np.ndarray,
    object_outside: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    height_centers = 325.0 + np.arange(truth.shape[0]) * 50.0
    for index, height in enumerate(height_centers):
        rows.append(
            {
                "category": "in_grid_height_bin",
                "height_center_km": float(height),
                "truth_count": float(truth[index].sum()),
                "voxel_candidate_count": float(voxel[index].sum()),
                "object_kepler_cic_count": float(object_counts[index].sum()),
            }
        )
    for index, category in enumerate(("outside_below_300_km", "outside_above_1500_km")):
        rows.append(
            {
                "category": category,
                "height_center_km": "",
                "truth_count": float(truth_outside[index]),
                "voxel_candidate_count": float(voxel_outside[index]),
                "object_kepler_cic_count": float(object_outside[index]),
            }
        )
    return rows


def render_altitude_and_mass(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    profile = [row for row in rows if row["category"] == "in_grid_height_bin"]
    height = np.asarray([float(row["height_center_km"]) for row in profile])
    series = {
        "Ground truth": np.asarray([float(row["truth_count"]) for row in profile]),
        "Voxel residual network (candidate)": np.asarray(
            [float(row["voxel_candidate_count"]) for row in profile]
        ),
        "Per-satellite Kepler + CIC": np.asarray(
            [float(row["object_kepler_cic_count"]) for row in profile]
        ),
    }
    outside = {row["category"]: row for row in rows if row["category"] != "in_grid_height_bin"}
    compositions = {
        "Ground\ntruth": np.asarray(
            [
                float(outside["outside_below_300_km"]["truth_count"]),
                float(series["Ground truth"].sum()),
                float(outside["outside_above_1500_km"]["truth_count"]),
            ]
        ),
        "Voxel\ncandidate": np.asarray(
            [
                float(outside["outside_below_300_km"]["voxel_candidate_count"]),
                float(series["Voxel residual network (candidate)"].sum()),
                float(outside["outside_above_1500_km"]["voxel_candidate_count"]),
            ]
        ),
        "Object\nKepler+CIC": np.asarray(
            [
                float(outside["outside_below_300_km"]["object_kepler_cic_count"]),
                float(series["Per-satellite Kepler + CIC"].sum()),
                float(outside["outside_above_1500_km"]["object_kepler_cic_count"]),
            ]
        ),
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.5),
        gridspec_kw={"width_ratios": (1.75, 1.0)},
        constrained_layout=True,
    )
    profile_styles = (
        ("Ground truth", "#000000", "o", "-"),
        ("Voxel residual network (candidate)", "#0072B2", "^", "-"),
        ("Per-satellite Kepler + CIC", "#D55E00", "s", "--"),
    )
    for label, color, marker, line in profile_styles:
        axes[0].plot(height, series[label], label=label, color=color, marker=marker, linestyle=line)
    axes[0].set_xlabel("Altitude-bin center (km)")
    axes[0].set_ylabel("CIC expected count per 50 km shell")
    axes[0].set_title("Altitude-shell density profile")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    labels = list(compositions)
    values = np.stack([compositions[label] for label in labels])
    bottoms = np.zeros(len(labels), dtype=np.float64)
    categories = (
        ("Below 300 km", "#56B4E9"),
        ("Inside 300-1500 km", "#009E73"),
        ("Above 1500 km", "#CC79A7"),
    )
    for index, (category, color) in enumerate(categories):
        axes[1].bar(labels, values[:, index], bottom=bottoms, label=category, color=color)
        bottoms += values[:, index]
    for x, total in enumerate(bottoms):
        axes[1].text(x, total, f"{total:.0f}", ha="center", va="bottom")
    axes[1].set_ylabel("Expected satellite count")
    axes[1].set_title("In-grid and outside mass")
    axes[1].legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    fig.suptitle("+1 h altitude distribution and mass conservation", fontweight="normal")
    return save_figure(fig, output_dir, "03_altitude_profile_and_mass")


def collect_test_origin_refs(
    config: dict[str, Any],
    sample_count: int,
) -> tuple[list[tuple[int, Path, int]], int, dict[str, str]]:
    _, shards = shard_manifest(config)
    ranges = split_ranges(config, shards)
    test_range = ranges["test"]
    selections = shard_selections_for_range(shards, test_range)
    refs: list[tuple[int, Path, int]] = []
    for selection in selections:
        with np.load(selection.path, allow_pickle=False) as archive:
            times = np.asarray(archive["time_ns"], dtype=np.int64)
            origins = np.asarray(archive["origin_indices"], dtype=np.int64)
        mask = (times[origins] >= int(test_range.start.value)) & (
            times[origins] <= int(test_range.end.value)
        )
        dataset_item_indices = np.flatnonzero(mask)
        selected_times = times[origins[dataset_item_indices]]
        if len(selected_times) != selection.origin_count:
            raise RuntimeError(f"origin count drifted for {selection.path}")
        refs.extend(
            (int(origin_time), selection.path, int(item_index))
            for item_index, origin_time in zip(dataset_item_indices, selected_times)
        )
    refs.sort(key=lambda value: value[0])
    available = len(refs)
    if available == 0:
        raise RuntimeError("locked test range contains no origins")
    actual = min(sample_count, available)
    indices = np.floor((np.arange(actual) + 0.5) * available / actual).astype(int)
    sampled = [refs[int(index)] for index in indices]
    return sampled, available, {
        "start": test_range.start.isoformat(),
        "end": test_range.end.isoformat(),
    }


def paired_origin_rows(
    model: torch.nn.Module,
    config: dict[str, Any],
    temperature: float,
    device: torch.device,
    sample_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs, available, test_range = collect_test_origin_refs(config, sample_count)
    by_path: dict[Path, list[tuple[int, int]]] = {}
    for origin_time_ns, path, item_index in refs:
        by_path.setdefault(path, []).append((origin_time_ns, item_index))
    rows: list[dict[str, Any]] = []
    lead = lead_index(config["forecast"]["horizons"])
    for shard_number, (path, shard_refs) in enumerate(by_path.items(), start=1):
        dataset = ClosedCohortDataset(path, config, return_tensors=False)
        for origin_time_ns, item_index in shard_refs:
            if int(dataset.times_ns[dataset.origins[item_index]]) != origin_time_ns:
                raise RuntimeError("sampled origin reference drifted while loading a shard")
            sample = dataset[item_index]
            voxel, voxel_outside = infer_voxel_candidate(
                model, sample, lead, temperature, device
            )
            total = float(np.asarray(sample["future_total"])[lead])
            object_counts, object_outside = normalized_counts(
                np.asarray(sample["physics_prior"])[lead],
                np.asarray(sample["physics_prior_outside"])[lead],
                total,
            )
            truth = np.asarray(sample["target_counts"])[lead]
            truth_outside = np.asarray(sample["target_outside"])[lead]
            voxel_raw_metrics = density_metrics(
                voxel, voxel_outside, truth, truth_outside, factor=1
            )
            object_raw_metrics = density_metrics(
                object_counts, object_outside, truth, truth_outside, factor=1
            )
            voxel_regional_metrics = density_metrics(
                voxel, voxel_outside, truth, truth_outside, factor=2
            )
            object_regional_metrics = density_metrics(
                object_counts, object_outside, truth, truth_outside, factor=2
            )
            rows.append(
                {
                    "origin_time_ns": origin_time_ns,
                    "origin_time_utc": utc_string(origin_time_ns),
                    "shard": path.name,
                    "member_count": int(round(total)),
                    "horizon_hours": HORIZON_HOURS,
                    "voxel_tv": voxel_raw_metrics["tv"],
                    "object_tv": object_raw_metrics["tv"],
                    "voxel_minus_object_tv": (
                        voxel_raw_metrics["tv"] - object_raw_metrics["tv"]
                    ),
                    "voxel_regional_relative_mae": voxel_regional_metrics[
                        "regional_relative_mae"
                    ],
                    "object_regional_relative_mae": object_regional_metrics[
                        "regional_relative_mae"
                    ],
                    "voxel_minus_object_regional_relative_mae": (
                        voxel_regional_metrics["regional_relative_mae"]
                        - object_regional_metrics["regional_relative_mae"]
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "event": "paired_subset_shard_done",
                    "shard": path.name,
                    "shard_number": shard_number,
                    "shards": len(by_path),
                    "origins_done": len(rows),
                    "origins_total": len(refs),
                }
            ),
            flush=True,
        )
        del dataset
        gc.collect()
    rows.sort(key=lambda row: int(row["origin_time_ns"]))
    return rows, {
        "available_origins": available,
        "sampled_origins": len(rows),
        "test_range": test_range,
        "selection": "equal-rank deterministic sampling over chronologically sorted locked-test origins",
    }


def render_paired_scatter(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.1), constrained_layout=True)
    members = np.asarray([int(row["member_count"]) for row in rows], dtype=np.float64)
    panels = (
        ("object_tv", "voxel_tv", "Joint TV"),
        (
            "object_regional_relative_mae",
            "voxel_regional_relative_mae",
            "Regional relative MAE (2x2x2 pooling)",
        ),
    )
    scatters = []
    for axis, (x_key, y_key, metric_label) in zip(axes, panels):
        x = np.asarray([float(row[x_key]) for row in rows])
        y = np.asarray([float(row[y_key]) for row in rows])
        lower = min(float(x.min()), float(y.min()))
        upper = max(float(x.max()), float(y.max()))
        padding = max((upper - lower) * 0.07, 1.0e-6)
        bounds = (max(0.0, lower - padding), upper + padding)
        scatter = axis.scatter(
            x,
            y,
            c=members,
            cmap="viridis",
            s=34,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
        scatters.append(scatter)
        axis.plot(bounds, bounds, color="black", linestyle="--", linewidth=1.0, label="Equal error")
        axis.set_xlim(bounds)
        axis.set_ylim(bounds)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"Per-satellite Kepler + CIC: {metric_label}")
        axis.set_ylabel(f"Voxel residual network: {metric_label}")
        axis.set_title(metric_label)
        axis.grid(True, alpha=0.22)
        voxel_wins = int(np.sum(y < x))
        object_wins = int(np.sum(y > x))
        ties = int(len(x) - voxel_wins - object_wins)
        axis.text(
            0.03,
            0.97,
            f"Voxel lower: {voxel_wins}\nObject lower: {object_wins}\nTies: {ties}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    fig.colorbar(
        scatters[-1],
        ax=axes.tolist(),
        location="right",
        shrink=0.82,
        label="Origin-known member count",
    )
    fig.suptitle(
        f"Paired +1 h locked-test subset | n={len(rows)} | below diagonal favors voxel candidate",
        fontweight="normal",
    )
    return save_figure(fig, output_dir, "04_paired_origin_scatter")


def summarize_paired_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in ("tv", "regional_relative_mae"):
        voxel = np.asarray([float(row[f"voxel_{metric}"]) for row in rows])
        object_values = np.asarray([float(row[f"object_{metric}"]) for row in rows])
        difference = voxel - object_values
        summary[metric] = {
            "voxel_mean": float(voxel.mean()),
            "object_mean": float(object_values.mean()),
            "voxel_median": float(np.median(voxel)),
            "object_median": float(np.median(object_values)),
            "voxel_lower_count": int(np.sum(voxel < object_values)),
            "object_lower_count": int(np.sum(object_values < voxel)),
            "tie_count": int(np.sum(voxel == object_values)),
            "voxel_minus_object_quantiles": {
                "q05": float(np.quantile(difference, 0.05)),
                "q50": float(np.quantile(difference, 0.50)),
                "q95": float(np.quantile(difference, 0.95)),
            },
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_mass(name: str, counts: np.ndarray, outside: np.ndarray, expected: float) -> None:
    total = float(np.asarray(counts).sum() + np.asarray(outside).sum())
    relative_error = abs(total - expected) / max(expected, 1.0)
    if relative_error > 1.0e-5:
        raise ValueError(f"{name} relative mass error {relative_error:.6g} exceeds tolerance")


def main() -> int:
    args = parse_args()
    if args.sample_origins <= 0:
        raise SystemExit("--sample-origins must be positive")
    config_path = require_file(args.config, "config")
    checkpoint_path = require_file(args.checkpoint, "checkpoint")
    decision_path = require_file(args.decision, "deployment decision")
    object_prediction_path = require_file(args.object_prediction, "object prediction")
    truth_shard_path = require_file(args.truth_shard, "truth shard")
    config = load_config(config_path)
    decision = load_decision(decision_path)
    origin_time_ns = utc_ns(args.origin_time)
    device = resolve_device(args.device)
    model, checkpoint = load_model(checkpoint_path, config, device)
    dataset = ClosedCohortDataset(truth_shard_path, config, return_tensors=False)
    item = find_origin_item(dataset, origin_time_ns)
    sample = dataset[item]
    lead = lead_index(dataset.horizons)
    truth_ids = origin_population_ids(truth_shard_path, origin_time_ns)
    object_prediction = load_object_prediction(
        object_prediction_path, origin_time_ns, truth_ids
    )
    temperature = float(decision["temperatures"][str(HORIZON_HOURS)])
    voxel, voxel_outside = infer_voxel_candidate(
        model, sample, lead, temperature, device
    )
    truth = np.asarray(sample["target_counts"])[lead].astype(np.float64)
    truth_outside = np.asarray(sample["target_outside"])[lead].astype(np.float64)
    object_counts = np.asarray(object_prediction["counts"], dtype=np.float64)
    object_outside = np.asarray(object_prediction["outside"], dtype=np.float64)
    future_total = float(np.asarray(sample["future_total"])[lead])
    if len(truth_ids) != int(round(future_total)):
        raise ValueError("truth population size does not match the dataset future_total")
    if truth.shape != voxel.shape or truth.shape != object_counts.shape:
        raise ValueError("truth, voxel candidate, and object route do not share one grid")
    verify_mass("truth", truth, truth_outside, future_total)
    verify_mass("voxel candidate", voxel, voxel_outside, future_total)
    verify_mass("object route", object_counts, object_outside, future_total)
    stored_object_counts, stored_object_outside = normalized_counts(
        np.asarray(sample["physics_prior"])[lead],
        np.asarray(sample["physics_prior_outside"])[lead],
        future_total,
    )
    object_route_max_abs_difference = max(
        float(np.max(np.abs(stored_object_counts - object_counts))),
        float(np.max(np.abs(stored_object_outside - object_outside))),
    )
    representative_metrics: dict[str, dict[str, float]] = {}
    for key, prediction, outside in (
        ("voxel_candidate", voxel, voxel_outside),
        ("object_kepler_cic", object_counts, object_outside),
    ):
        raw = density_metrics(
            prediction, outside, truth, truth_outside, factor=1
        )
        regional = density_metrics(
            prediction, outside, truth, truth_outside, factor=2
        )
        representative_metrics[key] = {
            "joint_tv": raw["tv"],
            "density_mass_overlap_percent": raw["density_mass_overlap_percent"],
            "raw_grid_relative_count_error": raw["regional_relative_mae"],
            "regional_relative_mae_2x2x2": regional["regional_relative_mae"],
        }
    validated = {
        "origin_time_utc": utc_string(origin_time_ns),
        "target_time_utc": utc_string(origin_time_ns + NS_PER_HOUR),
        "member_count": int(round(future_total)),
        "device": str(device),
        "checkpoint_status": checkpoint.get("status"),
        "deployment_accepted": bool(decision.get("accepted", False)),
        "deployment_selected_model": decision.get("selected_model"),
        "object_route_max_abs_difference_vs_stored_shard_prior": object_route_max_abs_difference,
        "representative_metrics": representative_metrics,
    }
    if args.dry_run:
        print(json.dumps({"status": "validated", **validated}, indent=2))
        return 0

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    produced: list[Path] = []
    six_paths, density_rows = render_density_six_panel(
        truth, voxel, object_counts, origin_time_ns, output_dir
    )
    produced.extend(six_paths)
    density_csv = output_dir / "01_density_six_panel.csv"
    write_csv(density_csv, density_rows)
    produced.append(density_csv)

    scale_rows = multiscale_rows(
        truth,
        truth_outside,
        voxel,
        voxel_outside,
        object_counts,
        object_outside,
    )
    produced.extend(render_multiscale(scale_rows, output_dir))
    scale_csv = output_dir / "02_multiscale_overlap.csv"
    write_csv(scale_csv, scale_rows)
    produced.append(scale_csv)

    profile_rows = altitude_mass_rows(
        truth,
        truth_outside,
        voxel,
        voxel_outside,
        object_counts,
        object_outside,
    )
    produced.extend(render_altitude_and_mass(profile_rows, output_dir))
    profile_csv = output_dir / "03_altitude_profile_and_mass.csv"
    write_csv(profile_csv, profile_rows)
    produced.append(profile_csv)

    del dataset
    gc.collect()
    paired_rows, subset_metadata = paired_origin_rows(
        model,
        config,
        temperature,
        device,
        args.sample_origins,
    )
    produced.extend(render_paired_scatter(paired_rows, output_dir))
    paired_csv = output_dir / "04_paired_origin_scatter.csv"
    write_csv(paired_csv, paired_rows)
    produced.append(paired_csv)

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": {
            "horizon_hours": HORIZON_HOURS,
            "voxel_method": "PhysicsResidualUNetConvGRU3D frozen candidate, calibrated",
            "object_method": "causal per-satellite two-body Kepler propagation followed by CIC",
            "grid_shape": list(truth.shape),
            "grid_resolution": {
                "height_km": 50.0,
                "latitude_deg": 5.0,
                "longitude_deg": 5.0,
            },
            "population_semantics": "same origin-known closed cohort",
        },
        "status": {
            "voxel_candidate_formally_accepted": bool(decision.get("accepted", False)),
            "deployment_selected_model": decision.get("selected_model"),
            "interpretation": (
                "The voxel network is a frozen but unaccepted candidate. The object route loses "
                "identity after CIC aggregation. Density overlap is not per-satellite exact-grid accuracy."
            ),
        },
        "representative_origin": validated,
        "paired_subset": {
            **subset_metadata,
            "metrics": summarize_paired_rows(paired_rows),
        },
        "inputs": {
            "config": {"path": portable_path(config_path), "sha256": sha256(config_path)},
            "checkpoint": {
                "path": portable_path(checkpoint_path),
                "sha256": sha256(checkpoint_path),
            },
            "decision": {
                "path": portable_path(decision_path),
                "sha256": sha256(decision_path),
            },
            "object_prediction": {
                "path": portable_path(object_prediction_path),
                "sha256": sha256(object_prediction_path),
            },
            "truth_shard": {
                "path": portable_path(truth_shard_path),
                "sha256": sha256(truth_shard_path),
            },
            "config_digest": config_digest(config),
        },
        "outputs": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in produced
        ],
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "rendered",
                "output_dir": str(output_dir),
                "files": len(produced) + 1,
                "representative_metrics": representative_metrics,
                "paired_subset": subset_metadata,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
