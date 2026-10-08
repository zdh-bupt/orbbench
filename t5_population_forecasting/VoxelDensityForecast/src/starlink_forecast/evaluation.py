from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import binom


def normalize_joint(voxel: np.ndarray, outside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    voxel_values = np.asarray(voxel, dtype=np.float64)
    outside_values = np.asarray(outside, dtype=np.float64)
    total = voxel_values.sum() + outside_values.sum()
    if total <= 0:
        categories = voxel_values.size + outside_values.size
        return (
            np.full_like(voxel_values, 1.0 / categories),
            np.full_like(outside_values, 1.0 / categories),
        )
    return voxel_values / total, outside_values / total


def joint_tv(
    voxel_probability: np.ndarray,
    outside_probability: np.ndarray,
    target_voxel: np.ndarray,
    target_outside: np.ndarray,
) -> float:
    target_voxel_probability, target_outside_probability = normalize_joint(target_voxel, target_outside)
    return 0.5 * float(
        np.abs(voxel_probability - target_voxel_probability).sum()
        + np.abs(outside_probability - target_outside_probability).sum()
    )


def pool_counts(field: np.ndarray, factors: tuple[int, int, int] = (2, 2, 2)) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if any(size % factor for size, factor in zip(values.shape, factors)):
        raise ValueError("regional pooling factors must divide the grid")
    d, h, w = values.shape
    fd, fh, fw = factors
    return values.reshape(d // fd, fd, h // fh, fh, w // fw, fw).sum(axis=(1, 3, 5))


def regional_relative_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    pooled_prediction = pool_counts(prediction)
    pooled_target = pool_counts(target)
    return float(np.abs(pooled_prediction - pooled_target).sum() / max(target.sum(), 1.0))


def _dilate(events: np.ndarray) -> np.ndarray:
    result = np.zeros_like(events, dtype=bool)
    depth, latitude, _ = events.shape
    for dz in (-1, 0, 1):
        for dlat in (-1, 0, 1):
            shifted_lat = np.zeros_like(events, dtype=bool)
            source_lat_start = max(0, -dlat)
            source_lat_end = min(latitude, latitude - dlat)
            target_lat_start = max(0, dlat)
            target_lat_end = min(latitude, latitude + dlat)
            shifted_lat[:, target_lat_start:target_lat_end] = events[:, source_lat_start:source_lat_end]
            for dlon in (-1, 0, 1):
                shifted = np.roll(shifted_lat, dlon, axis=2)
                if dz < 0:
                    result[: depth + dz] |= shifted[-dz:]
                elif dz > 0:
                    result[dz:] |= shifted[: depth - dz]
                else:
                    result |= shifted
    return result


def neighbor_tolerant_hotspot_f1(
    expected_count: np.ndarray,
    target_hard_count: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    predicted = np.asarray(expected_count) >= threshold
    observed = np.asarray(target_hard_count) >= threshold
    if not predicted.any() and not observed.any():
        return 1.0
    precision = float((predicted & _dilate(observed)).sum() / max(predicted.sum(), 1))
    recall = float((observed & _dilate(predicted)).sum() / max(observed.sum(), 1))
    return 2.0 * precision * recall / max(precision + recall, 1.0e-12)


def hotspot_ece(
    voxel_probability: np.ndarray,
    future_total: int,
    target_hard_count: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    probability = np.clip(np.asarray(voxel_probability, dtype=np.float64), 0.0, 1.0)
    occupancy_probability = 1.0 - np.power(1.0 - probability, max(int(future_total), 0))
    observed = (np.asarray(target_hard_count) > 0).astype(np.float64)
    confidence = occupancy_probability.ravel()
    truth = observed.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(mask):
            result += float(mask.mean() * abs(confidence[mask].mean() - truth[mask].mean()))
    return result


def binomial_interval_coverage(
    voxel_probability: np.ndarray,
    outside_probability: np.ndarray,
    future_total: int,
    target_hard_voxel: np.ndarray,
    target_hard_outside: np.ndarray,
    *,
    interval: float = 0.90,
) -> float:
    probability = np.concatenate((voxel_probability.ravel(), outside_probability.ravel()))
    observed = np.concatenate((target_hard_voxel.ravel(), target_hard_outside.ravel()))
    alpha = (1.0 - interval) / 2.0
    lower = binom.ppf(alpha, int(future_total), probability)
    upper = binom.ppf(1.0 - alpha, int(future_total), probability)
    return float(np.mean((observed >= lower) & (observed <= upper)))


def block_bootstrap_mean_interval(
    times_ns: np.ndarray,
    values: np.ndarray,
    *,
    block_days: int,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    times = np.asarray(times_ns, dtype=np.int64)
    scores = np.asarray(values, dtype=np.float64)
    if len(times) != len(scores) or not len(scores):
        raise ValueError("bootstrap inputs must be non-empty and aligned")
    block_ns = np.int64(block_days * 86_400_000_000_000)
    labels = (times - times.min()) // block_ns
    block_values = np.asarray([scores[labels == label].mean() for label in np.unique(labels)])
    rng = np.random.default_rng(seed)
    draws = rng.choice(block_values, size=(samples, len(block_values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def evaluate_predictions(
    records: list[dict[str, Any]],
    horizons: tuple[int, ...],
    config: dict[str, Any],
    *,
    formally_eligible: bool,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    times_by_horizon: dict[int, list[int]] = defaultdict(list)
    for record in records:
        horizon = int(record["horizon"])
        times_by_horizon[horizon].append(int(record["origin_time_ns"]))
        for model_name in ("learned", "physics", "persistence"):
            for metric, value in record[model_name].items():
                grouped[(model_name, horizon, metric)].append(float(value))
    summary: dict[str, Any] = {}
    for model_name in ("learned", "physics", "persistence"):
        summary[model_name] = {
            str(horizon): {
                metric: _mean(values)
                for (name, lead, metric), values in grouped.items()
                if name == model_name and lead == horizon
            }
            for horizon in horizons
        }

    acceptance = config["acceptance"]
    gates: dict[str, Any] = {"formal_checkpoint": formally_eligible}
    all_passed = formally_eligible
    for horizon in horizons:
        key = str(horizon)
        learned = summary["learned"][key]
        physics = summary["physics"][key]
        persistence = summary["persistence"][key]
        horizon_gates = {
            "tv": learned["tv"] <= float(acceptance["tv_max"][horizon]),
            "regional_relative_mae": learned["regional_relative_mae"]
            <= float(acceptance["regional_relative_mae_max"][horizon]),
            "neighbor_hotspot_f1": learned["neighbor_hotspot_f1"]
            >= float(acceptance["neighbor_hotspot_f1_min"][horizon]),
            "conservation": learned["conservation_error"] <= 1.0e-6,
            "positive_skill_vs_persistence": learned["tv"] < persistence["tv"],
            "global_degradation_vs_physics": learned["tv"]
            <= physics["tv"] * (1.0 + float(acceptance["max_global_degradation_vs_physics"])),
        }
        physics_values = np.asarray(grouped[("physics", horizon, "tv")])
        learned_values = np.asarray(grouped[("learned", horizon, "tv")])
        hard = physics_values >= np.quantile(physics_values, 0.75)
        relative_improvement = (physics_values[hard] - learned_values[hard]) / np.maximum(
            physics_values[hard], 1.0e-12
        )
        hard_times = np.asarray(times_by_horizon[horizon], dtype=np.int64)[hard]
        ci_low, ci_high = block_bootstrap_mean_interval(
            hard_times,
            relative_improvement,
            block_days=int(acceptance["bootstrap_block_days"]),
            samples=int(acceptance["bootstrap_samples"]),
            seed=int(config["training"]["seed"]) + horizon,
        )
        hard_mean = float(relative_improvement.mean())
        horizon_gates["hard_subset_improvement"] = (
            hard_mean >= float(acceptance["min_hard_subset_improvement_vs_physics"])
            and ci_low > 0.0
        )
        horizon_gates["hard_subset_improvement_mean"] = hard_mean
        horizon_gates["hard_subset_improvement_ci95"] = [ci_low, ci_high]
        horizon_gates["passed"] = all(
            value for name, value in horizon_gates.items() if name not in {
                "hard_subset_improvement_mean", "hard_subset_improvement_ci95", "passed"
            }
        )
        gates[key] = horizon_gates
        all_passed = all_passed and bool(horizon_gates["passed"])

    coverage = _mean(
        [value for (name, _, metric), values in grouped.items() if name == "learned" and metric == "interval_90_coverage" for value in values]
    )
    ece = _mean(
        [value for (name, _, metric), values in grouped.items() if name == "learned" and metric == "hotspot_ece" for value in values]
    )
    common = {
        "interval_90_coverage": coverage,
        "interval_coverage_passed": float(acceptance["interval_90_coverage_min"])
        <= coverage
        <= float(acceptance["interval_90_coverage_max"]),
        "hotspot_ece": ece,
        "hotspot_ece_passed": ece <= float(acceptance["hotspot_ece_max"]),
    }
    all_passed = all_passed and common["interval_coverage_passed"] and common["hotspot_ece_passed"]
    gates["common"] = common
    gates["accepted"] = bool(all_passed)
    return {
        "sample_lead_records": len(records),
        "metrics": summary,
        "acceptance": gates,
        "selected_model": "learned_physics_residual" if all_passed else "causal_physics_prior",
    }
