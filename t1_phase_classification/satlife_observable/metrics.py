from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)


UNKNOWN_ID = 5
UNMAPPED_ID = -1


def _cluster_key(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return value
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    return str(value)


def fit_hungarian_mapping(true_labels: np.ndarray, cluster_ids: np.ndarray) -> dict[str, int]:
    true_labels = np.asarray(true_labels)
    cluster_ids = np.asarray(cluster_ids)
    labels = np.sort(np.unique(true_labels))
    clusters = pd.unique(cluster_ids)
    contingency = np.zeros((len(clusters), len(labels)), dtype=np.int64)
    cluster_lookup = {value: position for position, value in enumerate(clusters)}
    label_lookup = {value: position for position, value in enumerate(labels)}
    for cluster, label in zip(cluster_ids, true_labels, strict=True):
        contingency[cluster_lookup[cluster], label_lookup[label]] += 1
    rows, columns = linear_sum_assignment(-contingency)
    return {
        _cluster_key(clusters[row]): int(labels[column])
        for row, column in zip(rows, columns, strict=True)
    }


def apply_mapping(cluster_ids: Iterable[object], mapping: dict[str, int]) -> np.ndarray:
    canonical_mapping = {_cluster_key(key): label for key, label in mapping.items()}
    return np.asarray(
        [canonical_mapping.get(_cluster_key(value), UNMAPPED_ID) for value in cluster_ids],
        dtype=np.int64,
    )


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"1", "true", "yes"})


@dataclass(frozen=True)
class Boundary:
    norad_id: int
    timestamp: pd.Timestamp
    from_label: int
    to_label: int


@dataclass(frozen=True)
class Segment:
    norad_id: int
    label: int
    start: pd.Timestamp
    end: pd.Timestamp
    source_segment_id: str | None = None


def prepare_valid_timeline(
    frame: pd.DataFrame,
    valid_labels: set[int],
    maximum_gap_hours: float = 36.0,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    primary = _as_bool(frame["IS_PRIMARY_LABEL"])
    working = frame.copy()
    working["_TRUTH_VALID"] = primary & working["LABEL_ID"].astype(int).isin(valid_labels)
    for norad_id, track in working.groupby("NORAD_CAT_ID", sort=True):
        track = track.sort_values("TARGET_EPOCH_UTC").copy()
        times = pd.to_datetime(track["TARGET_EPOCH_UTC"], utc=True)
        gaps = times.diff().dt.total_seconds().div(3600.0).fillna(0.0)
        valid = track["_TRUTH_VALID"].astype(bool)
        previous_valid = valid.shift(fill_value=False)
        breaks = (~valid) | (~previous_valid) | gaps.gt(maximum_gap_hours)
        track["_VALID_RUN_ID"] = breaks.cumsum().astype(int)
        pieces.append(track.loc[valid])
    if not pieces:
        return working.iloc[0:0].copy()
    return pd.concat(pieces).sort_index()


def extract_boundaries(
    frame: pd.DataFrame,
    label_column: str,
    valid_labels: set[int],
    maximum_gap_hours: float = 36.0,
) -> list[Boundary]:
    boundaries: list[Boundary] = []
    group_columns = ["NORAD_CAT_ID"] + (["_VALID_RUN_ID"] if "_VALID_RUN_ID" in frame else [])
    for group_key, track in frame.groupby(group_columns, sort=True):
        norad_id = group_key[0] if isinstance(group_key, tuple) else group_key
        track = track.sort_values("TARGET_EPOCH_UTC")
        times = pd.to_datetime(track["TARGET_EPOCH_UTC"], utc=True).tolist()
        labels = track[label_column].astype(int).tolist()
        for position in range(1, len(track)):
            gap_hours = (times[position] - times[position - 1]).total_seconds() / 3600.0
            previous, current = labels[position - 1], labels[position]
            if gap_hours > maximum_gap_hours:
                continue
            if previous == current or previous not in valid_labels or current not in valid_labels:
                continue
            boundaries.append(
                Boundary(
                    norad_id=int(norad_id),
                    timestamp=times[position],
                    from_label=previous,
                    to_label=current,
                )
            )
    return boundaries


def match_boundaries(
    truth: list[Boundary], predictions: list[Boundary], tolerance_hours: float
) -> tuple[dict[str, float | int], pd.DataFrame]:
    tolerance_seconds = timedelta(hours=tolerance_hours).total_seconds()
    truth_count, prediction_count = len(truth), len(predictions)
    size = truth_count + prediction_count
    invalid_cost = 1_000_000.0
    unmatched_cost = 1.0
    cost = np.zeros((size, size), dtype=float)
    cost[:truth_count, :prediction_count] = invalid_cost
    cost[:truth_count, prediction_count:] = unmatched_cost
    cost[truth_count:, :prediction_count] = unmatched_cost
    for truth_index, true_boundary in enumerate(truth):
        for prediction_index, predicted_boundary in enumerate(predictions):
            if (
                true_boundary.norad_id == predicted_boundary.norad_id
                and true_boundary.from_label == predicted_boundary.from_label
                and true_boundary.to_label == predicted_boundary.to_label
            ):
                seconds = abs((true_boundary.timestamp - predicted_boundary.timestamp).total_seconds())
                if seconds <= tolerance_seconds:
                    cost[truth_index, prediction_index] = min(
                        seconds / max(tolerance_seconds, 1.0), 1.0 - 1e-9
                    )
    assigned_rows, assigned_columns = linear_sum_assignment(cost)
    rows: list[dict[str, object]] = []
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    for truth_index, prediction_index in zip(assigned_rows, assigned_columns, strict=True):
        if truth_index >= truth_count or prediction_index >= prediction_count:
            continue
        if cost[truth_index, prediction_index] >= unmatched_cost:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        true_boundary = truth[truth_index]
        predicted_boundary = predictions[prediction_index]
        seconds = abs((true_boundary.timestamp - predicted_boundary.timestamp).total_seconds())
        rows.append(
            {
                "NORAD_CAT_ID": true_boundary.norad_id,
                "FROM_LABEL_ID": true_boundary.from_label,
                "TO_LABEL_ID": true_boundary.to_label,
                "TRUE_TIME": true_boundary.timestamp.isoformat(),
                "PREDICTED_TIME": predicted_boundary.timestamp.isoformat(),
                "ABS_ERROR_HOURS": seconds / 3600.0,
            }
        )
    true_positive = len(used_truth)
    false_positive = len(predictions) - true_positive
    false_negative = len(truth) - true_positive
    precision = true_positive / len(predictions) if predictions else float(len(truth) == 0)
    recall = true_positive / len(truth) if truth else float(len(predictions) == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics: dict[str, float | int] = {
        "tolerance_hours": tolerance_hours,
        "true_boundaries": len(truth),
        "predicted_boundaries": len(predictions),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics, pd.DataFrame(rows)


def _chunk_edges(times: list[pd.Timestamp]) -> list[pd.Timestamp]:
    if len(times) == 1:
        return [times[0] - timedelta(hours=3), times[0] + timedelta(hours=3)]
    gaps = np.asarray(
        [(times[index] - times[index - 1]).total_seconds() for index in range(1, len(times))],
        dtype=float,
    )
    typical = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 6 * 3600.0
    first = times[0] - timedelta(seconds=typical / 2)
    last = times[-1] + timedelta(seconds=typical / 2)
    middle = [times[index - 1] + (times[index] - times[index - 1]) / 2 for index in range(1, len(times))]
    return [first, *middle, last]


def build_segments(
    frame: pd.DataFrame,
    label_column: str,
    valid_labels: set[int],
    maximum_gap_hours: float = 36.0,
    segment_id_column: str | None = None,
) -> list[Segment]:
    segments: list[Segment] = []
    group_columns = ["NORAD_CAT_ID"] + (["_VALID_RUN_ID"] if "_VALID_RUN_ID" in frame else [])
    for group_key, track in frame.groupby(group_columns, sort=True):
        norad_id = group_key[0] if isinstance(group_key, tuple) else group_key
        track = track.sort_values("TARGET_EPOCH_UTC")
        times_all = pd.to_datetime(track["TARGET_EPOCH_UTC"], utc=True).tolist()
        labels_all = track[label_column].astype(int).tolist()
        chunk_start = 0
        for position in range(1, len(track) + 1):
            at_end = position == len(track)
            if not at_end:
                gap_hours = (times_all[position] - times_all[position - 1]).total_seconds() / 3600.0
            if at_end or gap_hours > maximum_gap_hours:
                times = times_all[chunk_start:position]
                labels = labels_all[chunk_start:position]
                if segment_id_column is None:
                    segment_ids = [None] * len(labels)
                else:
                    segment_ids = track.iloc[chunk_start:position][segment_id_column].astype(str).tolist()
                edges = _chunk_edges(times)
                run_start = 0
                for run_end in range(1, len(labels) + 1):
                    run_finished = (
                        run_end == len(labels)
                        or labels[run_end] != labels[run_start]
                        or (
                            segment_id_column is not None
                            and segment_ids[run_end] != segment_ids[run_start]
                        )
                    )
                    if run_finished:
                        label = labels[run_start]
                        if label in valid_labels:
                            segments.append(
                                Segment(
                                    norad_id=int(norad_id),
                                    label=label,
                                    start=edges[run_start],
                                    end=edges[run_end],
                                    source_segment_id=segment_ids[run_start],
                                )
                            )
                        run_start = run_end
                chunk_start = position
    return segments


def interval_iou(left: Segment, right: Segment) -> float:
    start = max(left.start, right.start)
    end = min(left.end, right.end)
    intersection = max((end - start).total_seconds(), 0.0)
    union_start = min(left.start, right.start)
    union_end = max(left.end, right.end)
    union = max((union_end - union_start).total_seconds(), 0.0)
    return intersection / union if union else 0.0


def match_segments(
    truth: list[Segment], predictions: list[Segment]
) -> tuple[dict[str, object], pd.DataFrame]:
    pair_scores = np.zeros((len(truth), len(predictions)), dtype=float)
    for truth_index, true_segment in enumerate(truth):
        for prediction_index, predicted_segment in enumerate(predictions):
            if true_segment.norad_id != predicted_segment.norad_id:
                continue
            if true_segment.label != predicted_segment.label:
                continue
            pair_scores[truth_index, prediction_index] = interval_iou(
                true_segment, predicted_segment
            )
    if len(truth) and len(predictions):
        assigned_truth, assigned_predictions = linear_sum_assignment(-pair_scores)
    else:
        assigned_truth = np.asarray([], dtype=np.int64)
        assigned_predictions = np.asarray([], dtype=np.int64)
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    scores = np.zeros(len(truth), dtype=float)
    rows: list[dict[str, object]] = []
    for truth_index, prediction_index in zip(
        assigned_truth, assigned_predictions, strict=True
    ):
        score = float(pair_scores[truth_index, prediction_index])
        if score <= 0:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        scores[truth_index] = score
        true_segment = truth[truth_index]
        predicted_segment = predictions[prediction_index]
        rows.append(
            {
                "NORAD_CAT_ID": true_segment.norad_id,
                "LABEL_ID": true_segment.label,
                "TRUE_START": true_segment.start.isoformat(),
                "TRUE_END": true_segment.end.isoformat(),
                "PREDICTED_START": predicted_segment.start.isoformat(),
                "PREDICTED_END": predicted_segment.end.isoformat(),
                "DURATION_IOU": score,
            }
        )
    per_label: dict[str, float | None] = {}
    per_label_support: dict[str, int] = {}
    for label in sorted({segment.label for segment in truth}):
        selected = [scores[index] for index, segment in enumerate(truth) if segment.label == label]
        per_label[str(label)] = float(np.mean(selected)) if selected else None
        per_label_support[str(label)] = len(selected)
    per_norad = {}
    for norad_id in sorted({segment.norad_id for segment in truth}):
        selected = [scores[index] for index, segment in enumerate(truth) if segment.norad_id == norad_id]
        per_norad[str(norad_id)] = float(np.mean(selected)) if selected else None
    durations = np.asarray(
        [max((segment.end - segment.start).total_seconds(), 0.0) for segment in truth], dtype=float
    )
    duration_micro = float(np.average(scores, weights=durations)) if durations.sum() > 0 else 0.0
    label_values = [value for value in per_label.values() if value is not None]
    norad_values = [value for value in per_norad.values() if value is not None]
    symmetric_score = float(scores.sum() / max(len(truth), len(predictions), 1))
    metrics: dict[str, object] = {
        "true_segments": len(truth),
        "predicted_segments": len(predictions),
        "matched_segments": len(used_truth),
        "segment_macro_duration_iou": float(np.mean(scores)) if len(scores) else 0.0,
        "label_macro_duration_iou": float(np.mean(label_values)) if label_values else 0.0,
        "norad_macro_duration_iou": float(np.mean(norad_values)) if norad_values else 0.0,
        "duration_weighted_micro_iou": duration_micro,
        "symmetric_segment_iou_with_false_positive_penalty": symmetric_score,
        "per_label_duration_iou": per_label,
        "per_label_true_segment_support": per_label_support,
        "per_norad_duration_iou": per_norad,
        "unmatched_true_segments": len(truth) - len(used_truth),
        "unmatched_predicted_segments": len(predictions) - len(used_predictions),
        "oversegmentation_ratio": len(predictions) / len(truth) if truth else None,
    }
    return metrics, pd.DataFrame(rows)


def evaluate_predictions(
    frame: pd.DataFrame,
    predicted_label_column: str,
    valid_label_ids: list[int],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = _as_bool(frame["IS_PRIMARY_LABEL"])
    evaluated = frame.loc[primary].copy()
    truth = evaluated["LABEL_ID"].to_numpy(np.int64)
    predictions = evaluated[predicted_label_column].to_numpy(np.int64)
    truth_support = set(np.unique(truth).tolist())
    evaluable_label_ids = [value for value in valid_label_ids if value in truth_support]
    untestable_label_ids = [value for value in valid_label_ids if value not in truth_support]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=evaluable_label_ids,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "LABEL_ID": evaluable_label_ids,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    matrix_labels = [*evaluable_label_ids, UNMAPPED_ID]
    confusion = pd.DataFrame(
        confusion_matrix(truth, predictions, labels=matrix_labels),
        index=[str(value) for value in matrix_labels],
        columns=[str(value) for value in matrix_labels],
    )
    valid_set = set(evaluable_label_ids)
    timeline = prepare_valid_timeline(frame, valid_set)
    true_boundaries = extract_boundaries(timeline, "LABEL_ID", valid_set)
    predicted_boundaries = extract_boundaries(timeline, predicted_label_column, valid_set)
    boundary_metrics: dict[str, object] = {}
    boundary_matches: list[pd.DataFrame] = []
    for tolerance in [12.0, 24.0, 48.0]:
        result, matches = match_boundaries(true_boundaries, predicted_boundaries, tolerance)
        boundary_metrics[f"{int(tolerance)}h"] = result
        if not matches.empty:
            matches.insert(0, "TOLERANCE_HOURS", tolerance)
            boundary_matches.append(matches)
    true_segments = build_segments(
        timeline,
        "LABEL_ID",
        valid_set,
        segment_id_column="SEGMENT_ID" if "SEGMENT_ID" in timeline else None,
    )
    predicted_segments = build_segments(timeline, predicted_label_column, valid_set)
    segment_metrics, segment_matches = match_segments(true_segments, predicted_segments)
    metrics: dict[str, object] = {
        "sample_count": len(evaluated),
        "satellite_count": int(evaluated["NORAD_CAT_ID"].nunique()),
        "requested_label_ids": valid_label_ids,
        "evaluable_label_ids": evaluable_label_ids,
        "untestable_label_ids": untestable_label_ids,
        "adjusted_rand_index": float(adjusted_rand_score(truth, predictions)),
        "normalized_mutual_information": float(normalized_mutual_info_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=evaluable_label_ids, average="macro", zero_division=0)),
        "boundary": boundary_metrics,
        "segments": segment_metrics,
    }
    all_boundary_matches = pd.concat(boundary_matches, ignore_index=True) if boundary_matches else pd.DataFrame()
    return metrics, per_class, confusion, all_boundary_matches, segment_matches
