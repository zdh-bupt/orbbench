from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from satlife_observable.metrics import (
    Boundary,
    Segment,
    apply_mapping,
    build_segments,
    evaluate_predictions,
    extract_boundaries,
    match_boundaries,
    match_segments,
)


class ObservableMetricTests(unittest.TestCase):
    def test_segment_matching_maximizes_total_iou(self) -> None:
        base = pd.Timestamp("2026-01-01", tz="UTC")
        truth = [
            Segment(1, 0, base, base + pd.Timedelta(hours=1)),
            Segment(1, 0, base + pd.Timedelta(hours=2), base + pd.Timedelta(hours=6)),
        ]
        predictions = [
            Segment(1, 0, base, base + pd.Timedelta(hours=4)),
            Segment(1, 0, base + pd.Timedelta(hours=5), base + pd.Timedelta(hours=6)),
        ]
        metrics, _ = match_segments(truth, predictions)
        self.assertEqual(metrics["matched_segments"], 2)
        self.assertAlmostEqual(metrics["segment_macro_duration_iou"], 0.25)

    def test_cluster_mapping_treats_integral_float_and_integer_as_same_id(self) -> None:
        mapped = apply_mapping(np.asarray([0, 1.0, np.int64(2)]), {"0.0": 2, "1": 1, "2": 0})
        np.testing.assert_array_equal(mapped, np.asarray([2, 1, 0]))

    def setUp(self) -> None:
        rows = []
        times = pd.date_range("2026-01-01", periods=4, freq="6h", tz="UTC")
        for norad_id, labels in [(1, [0, 0, 1, 1]), (2, [0, 0, 2, 2])]:
            for position, (timestamp, label) in enumerate(zip(times, labels, strict=True)):
                rows.append(
                    {
                        "SAMPLE_ID": f"{norad_id}:{position}",
                        "NORAD_CAT_ID": norad_id,
                        "TARGET_EPOCH_UTC": timestamp.isoformat(),
                        "LABEL_ID": label,
                        "IS_PRIMARY_LABEL": True,
                        "PREDICTED_LABEL_ID": label,
                    }
                )
        self.frame = pd.DataFrame(rows)

    def test_exact_prediction_scores_one(self) -> None:
        metrics, _, _, _, _ = evaluate_predictions(
            self.frame, "PREDICTED_LABEL_ID", [0, 1, 2, 3, 4]
        )
        self.assertAlmostEqual(metrics["boundary"]["24h"]["f1"], 1.0)
        self.assertAlmostEqual(metrics["segments"]["label_macro_duration_iou"], 1.0)
        self.assertEqual(metrics["evaluable_label_ids"], [0, 1, 2])
        self.assertEqual(metrics["untestable_label_ids"], [3, 4])

    def test_boundary_matching_never_crosses_satellites(self) -> None:
        truth = extract_boundaries(self.frame, "LABEL_ID", {0, 1, 2})
        predictions = [truth[0]]
        metrics, _ = match_boundaries(truth, predictions, 24.0)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)

    def test_transition_direction_is_required(self) -> None:
        truth = extract_boundaries(self.frame, "LABEL_ID", {0, 1, 2})[:1]
        reversed_boundary = type(truth[0])(
            norad_id=truth[0].norad_id,
            timestamp=truth[0].timestamp,
            from_label=truth[0].to_label,
            to_label=truth[0].from_label,
        )
        metrics, _ = match_boundaries(truth, [reversed_boundary], 24.0)
        self.assertEqual(metrics["true_positive"], 0)

    def test_boundary_matching_maximizes_cardinality_before_time_error(self) -> None:
        base = pd.Timestamp("2026-01-01", tz="UTC")
        truth = [
            Boundary(1, base, 0, 1),
            Boundary(1, base + pd.Timedelta(hours=10), 0, 1),
        ]
        predictions = [
            Boundary(1, base + pd.Timedelta(hours=6), 0, 1),
            Boundary(1, base + pd.Timedelta(hours=14), 0, 1),
        ]
        metrics, _ = match_boundaries(truth, predictions, 6.0)
        self.assertEqual(metrics["true_positive"], 2)

    def test_unknown_truth_breaks_timeline(self) -> None:
        frame = pd.DataFrame(
            {
                "SAMPLE_ID": ["a", "b", "c"],
                "NORAD_CAT_ID": [1, 1, 1],
                "TARGET_EPOCH_UTC": pd.date_range("2026-01-01", periods=3, freq="6h", tz="UTC").astype(str),
                "LABEL_ID": [0, 5, 1],
                "IS_PRIMARY_LABEL": [True, False, True],
                "SEGMENT_ID": ["a", "unknown", "b"],
                "PREDICTED_LABEL_ID": [0, 1, 1],
            }
        )
        metrics, _, _, _, _ = evaluate_predictions(frame, "PREDICTED_LABEL_ID", [0, 1])
        self.assertEqual(metrics["boundary"]["24h"]["true_boundaries"], 0)
        self.assertEqual(metrics["boundary"]["24h"]["predicted_boundaries"], 0)

    def test_explicit_true_segment_ids_are_respected(self) -> None:
        frame = pd.DataFrame(
            {
                "NORAD_CAT_ID": [1, 1, 1, 1],
                "TARGET_EPOCH_UTC": pd.date_range("2026-01-01", periods=4, freq="6h", tz="UTC").astype(str),
                "LABEL_ID": [0, 0, 0, 0],
                "SEGMENT_ID": ["a", "a", "b", "b"],
            }
        )
        segments = build_segments(frame, "LABEL_ID", {0}, segment_id_column="SEGMENT_ID")
        self.assertEqual(len(segments), 2)

    def test_segments_use_time_duration(self) -> None:
        segments = build_segments(self.frame, "LABEL_ID", {0, 1, 2})
        self.assertEqual(len(segments), 4)
        self.assertTrue(all(segment.end > segment.start for segment in segments))


if __name__ == "__main__":
    unittest.main()
