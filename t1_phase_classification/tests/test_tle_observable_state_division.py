from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_tle_observable_state_division import state_from_features


class ObservableStateDivisionTests(unittest.TestCase):
    def classify(
        self,
        sample_count: int,
        duration_days: float,
        net_change_km: float,
        range_km: float,
        positive_fraction: float,
        negative_fraction: float,
    ) -> str:
        return state_from_features(
            sample_count,
            duration_days,
            net_change_km,
            range_km,
            positive_fraction,
            negative_fraction,
        )[0]

    def test_return_states_are_separated_at_ten_days(self) -> None:
        short = self.classify(5, 9.99, 0.1, 5.0, 0.5, 0.5)
        long = self.classify(5, 10.0, 0.1, 5.0, 0.5, 0.5)
        self.assertEqual(short, "SHORT_EXCURSION_RETURN")
        self.assertEqual(long, "OFFSET_PLATEAU")

    def test_monotonic_states_follow_signed_change(self) -> None:
        raised = self.classify(8, 4.0, 8.0, 8.2, 0.75, 0.25)
        decayed = self.classify(8, 4.0, -2.0, 2.1, 0.25, 0.75)
        self.assertEqual(raised, "SUSTAINED_RAISE")
        self.assertEqual(decayed, "SUSTAINED_DECAY")

    def test_stable_rule_does_not_overlap_decay_boundary(self) -> None:
        stable = self.classify(8, 4.0, -0.89, 0.89, 0.25, 0.75)
        decayed = self.classify(8, 4.0, -0.90, 0.90, 0.25, 0.75)
        self.assertEqual(stable, "STABLE_ORBIT")
        self.assertEqual(decayed, "SUSTAINED_DECAY")

    def test_insufficient_or_ambiguous_fragment_is_unknown(self) -> None:
        too_short = self.classify(2, 1.0, 0.0, 0.2, 0.5, 0.5)
        ambiguous = self.classify(8, 5.0, 1.5, 3.0, 0.5, 0.5)
        self.assertEqual(too_short, "UNKNOWN_TRANSITION")
        self.assertEqual(ambiguous, "UNKNOWN_TRANSITION")


if __name__ == "__main__":
    unittest.main()
