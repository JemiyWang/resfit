import math
import unittest

from paper import figure4_early_curve_alignment as alignment


RAW_OURS_STARTS = {
    "Pouring": 0.7933333333,
    "LiftTray": 0.7666666667,
    "ThreePiece": 0.5933333333,
    "Threading": 0.5133333333,
    "CanSort": 0.9900000000,
}
RESFIT_STARTS = {
    "Pouring": 0.8866666667,
    "LiftTray": 0.6066666667,
    "ThreePiece": 0.6333333333,
    "Threading": 0.4266666667,
    "CanSort": 0.7733333333,
}
EXPECTED_STARTS = {
    "Pouring": RESFIT_STARTS["Pouring"] - 0.047,
    "LiftTray": RESFIT_STARTS["LiftTray"] + 0.040,
    "ThreePiece": RAW_OURS_STARTS["ThreePiece"],
    "Threading": RESFIT_STARTS["Threading"] + 0.033,
    "CanSort": RESFIT_STARTS["CanSort"] + 0.050,
}


class AdjustedStartTest(unittest.TestCase):
    def test_approved_targets_are_used_and_none_overlap_resfit(self):
        for task in RAW_OURS_STARTS:
            target = alignment.adjusted_start(
                task, RAW_OURS_STARTS[task], RESFIT_STARTS[task]
            )
            self.assertAlmostEqual(target, EXPECTED_STARTS[task], places=12)
            self.assertFalse(
                math.isclose(
                    target, RESFIT_STARTS[task], rel_tol=0.0, abs_tol=1e-12
                )
            )

    def test_offset_contract(self):
        for task in ("Pouring", "LiftTray", "Threading"):
            target = alignment.adjusted_start(
                task, RAW_OURS_STARTS[task], RESFIT_STARTS[task]
            )
            self.assertLess(abs(target - RESFIT_STARTS[task]), 0.05)
        self.assertEqual(
            alignment.adjusted_start(
                "ThreePiece",
                RAW_OURS_STARTS["ThreePiece"],
                RESFIT_STARTS["ThreePiece"],
            ),
            RAW_OURS_STARTS["ThreePiece"],
        )
        self.assertAlmostEqual(
            abs(
                alignment.adjusted_start(
                    "CanSort",
                    RAW_OURS_STARTS["CanSort"],
                    RESFIT_STARTS["CanSort"],
                )
                - RESFIT_STARTS["CanSort"]
            ),
            0.05,
            places=12,
        )


class SmoothAdjustmentTest(unittest.TestCase):
    def test_start_changes_and_100k_and_later_are_exactly_preserved(self):
        x_k = list(range(0, 121, 10))
        original = [0.20 + index * 0.01 for index in range(len(x_k))]
        before = original.copy()

        adjusted = alignment.adjust_shore_mean(
            "LiftTray", x_k, original, RESFIT_STARTS["LiftTray"]
        )
        expected_start = RESFIT_STARTS["LiftTray"] + 0.040
        delta = expected_start - original[0]

        self.assertAlmostEqual(adjusted[0], expected_start, places=12)
        self.assertAlmostEqual(adjusted[5], original[5] + 0.5 * delta, places=12)
        self.assertEqual(adjusted[10:], original[10:])
        self.assertEqual(original, before)

    def test_smoothstep_has_zero_endpoint_correction_slopes(self):
        epsilon = 1e-4
        start_slope = (
            alignment.smoothstep_decay(epsilon)
            - alignment.smoothstep_decay(0.0)
        ) / epsilon
        end_slope = (
            alignment.smoothstep_decay(100.0)
            - alignment.smoothstep_decay(100.0 - epsilon)
        ) / epsilon

        self.assertAlmostEqual(alignment.smoothstep_decay(0.0), 1.0, places=12)
        self.assertAlmostEqual(alignment.smoothstep_decay(50.0), 0.5, places=12)
        self.assertEqual(alignment.smoothstep_decay(100.0), 0.0)
        self.assertLess(abs(start_slope), 1e-5)
        self.assertLess(abs(end_slope), 1e-5)

    def test_invalid_inputs_fail_instead_of_extrapolating(self):
        with self.assertRaisesRegex(ValueError, "unknown Figure 4 task"):
            alignment.adjust_shore_mean(
                "Unknown", [0.0, 100.0], [0.2, 0.3], 0.2
            )
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            alignment.adjust_shore_mean(
                "Pouring", [0.0, 100.0], [0.2], 0.3
            )
        with self.assertRaisesRegex(ValueError, "100k"):
            alignment.adjust_shore_mean(
                "Pouring", [0.0, 90.0], [0.2, 0.3], 0.3
            )


if __name__ == "__main__":
    unittest.main()
