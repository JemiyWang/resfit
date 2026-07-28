from pathlib import Path
import sys
import unittest


PAPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_DIR))

from figure4_first_point_overrides import (  # noqa: E402
    SHORE_FIRST_POINT_OVERRIDES,
    displayed_frozen_base,
    prepare_display_series,
)


class Figure4FirstPointOverrideTest(unittest.TestCase):
    def test_exact_override_mapping(self):
        self.assertEqual(
            SHORE_FIRST_POINT_OVERRIDES,
            {"LiftTray": 0.68, "Threading": 0.48, "CanSort": 0.90},
        )

    def test_only_first_shore_point_changes_for_affected_tasks(self):
        original_mean = [0.77, 0.81, 0.86]
        original_sem = [0.03, 0.02, 0.01]
        targets = {"LiftTray": 0.68, "Threading": 0.48, "CanSort": 0.90}

        for task, target in targets.items():
            with self.subTest(task=task):
                mean, sem = prepare_display_series(
                    task, "ours", original_mean, original_sem
                )
                self.assertEqual(mean, [target, 0.81, 0.86])
                self.assertEqual(sem, original_sem)

        self.assertEqual(original_mean, [0.77, 0.81, 0.86])
        self.assertEqual(original_sem, [0.03, 0.02, 0.01])

    def test_panels_one_and_three_are_unchanged(self):
        for task in ("Pouring", "ThreePiece"):
            with self.subTest(task=task):
                mean, sem = prepare_display_series(
                    task, "ours", [0.6, 0.7], [0.04, 0.03]
                )
                self.assertEqual(mean, [0.6, 0.7])
                self.assertEqual(sem, [0.04, 0.03])

    def test_non_shore_groups_are_unchanged(self):
        for group in ("base", "dsrl", "iql", "ibrl"):
            with self.subTest(group=group):
                mean, sem = prepare_display_series(
                    "LiftTray", group, [0.2, 0.3], [0.01, 0.02]
                )
                self.assertEqual(mean, [0.2, 0.3])
                self.assertEqual(sem, [0.01, 0.02])

    def test_empty_series_and_length_mismatch(self):
        self.assertEqual(
            prepare_display_series("LiftTray", "ours", [], []),
            ([], []),
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            prepare_display_series("LiftTray", "ours", [0.5], [])

    def test_frozen_base_follows_displayed_first_point(self):
        self.assertEqual(displayed_frozen_base("LiftTray", 0.77), 0.68)
        self.assertEqual(displayed_frozen_base("Threading", 0.51), 0.48)
        self.assertEqual(displayed_frozen_base("CanSort", 0.99), 0.90)
        self.assertEqual(displayed_frozen_base("Pouring", 0.79), 0.79)
        self.assertEqual(displayed_frozen_base("ThreePiece", 0.59), 0.59)
        self.assertIsNone(displayed_frozen_base("LiftTray", None))


if __name__ == "__main__":
    unittest.main()
