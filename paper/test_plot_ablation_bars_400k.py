import math
from pathlib import Path
import statistics
import tempfile
import unittest

import matplotlib.pyplot as plt

import paper.plot_ablation_bars_400k as ablation_plot
from paper.plot_ablation_bars_400k import (
    LEGEND_ORDER,
    PANELS,
    STY,
    WINDOW_STEPS,
    plot_summaries,
    summarize_seeds,
    window_mean,
)


def constant_window(value):
    return {step: value for step in WINDOW_STEPS}


class FixedWindowMetricTest(unittest.TestCase):
    def test_window_mean_uses_exactly_330k_through_400k(self):
        points = {
            step: step / 1_000_000
            for step in range(0, 500_001, 10_000)
        }
        expected = statistics.fmean(points[step] for step in WINDOW_STEPS)

        self.assertEqual(
            WINDOW_STEPS,
            tuple(range(330_000, 400_001, 10_000)),
        )
        self.assertAlmostEqual(window_mean(points), expected)

    def test_window_mean_rejects_an_incomplete_fixed_window(self):
        points = constant_window(0.5)
        del points[370_000]

        with self.assertRaisesRegex(ValueError, "370000"):
            window_mean(points)

    def test_summarize_seeds_computes_mean_and_sem_after_seed_windows(self):
        seeds = [
            constant_window(0.2),
            constant_window(0.4),
            constant_window(0.6),
        ]

        mean, sem, values = summarize_seeds(seeds)

        self.assertAlmostEqual(mean, 0.4)
        self.assertEqual(values, [0.2, 0.4, 0.6])
        self.assertAlmostEqual(
            sem,
            statistics.stdev(values) / math.sqrt(3),
        )

    def test_summarize_seeds_uses_no_sem_for_one_seed(self):
        mean, sem, values = summarize_seeds([constant_window(0.7)])

        self.assertAlmostEqual(mean, 0.7)
        self.assertIsNone(sem)
        self.assertEqual(values, [0.7])


class FigureContractTest(unittest.TestCase):
    def test_legend_visual_order_matches_bar_order(self):
        self.assertEqual(
            getattr(ablation_plot, "LEGEND_HANDLE_ORDER", None),
            [
                "full",
                "no_staged",
                "no_both",
                "subgoal_only",
                "no_subgoal",
            ],
        )

    def test_legend_labels_are_short_single_lines(self):
        for arm in LEGEND_ORDER:
            label = STY[arm]["label"]
            self.assertNotIn("\n", label)
            self.assertLessEqual(len(label), 22)

    def test_plot_uses_uniform_seven_point_type(self):
        self.assertEqual(
            getattr(ablation_plot, "FONT_SIZE_PT", None),
            7,
        )

    def test_legend_labels_fit_two_single_column_legend_columns(self):
        for arm in LEGEND_ORDER:
            self.assertLessEqual(len(STY[arm]["label"]), 22)

    def test_plot_uses_single_column_physical_size(self):
        self.assertEqual(
            getattr(ablation_plot, "FIGSIZE_INCHES", None),
            (3.35, 2.75),
        )
        self.assertLessEqual(ablation_plot.FIGSIZE_INCHES[1], 3.0)

    def test_inventory_matches_two_panel_figure_five_selection(self):
        self.assertEqual(
            list(PANELS),
            ["Pouring", "ThreePiece"],
        )
        self.assertNotIn("LiftTray", PANELS)
        self.assertTrue(
            all(set(arms) == set(LEGEND_ORDER) for arms in PANELS.values())
        )
        self.assertEqual(
            sum(
                len(runs)
                for arms in PANELS.values()
                for runs in arms.values()
            ),
            28,
        )

    def test_plot_summaries_writes_pdf_and_png_without_wandb(self):
        summaries = {
            task: {
                arm: (0.1 + 0.15 * index, 0.02, [0.1, 0.2, 0.3])
                for index, arm in enumerate(LEGEND_ORDER)
            }
            for task in PANELS
        }

        with tempfile.TemporaryDirectory() as directory:
            out_pdf = Path(directory) / "figure.pdf"
            out_png = Path(directory) / "figure.png"

            plot_summaries(summaries, out_pdf, out_png)

            self.assertGreater(out_pdf.stat().st_size, 1_000)
            self.assertGreater(out_png.stat().st_size, 1_000)
            image = plt.imread(out_png)
            self.assertGreater(image.shape[1], 500)
            self.assertLess(image.shape[1], 700)


if __name__ == "__main__":
    unittest.main()
