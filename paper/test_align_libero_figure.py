import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

import align_libero_figure as figure4


class ExtractPanelSeriesTest(unittest.TestCase):
    def test_uses_stored_task57_results_when_source_has_one_panel(self):
        panels = figure4.extract_panel_series(figure4.SOURCE)

        self.assertEqual(len(panels), 2)
        self.assertEqual(set(panels[0]), {"shore", "dsrl"})
        self.assertEqual(set(panels[1]), {"shore", "dsrl"})
        self.assertEqual(len(panels[0]["shore"]["line"]), 41)
        self.assertEqual(len(panels[0]["dsrl"]["line"]), 40)
        self.assertIn("sem", panels[1]["shore"])
        self.assertIn("sem", panels[1]["dsrl"])
        self.assertAlmostEqual(
            panels[0]["shore"]["line"][-1][0], 400.0, places=3
        )
        self.assertAlmostEqual(
            panels[0]["shore"]["line"][-1][1], 1.0, places=3
        )
        self.assertAlmostEqual(
            panels[0]["dsrl"]["line"][-1][0], 400.0, places=3
        )
        self.assertAlmostEqual(panels[1]["shore"]["line"][-1][0], 400.0, places=3)
        self.assertAlmostEqual(
            panels[1]["shore"]["line"][-1][1], 1.0, places=3
        )
        self.assertGreater(len(panels[0]["shore"]["band"]), 80)
        self.assertGreater(len(panels[0]["dsrl"]["band"]), 75)
        self.assertGreater(len(panels[1]["shore"]["x"]), 40)
        self.assertGreaterEqual(len(panels[1]["dsrl"]["x"]), 40)


class AlignShoreStartApiTest(unittest.TestCase):
    def test_alignment_helper_exists(self):
        self.assertTrue(
            hasattr(figure4, "align_shore_start"),
            "align_shore_start is required",
        )

    def test_panel_preparation_helper_exists(self):
        self.assertTrue(
            hasattr(figure4, "prepare_panels"),
            "prepare_panels is required",
        )


class AlignShoreStartTest(unittest.TestCase):
    def setUp(self):
        self.panels = figure4.extract_panel_series(figure4.SOURCE)

    def test_left_extracted_series_changes_only_step_zero_center(self):
        source = self.panels[0]["shore"]
        before = copy.deepcopy(source)

        adjusted = figure4.align_shore_start(source)

        self.assertEqual(source, before)
        self.assertAlmostEqual(adjusted["line"][0][0], 0.0, places=9)
        self.assertAlmostEqual(adjusted["line"][0][1], 0.6, places=12)
        self.assertEqual(adjusted["line"][1:], before["line"][1:])

        old_zero_band = [y for x, y in before["band"] if abs(x) < 1e-9]
        new_zero_band = [
            y for x, y in adjusted["band"] if abs(x) < 1e-9
        ]
        self.assertAlmostEqual(
            max(new_zero_band) - min(new_zero_band),
            max(old_zero_band) - min(old_zero_band),
            places=12,
        )
        self.assertEqual(
            [(x, y) for x, y in adjusted["band"] if abs(x) >= 1e-9],
            [(x, y) for x, y in before["band"] if abs(x) >= 1e-9],
        )

    def test_right_array_series_changes_only_step_zero_center(self):
        source = self.panels[1]["shore"]
        before = copy.deepcopy(source)

        adjusted = figure4.align_shore_start(source)

        self.assertEqual(source, before)
        self.assertAlmostEqual(adjusted["x"][0], 0.0, places=9)
        self.assertAlmostEqual(adjusted["y"][0], 0.6, places=12)
        self.assertEqual(adjusted["y"][1:], before["y"][1:])
        self.assertEqual(adjusted["sem"], before["sem"])
        self.assertEqual(adjusted["observed"], before["observed"])
        self.assertAlmostEqual(adjusted["line"][0][1], 0.6, places=12)
        self.assertEqual(adjusted["line"][1:], before["line"][1:])

    def test_prepare_panels_aligns_only_shore(self):
        raw = figure4.extract_panel_series(figure4.SOURCE)
        prepared = figure4.prepare_panels(figure4.SOURCE)

        self.assertEqual(len(prepared), 2)
        for panel in prepared:
            series = panel["shore"]
            start = (
                series["y"][0]
                if "sem" in series
                else series["line"][0][1]
            )
            self.assertAlmostEqual(start, 0.6, places=12)
        self.assertEqual(prepared[0]["dsrl"], raw[0]["dsrl"])
        self.assertEqual(prepared[1]["dsrl"], raw[1]["dsrl"])

    def test_alignment_rejects_nonzero_first_step(self):
        with self.assertRaisesRegex(ValueError, "step zero"):
            figure4.align_shore_start(
                {"line": [(10.0, 0.2)], "band": [(10.0, 0.1)]}
            )

    def test_alignment_rejects_out_of_range_target(self):
        with self.assertRaisesRegex(ValueError, "target"):
            figure4.align_shore_start(
                {"line": [(0.0, 0.2)], "band": [(0.0, 0.1)]},
                target=1.1,
            )


class RenderFigureTest(unittest.TestCase):
    def test_output_has_no_white_curve_endpoint_markers(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "figure4.pdf"
            figure4.render_figure(figure4.SOURCE, output)
            page = fitz.open(output)[0]
            white_markers = [
                drawing
                for drawing in page.get_drawings()
                if drawing["fill"]
                and all(abs(channel - 1.0) < 1e-6 for channel in drawing["fill"])
                and drawing["rect"].width < 10
                and drawing["rect"].height < 10
            ]
            self.assertEqual(white_markers, [])

    def test_output_matches_figure6_size_and_has_two_panels(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "figure4.pdf"
            figure4.render_figure(figure4.SOURCE, output)
            page = fitz.open(output)[0]
            reference = fitz.open(
                figure4.PAPER
                / "figure"
                / "fig_staged_vs_pothiql_threading_piece_v2.pdf"
            )[0]

            self.assertLess(abs(page.rect.width - reference.rect.width), 3.0)
            self.assertLess(abs(page.rect.height - reference.rect.height), 3.0)
            text = page.get_text()
            self.assertIn("LIBERO-10 · Task 8", text)
            self.assertIn("LIBERO-90 · Task 57", text)
            self.assertIn("SHORE-RL (Ours)", text)
            self.assertIn("DSRL", text)


if __name__ == "__main__":
    unittest.main()
