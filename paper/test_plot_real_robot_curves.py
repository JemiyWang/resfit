import unittest

import align_libero_figure as figure5
import plot_real_robot_curves as figure8


class Figure8MarkerStyleTest(unittest.TestCase):
    def test_method_colors_match_figure5_palette(self):
        self.assertEqual(
            figure8.METHOD_STYLES["SHORE-RL"]["color"],
            figure5.METHODS["shore"]["hex"],
        )
        self.assertEqual(
            figure8.METHOD_STYLES["ResFit"]["color"],
            figure5.METHODS["dsrl"]["hex"],
        )

    def test_markers_are_large_enough_for_reduced_manuscript_figure(self):
        self.assertEqual(figure8.MARKER_SIZE, 7.5)
        self.assertEqual(figure8.MARKER_EDGE_WIDTH, 1.4)
        self.assertEqual(figure8.METHOD_STYLES["SHORE-RL"]["marker"], "o")
        self.assertEqual(figure8.METHOD_STYLES["ResFit"]["marker"], "s")


class Figure9SingleColumnTypographyTest(unittest.TestCase):
    def test_base_font_is_eight_points_after_single_column_scaling(self):
        source_width = 776.88 / 72.0
        single_column_width = (7.0 - 0.375) / 2
        expected = (8.0 / 17.0) * (source_width / single_column_width)

        self.assertAlmostEqual(
            getattr(figure8, "FONT_COMPENSATION", 0.0),
            expected,
            places=8,
        )

        compensate = getattr(figure8, "compensated_font_size", lambda _: 0.0)
        on_page_scale = single_column_width / source_width
        self.assertAlmostEqual(
            getattr(figure8, "TARGET_SOURCE_FONT_SIZE", 0.0)
            * on_page_scale,
            8.0,
            places=8,
        )
        self.assertAlmostEqual(
            compensate(17) * on_page_scale,
            8.0,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
