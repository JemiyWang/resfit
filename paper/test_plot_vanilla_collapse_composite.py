from pathlib import Path

import fitz


SOURCE = Path(__file__).with_name("plot_vanilla_collapse_composite.py")
FIGURE = Path(__file__).with_name("figure") / "fig_vanilla_collapse.pdf"


def test_composite_plot_generator_exists():
    assert SOURCE.exists()


def test_task_grid_uses_six_unlabelled_images_with_task_specific_borders():
    source = SOURCE.read_text()

    for filename in ("can0.png", "can1.png", "can2.png", "piece0.png", "piece1.png", "piece2.png"):
        assert filename in source
    assert "image_grid = main_grid[0, 0].subgridspec(2, 3, wspace=0.0, hspace=0.08)" in source
    assert "def add_task_image(ax, image_path, border_color):" in source
    image_function = source.split("def add_task_image", 1)[1].split("def fetch_histories", 1)[0]
    assert "set_title" not in image_function
    assert "#5DB8D2" in source
    assert "#F0A06F" in source
    assert "for spine in ax.spines.values()" in source


def test_image_columns_touch_and_legend_is_inside_right():
    source = SOURCE.read_text()

    assert "image_grid = main_grid[0, 0].subgridspec(2, 3, wspace=0.0, hspace=0.08)" in source
    legend_function = source.split("curve_ax.legend(", 1)[1].split("fig.savefig", 1)[0]
    assert 'loc="center right"' in legend_function
    assert "bbox_to_anchor" not in legend_function


def test_image_grid_cells_are_sized_for_touching_square_images():
    source = SOURCE.read_text()

    assert "width_ratios=(2.66, 2.66)" in source


def test_equal_panel_widths_labels_and_axis_text():
    source = SOURCE.read_text()

    assert "width_ratios=(2.66, 2.66)" in source
    assert "ax.set_xlabel(\"Env steps (k)\", fontsize=CURVE_XLABEL_SIZE" in source
    assert "ax.set_ylabel(\"Eval success rate\", fontsize=CURVE_YLABEL_SIZE" in source
    assert "def add_panel_tag(ax, label, color, x=0.02, y=0.98):" in source
    assert "add_panel_tag(image_axes[0], \"a\", \"#5DB8D2\", x=0.08, y=0.95)" in source
    assert "add_panel_tag(image_axes[3], \"b\", \"#F0A06F\", x=0.08, y=0.95)" in source
    assert "add_panel_tag(curve_ax, \"c\", \"#5EAE7B\", x=0.07)" in source


def test_equal_width_panels_keep_image_cells_square():
    source = SOURCE.read_text()

    assert "fig = plt.figure(figsize=(10.8, 4.35))" in source
    assert "wspace=0.20" in source


def test_a_and_b_tags_are_inset_from_the_image_borders():
    source = SOURCE.read_text()

    assert "def add_panel_tag(ax, label, color, x=0.02, y=0.98):" in source
    assert "add_panel_tag(image_axes[0], \"a\", \"#5DB8D2\", x=0.08, y=0.95)" in source
    assert "add_panel_tag(image_axes[3], \"b\", \"#F0A06F\", x=0.08, y=0.95)" in source


def test_panel_tag_letters_are_vertically_centered_in_their_circles():
    page = fitz.open(FIGURE)[0]
    spans = [
        span
        for block in page.get_text("dict")["blocks"]
        if "lines" in block
        for line in block["lines"]
        for span in line["spans"]
        if span["text"].strip() in {"a", "b", "c"}
    ]
    circles = [
        drawing["rect"]
        for drawing in page.get_drawings()
        if drawing["fill"] is not None
        and 20 < drawing["rect"].width < 30
        and 20 < drawing["rect"].height < 30
    ]

    assert len(spans) == 3
    assert len(circles) == 3
    for span in spans:
        text_rect = fitz.Rect(span["bbox"])
        circle = min(
            circles,
            key=lambda rect: abs(rect.x0 - text_rect.x0)
            + abs(rect.y0 - text_rect.y0),
        )
        assert abs(text_rect.y0 + text_rect.y1 - circle.y0 - circle.y1) < 3.0


def test_curve_text_and_panel_tags_are_sized_for_single_column_inclusion():
    source = SOURCE.read_text()

    assert "PAPER_TEXT_SIZE = 20" in source
    assert "fontsize=PAPER_TEXT_SIZE" in source
    assert "curve_ax.set_title" not in source


def test_curve_panel_matches_aligned_libero_plot_style():
    source = SOURCE.read_text()
    style_function = source.split("def style_curve_axes", 1)[1].split("def add_task_image", 1)[0]

    assert "CURVE_TICK_SIZE = 14" in source
    assert "CURVE_XLABEL_SIZE = 16" in source
    assert "CURVE_YLABEL_SIZE = 17" in source
    assert "CURVE_LEGEND_SIZE = 14" in source
    assert 'AXIS_COLOR = "#52514E"' in source
    assert 'GRID_COLOR = "#DCDCD7"' in source

    assert 'ax.set_xlabel("Env steps (k)", fontsize=CURVE_XLABEL_SIZE)' in style_function
    assert 'ax.set_ylabel("Eval success rate", fontsize=CURVE_YLABEL_SIZE)' in style_function
    assert "fontfamily=" not in style_function
    assert "Eval Success Rate" not in style_function

    assert "for spine in ax.spines.values():" in style_function
    assert "spine.set_visible(True)" in style_function
    assert "spine.set_color(AXIS_COLOR)" in style_function
    assert "spine.set_linewidth(0.8)" in style_function
    assert 'ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7' in style_function
    assert 'ax.tick_params(axis="both", labelsize=CURVE_TICK_SIZE' in style_function

    assert source.count("alpha=0.15") == 2
    assert "LONG_LINE_WIDTH = 2.7" in source
    assert "SHORT_LINE_WIDTH = 3.0" in source

    legend_function = source.split("curve_ax.legend(", 1)[1].split("fig.savefig", 1)[0]
    assert 'loc="center right"' in legend_function
    assert "bbox_to_anchor" not in legend_function
    assert "ncol=1" in legend_function
    assert "frameon=False" in legend_function
    assert 'prop={"size": CURVE_LEGEND_SIZE}' in legend_function
