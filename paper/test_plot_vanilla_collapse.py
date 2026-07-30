import ast
from pathlib import Path


SOURCE = Path(__file__).with_name("plot_vanilla_collapse.py")


def assigned_literal(name):
    tree = ast.parse(SOURCE.read_text())
    node = next(
        statement.value
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        )
    )
    return ast.literal_eval(node)


def test_target_wandb_runs_and_pastel_task_palette_are_pinned():
    assert assigned_literal("TARGET_RUNS") == {
        "LiftTray": ("dexmg-lifttray-final", "ih6hs4wt"),
        "Pouring": ("dexmg-pouring-final", "ftemqles"),
    }
    assert assigned_literal("TASK_COLORS") == {
        "ThreePiece": "#9DC3E6",
        "Threading": "#F2A7A7",
        "LiftTray": "#A8D5BA",
        "Pouring": "#F6E58D",
    }


def test_default_output_replaces_the_requested_figure():
    source = SOURCE.read_text()
    assert "fig_vanilla_collapse.pdf" in source
    assert "eval/success_rate" in source


def test_short_horizon_panel_uses_four_retained_tasks_without_the_annotation():
    source = SOURCE.read_text()
    assert "SHORT_TASK_RUNS" in source
    assert '"4/4 tasks → 0"' not in source
    assert "median of 4 tasks" in source
    assert "min–max band\\n(4 tasks)" in source


def test_short_horizon_aggregation_keeps_later_available_task_points():
    from paper.plot_vanilla_collapse import aggregate_short_histories

    steps, median, lower, upper = aggregate_short_histories({
        "Can": {0: 0.2, 10: 0.4},
        "Square": {0: 0.6},
    })

    assert steps == [0, 10]
    assert median == [0.4, 0.4]
    assert lower == [0.2, 0.4]
    assert upper == [0.6, 0.4]


def test_long_horizon_panel_is_a_pastel_red_median_band_at_300k():
    source = SOURCE.read_text()
    assert "long_steps, long_median, long_min, long_max" in source
    assert "color=\"#EFA3A3\"" in source
    assert "ax.set_xlim(0, 300)" in source
    assert "ax.set_xticks((0, 100, 200, 300))" in source


def test_long_horizon_panel_keeps_its_axis_labels_and_title():
    source = SOURCE.read_text()
    assert "style_axes(left)" in source
    assert "left.set_title(\"(a) Long-horizon: RL collapses\"" in source
    assert "left.set_xlabel(\"Environment steps (k)\"" in source
    assert "left.set_ylabel(\"Eval success rate\"" in source


def test_reporting_handles_dictionary_backed_histories():
    source = SOURCE.read_text()
    assert "points[-1]" not in source
    assert "max(points)" in source
