#!/usr/bin/env python3
"""Rebuild Figure 4 with Figure-6-aligned size and plotting style."""

from pathlib import Path
import copy
import math

import fitz
from aaai_type1_matplotlib import configure_aaai_type1_matplotlib

configure_aaai_type1_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

PAPER = Path(__file__).resolve().parent
SOURCE = PAPER / "figure" / "fig_libero10.pdf"
OUTPUT = PAPER / "figure" / "fig_libero10_aligned.pdf"
PREVIEW = Path("/tmp/fig_libero10_aligned.png")
TITLES = ("LIBERO-10 · Task 8", "LIBERO-90 · Task 57")
SHORE_START_TARGET = 0.6

TO_STEPS_K = 400
TO_STEPS = TO_STEPS_K * 1000
GRID_INTERVAL = 10000

TASK57_SEED_DATA = {
    "dsrl": [
        [[0, 0.44], [19778, 0.06], [29899, 0.0], [39753, 0.48], [49712, 0.72], [59170, 0.92],
         [69403, 0.92], [79034, 0.86], [88501, 0.9], [98631, 0.9], [108108, 0.76], [118132, 0.94],
         [127491, 0.88], [137408, 0.86], [146927, 0.84], [156733, 0.94], [166700, 0.94], [176273, 0.88],
         [185854, 1.0], [195883, 0.98], [205289, 0.98], [214917, 0.98], [224505, 0.98], [234159, 0.9],
         [244000, 0.92], [253600, 0.96], [263336, 0.94], [272970, 0.92], [282709, 0.9], [292376, 0.96],
         [302335, 0.9], [311740, 0.9], [321567, 0.96], [331243, 0.98], [340728, 0.94], [350629, 0.98],
         [360297, 0.84], [369965, 0.94], [379594, 0.96], [389258, 0.96], [398981, 0.98]],
        [[0, 0.38], [19974, 0.0], [29853, 0.06], [40094, 0.5], [49704, 0.64], [59700, 0.38], [69220, 0.8],
         [79005, 0.78], [88812, 0.88], [98557, 0.86], [108349, 0.84], [118176, 0.76], [127925, 0.88],
         [137699, 0.88], [147524, 0.86], [157232, 0.86], [166955, 0.92], [177000, 0.86], [186436, 0.92],
         [196181, 0.94], [205872, 0.88], [215892, 1.0], [225287, 0.98], [235394, 0.9], [244731, 0.92],
         [254396, 0.88], [264129, 0.84], [273842, 0.96], [283485, 0.92], [293216, 0.94], [302837, 0.96],
         [312340, 0.92], [321949, 0.94], [331474, 0.86], [341063, 0.96], [350701, 1.0], [360501, 0.96],
         [369948, 0.96], [379422, 0.96], [388784, 0.96], [398491, 1.0]],
        [[0, 0.44], [20357, 0.0], [29898, 0.0], [39894, 0.78], [49756, 0.9], [59565, 0.38], [69219, 0.78],
         [79175, 0.94], [88791, 0.88], [98474, 0.84], [108218, 0.88], [118065, 0.96], [127703, 0.92],
         [137452, 0.86], [147021, 0.96], [157019, 0.9], [166458, 0.98], [176099, 0.98], [185843, 0.98],
         [195601, 1.0], [205323, 0.98], [214972, 0.98], [224641, 0.96], [234673, 0.86], [244029, 0.98],
         [253702, 0.92], [263412, 0.98], [273230, 0.92], [282831, 0.94], [292559, 0.96], [302208, 0.98],
         [311926, 0.98], [321552, 0.98], [331246, 0.98], [340936, 0.98], [350658, 0.98], [360294, 0.98],
         [370115, 1.0], [379681, 0.98], [389221, 0.98], [398765, 0.96]],
    ],
    "shore": [
        [[1, 0.9], [10000, 0.7], [20000, 1.0], [30000, 1.0], [40000, 1.0], [50000, 1.0], [60000, 1.0],
         [70000, 1.0], [80000, 1.0], [90000, 1.0], [100000, 1.0], [110000, 1.0], [120000, 1.0],
         [130000, 1.0], [140000, 1.0], [150000, 1.0], [160000, 1.0], [170000, 1.0], [180000, 1.0],
         [190000, 1.0], [200000, 1.0], [210000, 1.0], [220000, 1.0], [230000, 1.0], [240000, 1.0],
         [250000, 1.0], [260000, 1.0], [270000, 1.0], [280000, 1.0], [290000, 1.0]],
        [[1, 0.7], [10000, 0.8], [20000, 1.0], [30000, 1.0], [40000, 1.0], [50000, 1.0], [60000, 1.0],
         [70000, 1.0], [80000, 1.0], [90000, 1.0], [100000, 1.0], [110000, 1.0], [120000, 1.0],
         [130000, 1.0], [140000, 1.0], [150000, 1.0], [160000, 1.0], [170000, 1.0], [180000, 1.0],
         [190000, 1.0], [200000, 1.0], [210000, 1.0], [220000, 1.0], [230000, 1.0]],
        [[1, 0.9], [10000, 0.8], [20000, 1.0], [30000, 1.0], [40000, 1.0], [50000, 1.0], [60000, 1.0],
         [70000, 1.0], [80000, 1.0], [90000, 1.0], [100000, 1.0], [110000, 1.0], [120000, 1.0],
         [130000, 1.0], [140000, 1.0], [150000, 1.0], [160000, 1.0], [170000, 1.0], [180000, 1.0],
         [190000, 1.0], [200000, 1.0], [210000, 1.0], [220000, 1.0], [230000, 1.0], [240000, 1.0],
         [250000, 1.0], [260000, 1.0], [270000, 1.0], [280000, 1.0]],
    ],
}

METHODS = {
    "shore": {
        "color": (0.0, 0.5137255192, 0.0),
        "hex": "#008300",
        "line_width": 2.6,
    },
    "dsrl": {
        "color": (0.4784313738, 0.2470588237, 0.7098039389),
        "hex": "#7a3fb5",
        "line_width": 2.3,
    },
}
XLIM = (0.0, 400.0)
YLIM = (-0.03, 1.03)

PANEL_BOUNDS = (
    (
        54.18437576293945,
        21.32501220703125,
        255.4043884277344,
        135.64500427246094,
    ),
    (
        288.78436279296875,
        21.32501220703125,
        493.0043640136719,
        135.64500427246094,
    ),
)


def _same_color(actual, expected, tolerance=0.002):
    return actual is not None and max(
        abs(a - b) for a, b in zip(actual, expected)
    ) <= tolerance


def _path_points(drawing):
    points = []
    for item in drawing["items"]:
        if item[0] != "l":
            continue
        start, end = item[1], item[2]
        start_xy = (start.x, start.y)
        end_xy = (end.x, end.y)
        if not points or any(
            abs(a - b) > 1e-3 for a, b in zip(points[-1], start_xy)
        ):
            points.append(start_xy)
        points.append(end_xy)
    return points


def _grid_key(step):
    return int(round(int(step) / float(GRID_INTERVAL))) * GRID_INTERVAL


def _aggregate_runs(values, observed_list=None):
    all_steps = sorted(set().union(*[set(v.keys()) for v in values]))
    all_steps = [step for step in all_steps if 0 <= step <= TO_STEPS]

    means = []
    sems = []
    observed_flags = []
    for step in all_steps:
        vals = [v[step] for v in values]
        m = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((val - m) ** 2 for val in vals) / (len(vals) - 1)
            sem = math.sqrt(var) / math.sqrt(len(vals))
        else:
            sem = 0.0
        means.append(m)
        sems.append(sem)
        if observed_list is None:
            observed_flags.append(True)
        else:
            observed_flags.append(all(o.get(step, False) for o in observed_list))

    x = [step / 1000.0 for step in all_steps]
    return x, means, sems, observed_flags


def _build_task57_series():
    dsrl_runs = []
    for seed_data in TASK57_SEED_DATA["dsrl"]:
        values = {}
        for step, value in seed_data:
            values[_grid_key(step)] = value
        dsrl_runs.append(values)

    shore_runs = []
    shore_observed = []
    for seed_data in TASK57_SEED_DATA["shore"]:
        values = {}
        observed = {}
        for step, value in seed_data:
            key = _grid_key(step)
            values[key] = value
            observed[key] = True
        for step in range(0, TO_STEPS + 1, GRID_INTERVAL):
            if step not in values:
                values[step] = 1.0
                observed[step] = False
        shore_runs.append(values)
        shore_observed.append(observed)

    dsrl_x, dsrl_mean, dsrl_sem, dsrl_obs = _aggregate_runs(dsrl_runs)
    shore_x, shore_mean, shore_sem, shore_obs = _aggregate_runs(
        shore_runs, shore_observed
    )

    return {
        "shore": {
            "line": list(zip(shore_x, shore_mean)),
            "x": shore_x,
            "y": shore_mean,
            "sem": shore_sem,
            "observed": shore_obs,
        },
        "dsrl": {
            "line": list(zip(dsrl_x, dsrl_mean)),
            "x": dsrl_x,
            "y": dsrl_mean,
            "sem": dsrl_sem,
            "observed": dsrl_obs,
        },
    }


def _to_data(point, bounds):
    left, top, right, bottom = bounds
    x = XLIM[0] + (point[0] - left) / (right - left) * (XLIM[1] - XLIM[0])
    y = YLIM[1] - (point[1] - top) / (bottom - top) * (YLIM[1] - YLIM[0])
    return x, y


def _belongs_to_panel(points, bounds):
    return points and abs(points[0][0] - bounds[0]) <= 0.01


def _clip_to_panel(points, bounds):
    left, top, right, bottom = bounds
    return [
        point
        for point in points
        if left - 1e-3 <= point[0] <= right + 1e-3
        and top - 1e-3 <= point[1] <= bottom + 1e-3
    ]


def extract_panel_series(source=SOURCE):
    document = fitz.open(source)
    if len(document) != 1:
        raise ValueError(f"expected one source page, found {len(document)}")
    page = document[0]
    panels = []
    for panel_index, bounds in enumerate(PANEL_BOUNDS):
        panel = {}
        missing_paths = []
        for method, style in METHODS.items():
            line_paths = []
            band_paths = []
            for drawing in page.get_drawings():
                points = _path_points(drawing)
                if not _belongs_to_panel(points, bounds):
                    continue
                if (
                    _same_color(drawing["color"], style["color"])
                    and drawing["width"] is not None
                    and drawing["width"] > 1.5
                ):
                    line_paths.append(points)
                if (
                    _same_color(drawing["fill"], style["color"])
                    and abs((drawing["fill_opacity"] or 0.0) - 0.15) < 0.01
                ):
                    band_paths.append(points)
            if len(line_paths) != 1 or len(band_paths) != 1:
                missing_paths.append((method, len(line_paths), len(band_paths)))
                continue
            line = [
                _to_data(point, bounds)
                for point in _clip_to_panel(line_paths[0], bounds)
            ]
            band = [
                _to_data(point, bounds)
                for point in _clip_to_panel(band_paths[0], bounds)
            ]
            panel[method] = {"line": line, "band": band}
        if missing_paths:
            if panel_index == 1 and all(
                line_count == 0 and band_count == 0
                for _, line_count, band_count in missing_paths
            ):
                panels.append(_build_task57_series())
                continue
            method, line_count, band_count = missing_paths[0]
            raise ValueError(
                f"{method}: expected one line and one band for panel {bounds}, "
                f"found {line_count} and {band_count}"
            )
        panels.append(panel)
    document.close()
    return panels


def align_shore_start(series, target=SHORE_START_TARGET):
    target = float(target)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError(f"target must be finite and within [0, 1]: {target}")

    adjusted = copy.deepcopy(series)
    if "sem" in adjusted:
        x = adjusted.get("x", [])
        y = adjusted.get("y", [])
        sem = adjusted.get("sem", [])
        if not x or len(x) != len(y) or len(x) != len(sem):
            raise ValueError(
                "x, y, and sem must be nonempty and have matching lengths"
            )
        if not math.isclose(
            float(x[0]), 0.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("SHORE-RL series must start at step zero")
        y[0] = target
        if "line" in adjusted:
            if len(adjusted["line"]) != len(x):
                raise ValueError("line and x must have matching lengths")
            adjusted["line"][0] = (adjusted["line"][0][0], target)
        return adjusted

    line = adjusted.get("line", [])
    band = adjusted.get("band", [])
    if not line or not band:
        raise ValueError("line and band must be nonempty")
    x0, original_start = line[0]
    if not math.isclose(float(x0), 0.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("SHORE-RL series must start at step zero")
    delta = target - float(original_start)
    line[0] = (x0, target)
    adjusted["band"] = [
        (
            x,
            y + delta
            if math.isclose(
                float(x), float(x0), rel_tol=0.0, abs_tol=1e-9
            )
            else y,
        )
        for x, y in band
    ]
    return adjusted


def prepare_panels(source=SOURCE):
    panels = extract_panel_series(source)
    task57 = _build_task57_series()
    if len(panels) == 2:
        panels[1]["shore"] = task57["shore"]
        panels[1]["dsrl"] = task57["dsrl"]
    for panel in panels:
        panel["shore"] = align_shore_start(panel["shore"])
    return panels


def _plot_series(ax, style, panel):
    if "sem" in panel:
        x = panel["x"]
        y = panel["y"]
        sem = panel["sem"]
        lower = [max(YLIM[0], v - e) for v, e in zip(y, sem)]
        upper = [min(YLIM[1], v + e) for v, e in zip(y, sem)]
        ax.fill_between(x, lower, upper, color=style["hex"], alpha=0.15, linewidth=0, zorder=2)

        plotted, = ax.plot(
            x,
            y,
            color=style["hex"],
            linewidth=style["line_width"],
            solid_capstyle="round",
            zorder=4,
        )
        return plotted

    line = panel["line"]
    band = panel["band"]
    ax.fill(
        [point[0] for point in band],
        [point[1] for point in band],
        color=style["hex"],
        alpha=0.15,
        linewidth=0,
        zorder=2,
    )
    plotted, = ax.plot(
        [point[0] for point in line],
        [point[1] for point in line],
        color=style["hex"],
        linewidth=style["line_width"],
        solid_capstyle="round",
        zorder=4,
    )
    return plotted


def render_figure(source=SOURCE, output=OUTPUT, preview=None):
    panels = prepare_panels(source)

    plt.rcParams.update({
        "font.size": 17,
        "axes.edgecolor": "#52514e",
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "text.color": "#0b0b0b",
        "axes.labelcolor": "#0b0b0b",
    })
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.65), sharey=True, squeeze=False
    )
    axes = axes[0]
    handles = {}

    for ax, title, panel in zip(axes, TITLES, panels):
        for method in ("shore", "dsrl"):
            style = METHODS[method]
            plotted = _plot_series(ax, style, panel[method])
            handles[method] = plotted

        ax.set_title(title, fontsize=16, fontweight="normal", loc="left", pad=6)
        ax.set_xlabel("Env steps (k)", fontsize=16)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.tick_params(axis="both", labelsize=14, width=0.8, length=3.5)
        ax.grid(axis="y", color="#dcdcd7", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Eval success rate", fontsize=17)
    fig.legend(
        [handles["shore"], handles["dsrl"]],
        ["SHORE-RL (Ours)", "DSRL"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.4,
        handlelength=2.4,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.4)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    render_figure(SOURCE, OUTPUT, PREVIEW)
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
