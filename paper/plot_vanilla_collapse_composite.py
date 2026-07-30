"""Create the composite vanilla-collapse figure with task thumbnails.

The three task images form a bordered strip on the left.  The long- and
short-horizon aggregates share the single axes on the right.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aaai_type1_matplotlib import configure_aaai_type1_matplotlib

configure_aaai_type1_matplotlib()
import matplotlib.pyplot as plt
import wandb

from plot_vanilla_collapse import (
    REFERENCE_RUNS,
    SHORT_TASK_RUNS,
    TARGET_RUNS,
    TASK_ORDER,
    aggregate_short_histories,
    fetch_success_history,
)


PAPER_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PAPER_DIR / "figure" / "fig_vanilla_collapse.pdf"
TASK_IMAGES = (
    (PAPER_DIR / "figure" / "can0.png", "#5DB8D2"),
    (PAPER_DIR / "figure" / "can1.png", "#5DB8D2"),
    (PAPER_DIR / "figure" / "can2.png", "#5DB8D2"),
    (PAPER_DIR / "figure" / "piece0.png", "#F0A06F"),
    (PAPER_DIR / "figure" / "piece1.png", "#F0A06F"),
    (PAPER_DIR / "figure" / "piece2.png", "#F0A06F"),
)

LONG_COLOR = "#EFA3A3"
LONG_LINE_COLOR = "#D96B6B"
SHORT_COLOR = "#A8D5BA"
SHORT_LINE_COLOR = "#5EAE7B"
LONG_LINE_WIDTH = 2.7
SHORT_LINE_WIDTH = 3.0
AXIS_COLOR = "#52514E"
GRID_COLOR = "#DCDCD7"
CURVE_TICK_SIZE = 14
CURVE_XLABEL_SIZE = 16
CURVE_YLABEL_SIZE = 17
CURVE_LEGEND_SIZE = 14

PAPER_TEXT_SIZE = 20


def style_curve_axes(ax):
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks((0, 100, 200, 300))
    ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    ax.set_xlabel("Env steps (k)", fontsize=CURVE_XLABEL_SIZE)
    ax.set_ylabel("Eval success rate", fontsize=CURVE_YLABEL_SIZE)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=CURVE_TICK_SIZE, width=0.8, length=3.5, colors=AXIS_COLOR)

def add_task_image(ax, image_path, border_color):
    ax.imshow(plt.imread(image_path))
    ax.set_xticks(())
    ax.set_yticks(())
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(2.0)


def add_panel_tag(ax, label, color, x=0.02, y=0.98):
    centered_label = rf"\vphantom{{b}}{label}"
    ax.text(x, y, centered_label, transform=ax.transAxes, ha="left", va="top",
            fontsize=PAPER_TEXT_SIZE, color=color, zorder=5,
            bbox=dict(boxstyle="circle,pad=0.16", facecolor="white",
                      edgecolor=color, linewidth=1.5))


def fetch_histories(run_map):
    """Fetch independent pinned histories concurrently for a fast, reproducible render."""

    def fetch_one(item):
        task, (project, run_id) = item
        api = wandb.Api(timeout=120)
        return task, dict(fetch_success_history(api, project, run_id))

    with ThreadPoolExecutor(max_workers=len(run_map)) as executor:
        return dict(executor.map(fetch_one, run_map.items()))


def main():
    long_histories = fetch_histories({**REFERENCE_RUNS, **TARGET_RUNS})
    short_histories = fetch_histories(SHORT_TASK_RUNS)

    long_steps, long_median, long_low, long_high = aggregate_short_histories(
        {task: long_histories[task] for task in TASK_ORDER}
    )
    short_steps, short_median, short_low, short_high = aggregate_short_histories(
        short_histories
    )

    fig = plt.figure(figsize=(10.8, 4.35))
    main_grid = fig.add_gridspec(
        1, 2, width_ratios=(2.66, 2.66), left=0.035, right=0.99,
        bottom=0.15, top=0.91, wspace=0.20,
    )
    image_grid = main_grid[0, 0].subgridspec(2, 3, wspace=0.0, hspace=0.08)

    image_axes = []
    for index, (image_path, border_color) in enumerate(TASK_IMAGES):
        row, column = divmod(index, 3)
        image_ax = fig.add_subplot(image_grid[row, column])
        add_task_image(image_ax, image_path, border_color)
        image_axes.append(image_ax)

    add_panel_tag(image_axes[0], "a", "#5DB8D2", x=0.08, y=0.95)
    add_panel_tag(image_axes[3], "b", "#F0A06F", x=0.08, y=0.95)

    curve_ax = fig.add_subplot(main_grid[0, 1])
    add_panel_tag(curve_ax, "c", "#5EAE7B", x=0.07)
    curve_ax.fill_between(long_steps, long_low, long_high, color=LONG_COLOR, alpha=0.15)
    curve_ax.plot(
        long_steps, long_median, color=LONG_LINE_COLOR, linewidth=LONG_LINE_WIDTH,
        label="Long horizon",
    )
    curve_ax.fill_between(short_steps, short_low, short_high, color=SHORT_COLOR, alpha=0.15)
    curve_ax.plot(
        short_steps, short_median, color=SHORT_LINE_COLOR, linewidth=SHORT_LINE_WIDTH,
        label="Short horizon",
    )
    style_curve_axes(curve_ax)
    curve_ax.legend(
        loc="center right",
        ncol=1,
        frameon=False,
        handlelength=2.4,
        prop={"size": CURVE_LEGEND_SIZE},
    )

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
