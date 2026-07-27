"""Figure 5 component ablation summarized at a fixed 400k-step budget.

For each seed, the metric is the mean eval success over the eight checkpoints
with 320k < step <= 400k. Bars aggregate those seed-level values and show
 +/-1 s.e.m. when at least two seeds are available.
"""

import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


ENT = "674575221-beijing-institute-of-technology"
CR = "dexmg-chunk-residual"
BUDGET = 400_000
WINDOW_STEPS = tuple(range(330_000, BUDGET + 1, 10_000))
FIGSIZE_INCHES = (3.35, 2.30)
FONT_SIZE_PT = 7
BAR_EDGE_COLOR = "#4A4742"

STY = {
    "full": {
        "label": "SHORE-RL",
        "color": "#FAD35B",
    },
    "no_staged": {
        "label": "w/o stage shaping",
        "color": "#ABACAB",
    },
    "no_subgoal": {
        "label": "w/o waypoint",
        "color": "#CC8675",
    },
    "no_both": {
        "label": "w/o waypoint + stage",
        "color": "#80AECA",
    },
    "subgoal_only": {
        "label": "w/o demo-BC + stage",
        "color": "#539955",
    },
}

# Keep the current Figure 5 legend order.
LEGEND_ORDER = [
    "full",
    "subgoal_only",
    "no_staged",
    "no_subgoal",
    "no_both",
]

# Matplotlib fills multi-column legends by column. This handle order makes the
# visual row-major reading order match LEGEND_ORDER and the bars left-to-right.
LEGEND_HANDLE_ORDER = [
    arm
    for column in range(2)
    for arm in LEGEND_ORDER[column::2]
]

# Exact run selection used by paper/plot_ablation_curves.py.
PANELS = {
    "Pouring": {
        "full": [
            (CR, "3nsvrbob"),
            (CR, "b4yerjy2"),
            (CR, "yo3rvi0t"),
        ],
        "subgoal_only": [
            (CR, "02oot9gd"),
            (CR, "hgn6n19p"),
        ],
        "no_staged": [
            (CR, "s3padi0r"),
            (CR, "t788eu2h"),
        ],
        "no_subgoal": [
            (CR, "m0w28c01"),
            (CR, "7pwb2cf6"),
            (CR, "7005hpgz"),
        ],
        "no_both": [
            (CR, "9j5i5nnd"),
            (CR, "31fzc3uc"),
            (CR, "dma01plo"),
        ],
    },
    "ThreePiece": {
        "full": [
            (CR, "kmtsayff"),
            (CR, "m2s74dqm"),
            (CR, "cgtmwrv3"),
        ],
        "subgoal_only": [
            (CR, "w8y426z6"),
            (CR, "abfqmeox"),
            (CR, "zcii68rz"),
        ],
        "no_staged": [
            (CR, "15z3ijc6"),
            (CR, "uvrakupm"),
            (CR, "exu0ta17"),
        ],
        "no_subgoal": [
            (CR, "r50syxbq"),
            (CR, "3icsrj6r"),
            (CR, "ya51qauh"),
        ],
        "no_both": [
            (CR, "ihfya5hb"),
            (CR, "xvo0bumf"),
            (CR, "6t0y1m6d"),
        ],
    },
}


def gridkey(step):
    """Snap an evaluation step to the nearest 10k."""
    return int(round(step / 10_000.0)) * 10_000


def window_mean(points):
    """Return one seed's mean over the complete fixed 330k--400k window."""
    missing = [step for step in WINDOW_STEPS if step not in points]
    if missing:
        raise ValueError(
            "incomplete 400k final window; missing steps: "
            f"{missing}"
        )
    return statistics.fmean(points[step] for step in WINDOW_STEPS)


def summarize_seeds(seeds):
    """Aggregate seed-level fixed-window means as mean, s.e.m., and values."""
    values = [window_mean(seed) for seed in seeds]
    if not values:
        raise ValueError("at least one seed is required")
    sem = (
        statistics.stdev(values) / math.sqrt(len(values))
        if len(values) > 1
        else None
    )
    return statistics.fmean(values), sem, values


def pull_run(api, project, run_id):
    """Fetch and grid one W&B run's eval-success history."""
    run = api.run(f"{ENT}/{project}/{run_id}")
    history = run.history(
        keys=["eval/success_rate"],
        samples=10_000,
        pandas=False,
    )
    points = {}
    for row in history:
        value = row.get("eval/success_rate")
        if value is None:
            continue
        points[gridkey(int(row["_step"]))] = float(value)
    return points


def build_summaries(api):
    """Pull the selected inventory and compute fixed-window summaries."""
    summaries = {}
    for task, arms in PANELS.items():
        summaries[task] = {}
        print(f"\n[{task}]")
        for arm in LEGEND_ORDER:
            seeds = [
                pull_run(api, project, run_id)
                for project, run_id in arms[arm]
            ]
            mean, sem, values = summarize_seeds(seeds)
            summaries[task][arm] = (mean, sem, values)
            sem_text = f"{sem:.3f}" if sem is not None else "n/a"
            print(
                f"  {arm:13s} n={len(values)} "
                f"per-seed={[round(value, 4) for value in values]} "
                f"mean={mean:.3f} sem={sem_text}"
            )
    return summaries


def plot_summaries(summaries, out_pdf, out_png):
    """Render a one-column grouped bar chart for the two tasks."""
    ink = "#0b0b0b"
    muted = "#52514e"
    grid = "#dcdcd7"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": FONT_SIZE_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": muted,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "xtick.color": muted,
            "ytick.color": muted,
            "text.color": ink,
            "axes.labelcolor": ink,
        }
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
    task_names = list(PANELS)
    centers = list(range(len(task_names)))
    bar_width = 0.76 / len(LEGEND_ORDER)

    for arm_index, arm in enumerate(LEGEND_ORDER):
        offset = (
            arm_index - (len(LEGEND_ORDER) - 1) / 2
        ) * bar_width
        x_positions = [center + offset for center in centers]
        means = [summaries[task][arm][0] for task in task_names]
        values_by_task = [summaries[task][arm][2] for task in task_names]
        bars = ax.bar(
            x_positions,
            means,
            width=bar_width * 0.90,
            color=STY[arm]["color"],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.55,
            zorder=3,
        )
        for x_pos, task, bar, values in zip(
            x_positions,
            task_names,
            bars,
            values_by_task,
        ):
            mean, sem, _ = summaries[task][arm]
            if sem is not None:
                ax.errorbar(
                    x_pos,
                    mean,
                    yerr=sem,
                    fmt="none",
                    ecolor=BAR_EDGE_COLOR,
                    elinewidth=0.65,
                    capsize=1.6,
                    capthick=0.65,
                    zorder=5,
                )
            if len(values) == 1:
                bar.set_hatch("//")

    ax.set_xlim(-0.53, len(task_names) - 0.47)
    ax.set_ylim(0, 1.0)
    display_names = [
        "PieceAssembly" if task == "ThreePiece" else task
        for task in task_names
    ]
    ax.set_xticks(centers, display_names)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(
        axis="x",
        labelsize=FONT_SIZE_PT,
        width=0.55,
        length=2.5,
        pad=2,
    )
    ax.tick_params(
        axis="y",
        labelsize=FONT_SIZE_PT,
        width=0.55,
        length=2.5,
        pad=2,
    )
    ax.grid(axis="y", color=grid, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Success rate", fontsize=FONT_SIZE_PT, labelpad=3)

    legend_handles = [
        Patch(
            facecolor=STY[arm]["color"],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.45,
            label=STY[arm]["label"],
        )
        for arm in LEGEND_HANDLE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_PT,
        bbox_to_anchor=(0.05, 0.015, 0.92, 0.26),
        mode="expand",
        borderaxespad=0,
        columnspacing=0.8,
        handlelength=1.25,
        handletextpad=0.4,
        labelspacing=0.35,
    )
    fig.subplots_adjust(
        left=0.17,
        right=0.985,
        top=0.985,
        bottom=0.34,
    )

    out_pdf = Path(out_pdf)
    out_png = Path(out_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    import wandb

    api = wandb.Api(timeout=120)
    summaries = build_summaries(api)
    figure_dir = Path(__file__).with_name("figure")
    out_pdf = figure_dir / "fig_ablation_bars_400k.pdf"
    out_png = figure_dir / "fig_ablation_bars_400k.png"
    plot_summaries(summaries, out_pdf, out_png)
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
