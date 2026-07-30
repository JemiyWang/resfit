"""Regenerate the vanilla-collapse motivation figure from pinned W&B runs."""
import argparse

from aaai_type1_matplotlib import configure_aaai_type1_matplotlib

configure_aaai_type1_matplotlib()
import matplotlib.pyplot as plt
import wandb


ENT = "674575221-beijing-institute-of-technology"
TARGET_RUNS = {
    "LiftTray": ("dexmg-lifttray-final", "ih6hs4wt"),
    "Pouring": ("dexmg-pouring-final", "ftemqles"),
}
# The two unchanged long-horizon references remain pinned to their original
# vanilla seed-1 runs; the two target entries above are the user-specified runs.
REFERENCE_RUNS = {
    "ThreePiece": ("dexmg-threepiece-final", "iqcf78ts"),
    "Threading": ("dexmg-twoarmthreading-final", "lnea5hv7"),
}
TASK_COLORS = {
    "ThreePiece": "#9DC3E6",  # light blue
    "Threading": "#F2A7A7",   # light red
    "LiftTray": "#A8D5BA",    # light green
    "Pouring": "#F6E58D",     # light yellow
}
TASK_ORDER = ("ThreePiece", "Threading", "LiftTray", "Pouring")

SHORT_TASK_RUNS = {
    "Can": ("robomimic-can-final", "z3p6eruh"),
    "Square": ("robomimic-square-final", "zo9kbn3v"),
    "CanSort": ("dexmg-cansorting-final", "swoei1ya"),
    "BoxCleanup": ("dexmg-box-clean-final", "on6leevr"),
}


def fetch_success_history(api, project, run_id):
    """Return all logged success-rate points, using W&B's dense sampled history."""
    run = api.run(f"{ENT}/{project}/{run_id}")
    records = run.history(samples=10_000, pandas=False)
    points = {
        int(record["_step"]): float(record["eval/success_rate"])
        for record in records
        if record.get("_step") is not None
        and record.get("eval/success_rate") is not None
    }
    if not points:
        raise RuntimeError(f"No eval/success_rate points found in {run.path}")
    return sorted((step / 1_000, value) for step, value in points.items())


def aggregate_short_histories(series_by_task):
    """Compute the per-step median and min--max envelope over retained tasks."""
    steps = sorted(set().union(*(set(points) for points in series_by_task.values())))
    median, lower, upper = [], [], []
    for step in steps:
        values = sorted(points[step] for points in series_by_task.values() if step in points)
        middle = len(values) // 2
        median.append(values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2)
        lower.append(values[0])
        upper.append(values[-1])
    return steps, median, lower, upper


def style_axes(ax):
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks((0, 100, 200, 300))
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    ax.grid(axis="y", color="#d9d9d4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11, colors="#555555")


def plot(output):
    api = wandb.Api(timeout=120)
    run_map = {**REFERENCE_RUNS, **TARGET_RUNS}
    long_series = {
        task: dict(fetch_success_history(api, *run_map[task]))
        for task in TASK_ORDER
    }
    long_steps, long_median, long_min, long_max = aggregate_short_histories(long_series)

    short_series = {
        task: dict(fetch_success_history(api, *run))
        for task, run in SHORT_TASK_RUNS.items()
    }
    short_steps, short_median, short_min, short_max = aggregate_short_histories(short_series)

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.35, 3.15), sharey=True)

    style_axes(left)
    left.set_title("(a) Long-horizon: RL collapses", fontsize=12, fontweight="bold", pad=6)
    left.set_xlabel("Environment steps (k)", fontsize=11)
    left.set_ylabel("Eval success rate", fontsize=11)

    left.fill_between(long_steps, long_min, long_max,
                      color="#EFA3A3", alpha=0.43, linewidth=0)
    left.plot(long_steps, long_median, color="#EFA3A3", linewidth=2.25,
              solid_capstyle="round")
    left.text(288, 1.01, "median of 4 tasks", ha="right", fontsize=10,
              fontweight="bold", color="#D96B6B")
    left.text(150, 0.34, "min–max band\n(4 tasks)", ha="center", va="center",
              fontsize=9.5, color="#555555")

    right.fill_between(short_steps, short_min, short_max,
                       color="#8fd5be", alpha=0.43, linewidth=0)
    right.plot(short_steps, short_median, color="#1fae78", linewidth=2.25,
               solid_capstyle="round")
    style_axes(right)
    right.set_title("(b) Short-horizon: RL stable", fontsize=12, fontweight="bold", pad=6)
    right.set_xlabel("Environment steps (k)", fontsize=11)
    right.text(288, 1.01, "median of 4 tasks", ha="right", fontsize=10,
               fontweight="bold", color="#16a571")
    right.text(150, 0.34, "min–max band\n(4 tasks)", ha="center", va="center",
               fontsize=9.5, color="#555555")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")
    for task, points in long_series.items():
        print(f"{task}: {len(points)} points through {max(points):.0f}k")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/mnt/mnt/data/resfit/paper/figure/fig_vanilla_collapse.pdf",
    )
    args = parser.parse_args()
    plot(args.output)


if __name__ == "__main__":
    main()
