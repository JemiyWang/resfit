"""Plot the real-robot success-rate curves used in Section 5.3."""

from pathlib import Path

from aaai_type1_matplotlib import configure_aaai_type1_matplotlib

configure_aaai_type1_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


STEPS = [0, 100, 200, 300, 400, 500]
DATA = {
    "paper": {
        "SHORE-RL": [22, 18, 30, 38, 44, 42],
        "ResFit": [22, 18, 22, 26, 28, 28],
    },
    "cup": {
        "SHORE-RL": [12, 10, 24, 32, 30, 32],
        "ResFit": [12, 8, 10, 6, 2, 2],
    },
    "block": {
        "SHORE-RL": [10, 6, 16, 24, 30, 28],
        "ResFit": [10, 4, 0, 2, 0, 0],
    },
}
TASK_ORDER = ("paper", "block", "cup")
TASK_LABELS = {
    "paper": "Paper-Roll placement",
    "block": "Block assembly",
    "cup": "Cup stacking",
}
SOURCE_PDF_WIDTH_PT = 840
SOURCE_PDF_HEIGHT_PT = 310
SOURCE_FIGSIZE = (
    SOURCE_PDF_WIDTH_PT / 72.0,
    SOURCE_PDF_HEIGHT_PT / 72.0,
)
AAAI_TEXT_WIDTH_IN = 7.0
AAAI_COLUMN_SEP_IN = 0.375
SINGLE_COLUMN_WIDTH_IN = (
    AAAI_TEXT_WIDTH_IN - AAAI_COLUMN_SEP_IN
) / 2
TARGET_ON_PAGE_BASE_FONT_SIZE = 8.0
ORIGINAL_BASE_FONT_SIZE = 17.0
FONT_COMPENSATION = (
    TARGET_ON_PAGE_BASE_FONT_SIZE
    / ORIGINAL_BASE_FONT_SIZE
    * (SOURCE_FIGSIZE[0] / SINGLE_COLUMN_WIDTH_IN)
)
TARGET_SOURCE_FONT_SIZE = ORIGINAL_BASE_FONT_SIZE * FONT_COMPENSATION * 0.9
MARKER_SIZE = 7.5
MARKER_EDGE_WIDTH = 1.4
METHOD_STYLES = {
    "SHORE-RL": {"color": "#008300", "ls": "-", "lw": 2.6, "marker": "o"},
    "ResFit": {
        "color": "#7a3fb5",
        "ls": (0, (4.5, 1.2, 1, 1.2)),
        "lw": 2.3,
        "marker": "s",
    },
}
FROZEN_STYLE = {
    "color": "#5f5f5b",
    "ls": (0, (1.5, 2.2)),
    "lw": 1.6,
}

save_kwargs = {
      "bbox_inches": "tight",
      "pad_inches": 0,
  }

def compensated_font_size(original_size):
    return original_size * FONT_COMPENSATION


def main():
    out_dir = Path(__file__).parent / "figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": TARGET_SOURCE_FONT_SIZE,
            "axes.edgecolor": "#52514e",
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "text.color": "#0b0b0b",
            "axes.labelcolor": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=SOURCE_FIGSIZE, sharey=True, squeeze=False
    )
    axes = axes[0]
    handles = {}
    for ax, task in zip(axes, TASK_ORDER):
        handles["Frozen base"] = ax.axhline(
            DATA[task]["SHORE-RL"][0] / 100.0,
            color=FROZEN_STYLE["color"],
            linestyle=FROZEN_STYLE["ls"],
            linewidth=FROZEN_STYLE["lw"],
            zorder=1,
        )
        for method in ("SHORE-RL", "ResFit"):
            style = METHOD_STYLES[method]
            line, = ax.plot(
                STEPS,
                [value / 100.0 for value in DATA[task][method]],
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                marker=style["marker"],
                markersize=MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=MARKER_EDGE_WIDTH,
                solid_capstyle="round",
            )
            handles[method] = line
        ax.set_title(
            TASK_LABELS[task],
            fontsize=TARGET_SOURCE_FONT_SIZE,
            fontweight="normal",
            loc="left",
            pad=6,
        )
        ax.set_xlabel("Env steps (k)", fontsize=TARGET_SOURCE_FONT_SIZE*0.85)
        ax.set_xlim(0, 500)
        ax.set_ylim(0.0, 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.tick_params(axis="both", labelsize=TARGET_SOURCE_FONT_SIZE, width=0.8, length=3.5)
        ax.grid(axis="y", color="#dcdcd7", linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Eval success rate", fontsize=TARGET_SOURCE_FONT_SIZE)
    fig.legend(
        [handles["SHORE-RL"], handles["ResFit"], handles["Frozen base"]],
        ["SHORE-RL (Ours)", "ResFit", "Frozen base"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=TARGET_SOURCE_FONT_SIZE*0.85,
        bbox_to_anchor=(0.5, -0.04),
        columnspacing=1.4,
        handlelength=2.4,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=0.8)
    fig.savefig(out_dir / "fig_real_robot_curves.pdf", **save_kwargs)
    fig.savefig(
        out_dir / "fig_real_robot_curves.png", dpi=240,
        **save_kwargs,
    )


if __name__ == "__main__":
    main()
