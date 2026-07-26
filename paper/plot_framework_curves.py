"""Generate the two compact evaluation curves used in the framework figure."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "figure"
X = np.linspace(0.0, 500.0, 101)


def build_curve_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic failure/success means and confidence spreads."""
    failure = (
        0.10
        + 0.15 * np.exp(-X / 125.0)
        + 0.018 * np.exp(-X / 420.0) * np.sin(X / 16.0)
        + 0.009 * np.sin(X / 5.5)
    )
    failure_spread = 0.035 + 0.070 * np.exp(-X / 240.0)

    success = (
        0.95
        - 0.70 * np.exp(-X / 135.0)
        + 0.020 * np.exp(-X / 420.0) * np.sin(X / 17.0)
        + 0.010 * np.sin(X / 6.0)
    )
    success_spread = 0.035 + 0.065 * np.exp(-X / 250.0)

    return (
        np.clip(failure, 0.0, 1.0),
        failure_spread,
        np.clip(success, 0.0, 1.0),
        success_spread,
    )


def render_curve(
    output_path: Path,
    mean: np.ndarray,
    spread: np.ndarray,
    color: str,
) -> None:
    """Render one compact, publication-ready evaluation curve."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "svg.fonttype": "none",
            "svg.hashsalt": "resfit-framework-curves",
        }
    )

    fig, ax = plt.subplots(figsize=(4.5, 2.0), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    lower = np.clip(mean - spread, 0.0, 1.0)
    upper = np.clip(mean + spread, 0.0, 1.0)
    ax.fill_between(X, lower, upper, color=color, alpha=0.20, linewidth=0)
    ax.plot(
        X,
        mean,
        color=color,
        linewidth=2.2,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    ax.set_xlim(0.0, 500.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 501.0, 100.0))
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel("Env steps (k)", fontsize=14, labelpad=3)
    ax.set_ylabel("Eval success rate", fontsize=14, labelpad=4)

    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(1.4)
    ax.tick_params(
        axis="both",
        color="#333333",
        labelcolor="#333333",
        labelsize=12,
        width=1.4,
        length=4,
        direction="out",
        pad=2,
    )

    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.29, top=0.81)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_path.suffix.lstrip(".").lower()
    metadata = {"Date": None} if output_format == "svg" else None
    fig.savefig(
        output_path,
        format=output_format,
        facecolor="white",
        metadata=metadata,
    )
    if output_format == "svg":
        svg = output_path.read_text(encoding="utf-8")
        normalized = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
        output_path.write_text(normalized, encoding="utf-8")
    plt.close(fig)


def main() -> None:
    failure, failure_spread, success, success_spread = build_curve_data()
    render_curve(
        OUTPUT_DIR / "framework_curve_failure.svg",
        failure,
        failure_spread,
        "#F6C1C8",
    )
    render_curve(
        OUTPUT_DIR / "framework_curve_success.svg",
        success,
        success_spread,
        "#46A253",
    )


if __name__ == "__main__":
    main()
