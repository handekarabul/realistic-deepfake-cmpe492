"""Report figures for CMPE 492. Run: python generate_figures.py (from report/)"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# CV-style defaults: clean, minimal, readable in LaTeX
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# Muted academic palette (not flashy)
COLOR_A = "#4C6A8A"      # Acc@0.5
COLOR_B = "#8A8A8A"      # Bal.Acc@Opt
COLOR_CHANCE = "#9A9A9A"
BAR_EDGE = "#333333"
BAR_LW = 0.6


def _style_axes(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, alpha=0.9)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#333333", length=4, width=0.8)


def _annotate_bars(ax, bars, fontsize=11, offset=0.025):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def plot_threshold_comparison():
    conditions = ["FF-F2F", "FF-DF", "WCE-strong", "WCE-mild", "Celeb-DF-v2"]
    acc_05 = [0.50, 0.50, 0.50, 0.50, 0.657]
    bal_opt = [0.594, 0.844, 0.719, 0.594, 0.508]

    x = np.arange(len(conditions))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    bars1 = ax.bar(
        x - width / 2,
        acc_05,
        width,
        label=r"Acc@0.5",
        color=COLOR_A,
        edgecolor=BAR_EDGE,
        linewidth=BAR_LW,
        zorder=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        bal_opt,
        width,
        label=r"Bal. Acc@Opt",
        color=COLOR_B,
        edgecolor=BAR_EDGE,
        linewidth=BAR_LW,
        zorder=3,
    )

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Evaluation condition")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.axhline(
        0.5,
        color=COLOR_CHANCE,
        linestyle=(0, (4, 4)),
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
        label="Chance (0.50)",
    )

    _style_axes(ax)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fancybox=False,
    )
    _annotate_bars(ax, bars1, fontsize=11)
    _annotate_bars(ax, bars2, fontsize=11)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.96)
    out = FIG_DIR / "fig_threshold_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_auc_comparison():
    conditions = [
        "Scratch\nFF++",
        "Pretrained\nFF++",
        "FF-F2F",
        "FF-DF",
        "WCE-strong",
        "WCE-mild",
        "Celeb-DF-v2",
    ]
    auc = [0.550, 0.526, 0.547, 0.793, 0.701, 0.566, 0.474]

    # Subtle grayscale emphasis: FF-DF darkest; Celeb distinct but not loud
    colors = [
        "#D4D4D4",
        "#D4D4D4",
        "#A8A8A8",
        "#3D3D3D",
        "#B8B8B8",
        "#B8B8B8",
        "#6E6E6E",
    ]
    hatches = ["", "", "", "", "", "", "///"]

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    bars = ax.bar(
        conditions,
        auc,
        color=colors,
        edgecolor=BAR_EDGE,
        linewidth=BAR_LW,
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_ylabel("Frame-level AUC")
    ax.set_xlabel("Evaluation condition")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.axhline(
        0.5,
        color=COLOR_CHANCE,
        linestyle=(0, (4, 4)),
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
        label="Chance (0.50)",
    )

    _style_axes(ax)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fancybox=False,
    )
    _annotate_bars(ax, bars, fontsize=11)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.96)
    out = FIG_DIR / "fig_auc_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_threshold_comparison()
    plot_auc_comparison()
    print("Done.")
