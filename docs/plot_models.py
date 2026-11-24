"""Regenerate docs/model_comparison.png from the numbers in README.md.

usage (repo root): python docs/plot_models.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

MODELS = [  # name, cAP clean, cAP RGB off, cAP events off (SkyEV val, 1280x736)
    ("Fusion, SkyEV + FRED (recommended)", 0.981, 0.953, 0.836),
    ("Fusion, SkyEV only", 0.982, 0.957, 0.830),
    ("Fusion, presence-first", 0.985, 0.965, 0.852),
    ("Fusion, presence-first + fine-tune", 0.984, 0.972, 0.859),
    ("RGB only", 0.883, 0.0, 0.883),
]
SERIES = [("clean", "#2a78d6"), ("RGB off", "#eb6834"), ("events off", "#1baf7a")]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n, k = len(MODELS), len(SERIES)
    h = 0.24
    y0 = np.arange(n)[::-1] * 1.0
    for j, (label, color) in enumerate(SERIES):
        ys = y0 + (1 - j) * (h + 0.02)
        vals = [m[1 + j] for m in MODELS]
        ax.barh(ys, vals, height=h, color=color, label=label, edgecolor=SURFACE, linewidth=0.5)
        for y, v in zip(ys, vals):
            ax.text(v + 0.006 if v > 0 else 0.006, y, f"{v:.3f}" if v > 0 else "0", va="center", ha="left", fontsize=7.5, color=INK2)
    ax.set_yticks(y0)
    ax.set_yticklabels([m[0] for m in MODELS], color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("cAP on the SkyEV validation split (prediction center inside the true box)", color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK2)
    ax.legend(loc="lower right", frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    out = Path(__file__).with_name("model_comparison.png")
    fig.savefig(out, facecolor=SURFACE)
    print(f"written {out}")


if __name__ == "__main__":
    main()
