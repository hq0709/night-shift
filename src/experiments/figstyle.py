"""Shared figure style. Every figure in the paper imports this, so the whole set
looks like it came from one place -- which is what a journal expects.

Two things here are submission requirements rather than taste:
  * pdf.fonttype = 42 -- IEEE rejects Type 3 fonts, which is what matplotlib
    embeds by default.
  * STIXGeneral -- metric-compatible with the Times that IEEEtran sets the body
    text in, so figure labels and body text are the same typeface at the same size.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL, TXT = 3.45, 7.16          # IEEEtran column and text width, inches

# Colourblind-safe (Okabe--Ito). RED = the manipulation bit, BLUE = the contrast.
RED, ORANGE, BLUE, SKY, GREEN, GREY = (
    "#B4451F", "#E08214", "#1F5B8C", "#5DA5DA", "#117733", "#7A7A7A")

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def panel(ax, letter, dx=-0.16, dy=1.04):
    """Bold (a)/(b)/(c) at the panel's upper left, the journal convention."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figs/{stem}.{ext}")
    plt.close(fig)
