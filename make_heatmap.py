#!/usr/bin/env python3
"""Heatmap summary: offered load, packet loss and delay for the four reported arms,
one row of panels per congestion regime.

Sources are the same two files the tables use, so nothing here is a separate
measurement:

  results/offered_grid.json    analytical bottleneck offered load
  results/final_ns3_grid.json  packet-level loss and delay

Each metric gets its own hue, light -> dark, and each panel is normalised WITHIN
each column, so shading reads as "how much worse than the best arm on this
network". A single scale per metric would be useless: overload delay spans 17 ms on
GEANT to 128 ms on Abilene, which would flatten every within-network difference.
Absolute values are printed in every cell, so colour never carries a number alone.

Hues are the project's categorical anchors and clear the CVD checks as a trio:
normal-vision dE >= 24.9, protanopia/deuteranopia >= 16.9.

  python make_heatmap.py --out paper/images
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ARMS = [("ospf", "OSPF"), ("ecmp", "ECMP"),
        ("single", "Single-agent"), ("marlh32", "MARL")]
TOPOS = [("abilene", "Abilene"), ("geant", "GEANT"), ("germany50", "Germany50")]
REGIMES = ["overload", "feasible"]

INK, MUTED = "#1A1F24", "#5C6A75"


def nice_scale(lo, hi, target=4):
    """Round the range outward to a 1/2/5 step, so the bar runs from a labelled
    bottom to a labelled top and every label ends in 0 or 5.
    Returns (lo, hi, ticks, format)."""
    span = hi - lo
    if span <= 0:
        return lo, lo + 1, [lo], "%g"
    mag = 10 ** np.floor(np.log10(span / target))
    step = next(s * mag for s in (1, 2, 5, 10) if s * mag >= span / target)
    lo_r = np.floor(lo / step) * step
    hi_r = np.ceil(hi / step) * step
    ticks = [float(v) for v in np.arange(lo_r, hi_r + step * 1e-9, step)]
    dec = max(0, int(-np.floor(np.log10(step)))) if step < 1 else 0
    return float(lo_r), float(hi_r), ticks, f"%.{dec}f"


def ramp(light, anchor, dark):
    """Start at a faint tint, not pure white, so the best cell still reads as a cell."""
    return LinearSegmentedColormap.from_list("r", [light, anchor, dark])


# one hue per metric: (key, source, field, title, format, ramp)
METRICS = [
    ("offered", "off", "mean", "Maximum offered load (%)", "%.1f",
     ramp("#EDF0FA", "#3B49B8", "#1B2266")),
    ("loss", "ns3", "loss_pct", "Packet loss (%)", "%.2f",
     ramp("#FBF0E4", "#D9730D", "#7A3F05")),
    ("delay", "ns3", "delay_ms", "Mean delay (ms)", "%.1f",
     ramp("#E6F5F4", "#14A8A0", "#0A5A56")),
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper/images")
A = ap.parse_args()
OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "text.color": INK,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})

SRC = {"off": json.loads(Path("results/offered_grid.json").read_text()),
       "ns3": json.loads(Path("results/final_ns3_grid.json").read_text())}


def matrix(source, field, regime):
    """[arm, topology] values for one regime; NaN where an arm is unmeasured."""
    M = np.full((len(ARMS), len(TOPOS)), np.nan)
    for r, (arm, _) in enumerate(ARMS):
        for c, (topo, _) in enumerate(TOPOS):
            rec = SRC[source].get(f"{topo}/{regime}", {}).get(arm)
            if rec is not None:
                M[r, c] = rec[field]
    return M


fig, axes = plt.subplots(len(REGIMES), len(METRICS), figsize=(10.0, 4.6))
for i, regime in enumerate(REGIMES):
    for j, (_, source, field, title, fmt, cmap) in enumerate(METRICS):
        ax = axes[i, j]
        M = matrix(source, field, regime)

        # normalise over the whole panel, against a range rounded outward to nice
        # bounds, so the scale bar starts and ends on a labelled tick
        lo, hi, ticks, tfmt = nice_scale(float(np.nanmin(M)), float(np.nanmax(M)))
        N = (M - lo) / (hi - lo)

        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                if np.isnan(M[r, c]):
                    continue
                ax.add_patch(plt.Rectangle((c - 0.47, r - 0.47), 0.94, 0.94,
                                           facecolor=cmap(N[r, c]),
                                           edgecolor="white", lw=1.2))
                ink = "white" if N[r, c] > 0.58 else INK
                ax.text(c, r, fmt % M[r, c], ha="center", va="center",
                        fontsize=8, color=ink)

        ax.set_xlim(-0.6, len(TOPOS) - 0.4)
        ax.set_ylim(len(ARMS) - 0.5, -0.5)          # first arm at the top
        ax.set_xticks(range(len(TOPOS)))
        ax.set_xticklabels([t[1] for t in TOPOS] if i == len(REGIMES) - 1 else [],
                           rotation=22, ha="right", color=MUTED)
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([a[1] for a in ARMS] if j == 0 else [], color=MUTED)
        if i == 0:
            ax.set_title(title, color=INK, pad=8)
        if j == 0:
            ax.text(-0.62, 0.5, regime.capitalize(), transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=10,
                    color=INK, style="italic")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

        # numeric scale strip, ticked at the panel's own range
        cb = ax.inset_axes([1.04, 0.06, 0.055, 0.88])
        cb.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect="auto", cmap=cmap,
                  extent=[0, 1, lo, hi], origin="lower")
        cb.set_xticks([])
        cb.yaxis.tick_right()
        cb.set_yticks(ticks)
        cb.set_yticklabels([tfmt % v for v in ticks], fontsize=6.5, color=MUTED)
        cb.tick_params(length=0, pad=2)
        for s in cb.spines.values():
            s.set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "fig_heatmap_summary.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'fig_heatmap_summary.png'}")
