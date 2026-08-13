#!/usr/bin/env python3
"""Regenerate every figure in the results chapter from the reported data.

Sources, and nothing else:
  results/offered_grid.json    analytical bottleneck offered load (all six cells)
  results/final_ns3_grid.json  packet-level loss / delay / utilisation
  logs/train_marlh32cm_s*.log  MARL h=32 training trace

MARL h=64 is the sensitivity arm and is deliberately NOT plotted; the figures show
the four arms the chapter argues over (OSPF, ECMP, single-agent, MARL h=32).

Palette is fixed and validated for colour-vision deficiency: all six pairs clear
dEOKLab >= 8 under simulated protanopia and deuteranopia, with a lightness spread of
0.37-0.66 so the figures survive greyscale printing.

  python make_figures.py --out paper/images
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [("ospf", "OSPF", "#37424C"),
        ("ecmp", "ECMP", "#14A8A0"),
        ("single", "Single-agent", "#D9730D"),
        ("marlh32", "MARL", "#3B49B8")]
TOPOS = [("abilene", "Abilene"),
         ("geant", "GÉANT"),
         ("germany50", "Germany50")]
INK, MUTED, GRID = "#1A1F24", "#5C6A75", "#DDE3E7"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="paper/images")
A = ap.parse_args()
OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def grouped(ax, cells, get, ylabel, capline=None, logy=False):
    """One panel: three topologies on x, four arms per topology."""
    n = len(ARMS)
    width = 0.8 / n
    for k, (key, label, colour) in enumerate(ARMS):
        xs, ys = [], []
        for i, (topo, _) in enumerate(TOPOS):
            v = get(cells, topo, key)
            if v is None:
                continue
            xs.append(i - 0.4 + width * (k + 0.5))
            ys.append(v[0])
        # 2px surface gap between adjacent bars, per the mark spec
        ax.bar(xs, ys, width * 0.88, color=colour, label=label, zorder=3,
               linewidth=0.6, edgecolor="white")
        # seed dispersion is carried by the appendix tables, not drawn here
    if capline is not None:
        # identified in the legend rather than annotated in-plot: at every in-plot
        # position the label collided with a bar
        ax.axhline(capline, color="#B3352B", lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_xticks(range(len(TOPOS)))
    ax.set_xticklabels([t[1] for t in TOPOS])
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, 2.55)


def two_panel(fname, get, ylabel, title_l, title_r, capline=None, logy=False):
    off = json.loads(Path("results/offered_grid.json").read_text())
    ns3 = json.loads(Path("results/final_ns3_grid.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, regime, ttl in zip(axes, ["overload", "feasible"], [title_l, title_r]):
        # the reference line is informative only where values can cross it; in the
        # feasible regime every bar is below capacity by the regime's definition
        grouped(ax, (off, ns3, regime), get, ylabel if ax is axes[0] else "",
                capline=capline if regime == "overload" else None, logy=logy)
        ax.set_title(ttl, color=INK, pad=6)
    h, l = axes[0].get_legend_handles_labels()
    if capline is not None:
        from matplotlib.lines import Line2D
        h.append(Line2D([0], [0], color="#B3352B", lw=1.0, ls=(0, (4, 3))))
        l.append("link capacity")
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT/fname}")


# ---- accessors -------------------------------------------------------------
def get_offered(cells, topo, arm):
    off, _, regime = cells
    r = off.get(f"{topo}/{regime}", {}).get(arm)
    return (r["mean"], r["sd"]) if r else None


def _ns3(cells, topo, arm, field, sd):
    _, ns3, regime = cells
    r = ns3.get(f"{topo}/{regime}", {}).get(arm)
    return (r[field], r[sd]) if r else None


def get_loss(c, t, a):    return _ns3(c, t, a, "loss_pct", "loss_sd")
def get_delay(c, t, a):   return _ns3(c, t, a, "delay_ms", "delay_sd")
def get_util(c, t, a):    return _ns3(c, t, a, "maxutil_pct", "maxutil_sd")


def training_curve():
    """MARL h=32 episode return against policy update, three seeds plus the mean."""
    runs = []
    # every seed that has a log, not a fixed three -- the reported seed count changed
    # from 3 to 10 and a hardcoded range silently left this figure stale
    for f in sorted(Path("logs").glob("train_marlh32cm_s*.log"),
                    key=lambda p: int(re.search(r"_s(\d+)", p.name).group(1))):
        if not f.exists():
            continue
        upd, ret = [], []
        for line in f.read_text().splitlines():
            m = re.search(r"upd\s+(\d+)/\d+\s+ep_ret~(-?[\d.]+)", line)
            if m:
                upd.append(int(m.group(1))); ret.append(float(m.group(2)))
        if upd:
            runs.append((np.array(upd), np.array(ret)))
    if not runs:
        print("  [skip] no MARL training logs found"); return

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    grid = runs[0][0]
    stack = np.vstack([np.interp(grid, u, r) for u, r in runs])

    # Each episode samples one of the 17 training topologies, whose bottlenecks differ
    # by an order of magnitude, so the per-update return is dominated by which topology
    # was drawn. A centred rolling mean over 15 updates removes that sampling noise and
    # leaves the learning trend; the raw seed mean stays visible underneath.
    W = 15
    def smooth(y):
        k = np.ones(W) / W
        pad = np.r_[np.full(W // 2, y[0]), y, np.full(W // 2, y[-1])]
        return np.convolve(pad, k, mode="valid")[:len(y)]

    mean = stack.mean(0)
    sm = np.vstack([smooth(stack[i]) for i in range(stack.shape[0])])
    ax.plot(grid, mean, color=ARMS[3][2], lw=0.7, alpha=0.22,
            label="seed mean (raw)", zorder=2)
    ax.fill_between(grid, sm.min(0), sm.max(0), color=ARMS[3][2], alpha=0.16,
                    lw=0, label="seed range (smoothed)", zorder=1)
    ax.plot(grid, sm.mean(0), color=ARMS[3][2], lw=1.9,
            label="seed mean (smoothed)", zorder=3)

    ax.set_xlabel("policy update")
    ax.set_ylabel("mean episode return")
    ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_xlim(0, grid.max())
    ax.annotate(f"{sm.mean(0)[-1]:.0f}", (grid[-1], sm.mean(0)[-1]), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=8, color=ARMS[3][2], fontweight="bold")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    ax.set_title("MARL ($h{=}32$) training return on the 17 training topologies",
                 color=INK, pad=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_marl_training_curves.png")
    plt.close(fig)
    print(f"  wrote {OUT/'fig_marl_training_curves.png'}")


if __name__ == "__main__":
    two_panel("fig_offered_util_regime.png", get_offered,
              "max offered load (%)", "Overload regime", "Feasible regime",
              capline=100)
    two_panel("fig_qos_loss_regime.png", get_loss,
              "packet loss (%)", "Overload regime", "Feasible regime")
    two_panel("fig_qos_delay_regime.png", get_delay,
              "mean delay (ms)", "Overload regime", "Feasible regime")
    two_panel("fig_ns3_util_regime.png", get_util,
              "max link utilisation (%)", "Overload regime", "Feasible regime")
    training_curve()
