#!/usr/bin/env python3
"""Deployment / runtime view: how a live flow is routed from source to destination
under the trained decentralized policy. Shows a small backbone topology, the OSPF
shortest path (through a congested link) vs. the learned path (detour that avoids it),
and a callout of the per-router decision pipeline (decentralized execution). Colours
consistent with the thesis figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BLUE, GREEN, RED, GREY, INK = "#4c72b0", "#55a868", "#c44e52", "#666666", "#222222"

# ---- topology (node -> position) ----
POS = {
    "S": (0.05, 0.42), "A": (0.17, 0.66), "B": (0.17, 0.18),
    "C": (0.34, 0.66), "E": (0.34, 0.18), "F": (0.50, 0.42), "D": (0.61, 0.42),
}
EDGES = [("S", "A"), ("S", "B"), ("A", "B"), ("A", "C"), ("B", "E"),
         ("C", "E"), ("C", "F"), ("E", "F"), ("F", "D"), ("C", "D")]
OSPF_PATH = ["S", "A", "C", "D"]          # shortest — crosses congested C-D
LEARN_PATH = ["S", "B", "E", "F", "D"]    # detour — avoids C-D
CONGESTED = ("C", "D")
LABELS = {"S": "S\n(source)", "D": "D\n(dest)"}


def edge(ax, u, v, color=GREY, lw=1.6, ls="-", z=1):
    (x1, y1), (x2, y2) = POS[u], POS[v]
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="round")


def path_overlay(ax, path, color, lw, ls="-", z=3):
    for u, v in zip(path, path[1:]):
        edge(ax, u, v, color=color, lw=lw, ls=ls, z=z)


def main():
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # base topology
    for u, v in EDGES:
        edge(ax, u, v, GREY, 1.6, z=1)
    # congested link highlighted
    edge(ax, *CONGESTED, RED, 6.5, z=2)
    cx = (POS[CONGESTED[0]][0] + POS[CONGESTED[1]][0]) / 2
    cy = (POS[CONGESTED[0]][1] + POS[CONGESTED[1]][1]) / 2
    ax.text(cx, cy + 0.05, "congested link\n(128% offered)", ha="center", va="bottom",
            fontsize=8, color=RED, fontweight="bold")

    # the two routings
    path_overlay(ax, OSPF_PATH, RED, 3.0, ls=(0, (4, 3)), z=3)
    path_overlay(ax, LEARN_PATH, GREEN, 4.0, z=4)

    # nodes
    for n, (x, y) in POS.items():
        fc = "#ffe9b0" if n in ("S", "D") else "white"
        ax.add_patch(Circle((x, y), 0.028, facecolor=fc, edgecolor=INK,
                            lw=1.6, zorder=6))
        ax.text(x, y, n, ha="center", va="center", fontsize=10.5,
                fontweight="bold", zorder=7)
    for n, lab in LABELS.items():
        x, y = POS[n]
        ax.text(x, y - 0.055, lab, ha="center", va="top", fontsize=8.5,
                color=INK, fontweight="bold")

    # flow direction + packet dots on the learned path
    for u, v in zip(LEARN_PATH, LEARN_PATH[1:]):
        (x1, y1), (x2, y2) = POS[u], POS[v]
        for t in (0.4, 0.6):
            ax.plot(x1 + t * (x2 - x1), y1 + t * (y2 - y1), "o",
                    color=GREEN, ms=4.5, zorder=5)

    # legend
    ax.plot([], [], color=GREEN, lw=4, label="learned path (avoids congestion)")
    ax.plot([], [], color=RED, lw=3, ls=(0, (4, 3)), label="OSPF shortest path")
    ax.plot([], [], color=RED, lw=6, label="saturated link")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=9, framealpha=0.95)

    # ---- per-router decision callout (right) ----
    bx, bw = 0.66, 0.32
    ax.add_patch(FancyBboxPatch((bx, 0.10), bw, 0.80,
                 boxstyle="round,pad=0.008,rounding_size=0.02",
                 linewidth=1.6, edgecolor=BLUE, facecolor="#f5f8fc", zorder=8))
    ax.text(bx + bw / 2, 0.865, "Inside each router\n(decentralized execution)",
            ha="center", va="top", fontsize=11.5, fontweight="bold", color=BLUE, zorder=9)

    steps = [
        ("1  Packet / flow arrives", "carrying its destination $d$"),
        ("2  Observe local state", "utilisation of incident links $\\{u_e\\}$"),
        ("3  GNN policy (forward pass)", "embed local + neighbourhood state"),
        ("4  Select next hop", "the link that best relieves congestion"),
        ("5  Forward packet", "toward $d$; repeat at the next router"),
    ]
    y = 0.78
    for i, (t, sub) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((bx + 0.03, y - 0.085), bw - 0.06, 0.09,
                     boxstyle="round,pad=0.006,rounding_size=0.015",
                     linewidth=1.3, edgecolor=BLUE, facecolor="white", zorder=9))
        ax.text(bx + bw / 2, y - 0.018, t, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=INK, zorder=10)
        ax.text(bx + bw / 2, y - 0.048, sub, ha="center", va="top",
                fontsize=8, color=INK, zorder=10)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((bx + bw / 2, y - 0.086),
                         (bx + bw / 2, y - 0.10), arrowstyle="-|>",
                         mutation_scale=12, lw=1.4, color=BLUE, zorder=9))
        y -= 0.128

    # zoom connector from a router to the callout
    ax.add_patch(FancyArrowPatch((POS["E"][0] + 0.02, POS["E"][1]),
                 (bx, 0.30), arrowstyle="-", lw=1.2, color=BLUE,
                 linestyle=(0, (3, 3)), zorder=2,
                 connectionstyle="arc3,rad=-0.15"))

    fig.suptitle("Deployment view — routing a live flow hop-by-hop under the learned policy",
                 fontsize=13.5, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
    out = "results/fig_deployment.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
