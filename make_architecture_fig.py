#!/usr/bin/env python3
"""System architecture figure for the Methods chapter.
Two phases: (A) fast training on an analytical surrogate — the RL loop with the GNN
backbone, actor-critic (PPO single-agent / MAPPO CTDE multi-agent), and the
congestion+delay reward; (B) high-fidelity packet-level evaluation in ns-3 against
OSPF. Data sources (SNDlib topologies + real traffic, temporal split) feed both.
Rendered as a labelled block diagram, colours consistent with the thesis figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, GREEN, RED, GREY = "#4c72b0", "#55a868", "#c44e52", "#555555"
INK = "#222222"


def box(ax, x, y, w, h, title, body, ec, fc="white", tfs=10.5, bfs=8.3):
    """Box with title near the top and body just beneath it (both inside the box)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 0.032, title, ha="center", va="top",
            fontsize=tfs, fontweight="bold", color=ec, zorder=3)
    if body:
        ax.text(x + w / 2, y + h - 0.072, body, ha="center", va="top",
                fontsize=bfs, color=INK, zorder=3, linespacing=1.35)


def arrow(ax, xy1, xy2, color=INK, style="-|>", lw=1.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=15,
                                 lw=lw, color=color, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(13.5, 9.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- Data sources (top band) ----
    box(ax, 0.28, 0.885, 0.44, 0.10, "Data — SNDlib real backbone networks",
        "Abilene (12n) · GÉANT (22n) · Germany50 (50n): topologies + capacities\n"
        "+ real measured traffic matrices, temporal split (train 70% / test 30%)",
        GREY, "#f2f2f2", tfs=10.5, bfs=8.2)

    # container geometry
    LX, RX, W = 0.06, 0.57, 0.39
    CY0, CH = 0.075, 0.775              # container bottom, height
    # four rows inside each column
    BH, GAP = 0.155, 0.022
    top = 0.645                         # box0 bottom (top row) — leaves room for header
    ys = [top - i * (BH + GAP) for i in range(4)]   # box bottom y's

    # ============ PHASE A: TRAINING (left) ============
    ax.add_patch(FancyBboxPatch((0.02, CY0), 0.45, CH,
                 boxstyle="round,pad=0.006,rounding_size=0.015",
                 linewidth=1.4, edgecolor=BLUE, facecolor="#f5f8fc", zorder=0))
    ax.text(LX + W / 2, CY0 + CH - 0.022, "A.  TRAINING — fast analytical surrogate",
            ha="center", va="top", fontsize=12, fontweight="bold", color=BLUE)

    box(ax, LX, ys[0], W, BH, "Environment  (analytical link-util MDP)",
        "state $s_t$: link utilisations $\\{u_e\\}$, adjacency $A$,\n"
        "current flow $(s_f,d_f,r_f)$ and its $k$-shortest paths $\\mathcal{P}_f$\n"
        "fast congestion estimate — no packet simulation", BLUE)
    box(ax, LX, ys[1], W, BH, "GNN backbone  (SeqGNN extractor)",
        "message passing over the topology graph →\n"
        "node / arc embeddings capturing local +\n"
        "network-wide state — the novel element", BLUE)
    box(ax, LX, ys[2], W, BH, "Policy  (Actor–Critic)",
        "single-agent PPO (centralized)  |  MAPPO (CTDE)\n"
        "action $a_t$: select path from $\\mathcal{P}_f$ / next hop\n"
        "centralized critic at train · local obs at execution", BLUE)
    box(ax, LX, ys[3], W, BH, "Reward  (congestion + delay)",
        "$r_t = -(U_t - U_{t-1}) - \\lambda(|p_t| - h^{\\star}_t)$\n"
        "minimise bottleneck utilisation;\n"
        "keep paths short when uncongested", BLUE)

    for i in range(3):
        arrow(ax, (LX + W / 2, ys[i]), (LX + W / 2, ys[i + 1] + BH), BLUE)
    # feedback loop reward -> env (in the left margin, inside the container)
    arrow(ax, (0.045, ys[3] + 0.03), (0.045, ys[0] + BH - 0.03),
          BLUE, ls="--", rad=-0.22)
    ax.text(0.033, (ys[0] + ys[3]) / 2 + BH / 2, "policy update", rotation=90,
            ha="center", va="center", fontsize=7.5, color=BLUE, style="italic")

    # ============ PHASE B: EVALUATION (right) ============
    ax.add_patch(FancyBboxPatch((0.55, CY0), 0.44, CH,
                 boxstyle="round,pad=0.006,rounding_size=0.015",
                 linewidth=1.4, edgecolor=GREEN, facecolor="#f4faf6", zorder=0))
    ax.text(RX + W / 2, CY0 + CH - 0.022, "B.  EVALUATION — packet-level ns-3",
            ha="center", va="top", fontsize=12, fontweight="bold", color=GREEN)

    box(ax, RX, ys[0], W, BH, "Extract routing  (trained policy)",
        "run policy deterministically on held-out matrices\n"
        "→ exact per-flow paths  $\\{s_f, d_f, r_f, \\mathrm{path}\\}$\n"
        "reproducible — no exploration noise", GREEN)
    box(ax, RX, ys[1], W, BH, "ns-3 simulation  (static host-routes)",
        "install the exact paths; send real UDP traffic\n"
        "at rate $r_f$; queues drop overflow —\n"
        "real packet-level dynamics", GREEN)
    box(ax, RX, ys[2], W, BH, "FlowMonitor  (measured QoS)",
        "max link utilisation · packet loss · delay ·\n"
        "throughput, measured per packet;\n"
        "mean ± std over model seeds 0/1/2", GREEN)
    box(ax, RX, ys[3], W, BH, "Baseline & comparison",
        "OSPF shortest-path (Dijkstra) routed identically\n"
        "on the same matrices; results stratified by\n"
        "regime: overload vs feasible", RED)

    for i in range(3):
        arrow(ax, (RX + W / 2, ys[i]), (RX + W / 2, ys[i + 1] + BH), GREEN)

    # ---- cross-phase + data feed arrows ----
    arrow(ax, (0.40, 0.885), (LX + W / 2, ys[0] + BH), GREY)     # data -> training
    arrow(ax, (0.60, 0.885), (RX + W / 2, ys[0] + BH), GREY)     # data -> eval
    # trained policy: bottom of training column -> top eval box
    arrow(ax, (LX + W, ys[2] + BH / 2), (RX, ys[0] + BH / 2), BLUE, lw=2.4)
    ax.text(0.50, (ys[0] + ys[2]) / 2 + BH / 2 + 0.01, "trained\npolicy",
            ha="center", va="center", fontsize=8.5, color=BLUE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=BLUE, lw=1.2))

    fig.suptitle("System Architecture — GNN-based (multi-agent) RL routing:\n"
                 "fast surrogate training, high-fidelity ns-3 evaluation against OSPF",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)
    out = "results/fig_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
