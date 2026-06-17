#!/usr/bin/env python3
"""Pipeline diagram: how a generated traffic matrix becomes measured packet loss
and delay. Shows the train-fast / judge-in-ns-3 separation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis("off")

BLUE, GREEN, RED, GREY = "#4c72b0", "#55a868", "#c44e52", "#7f7f7f"

def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                 fc="white", ec=color, lw=2.2))
    ax.text(x + w/2, y + h - 0.32, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=color)
    ax.text(x + w/2, y + h - 0.78, sub, ha="center", va="top", fontsize=8.2, color="#333")

def arrow(x1, y1, x2, y2, label="", color="#444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=20, lw=2, color=color))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.22, label, ha="center", fontsize=8,
                style="italic", color=color)

# --- top row: the live pipeline ---
box(0.3, 6.2, 2.6, 2.2, "1.  traffic.py",
    "gravity model →\nN×N demand matrix\n(Mbps per src→dst).\nDEMAND ONLY —\nno loss/delay yet.", BLUE)
box(3.5, 6.2, 2.7, 2.2, "2.  evaluate_ns3.py",
    "GNN picks a PATH per\nflow; also compute\nOSPF shortest path.\n→ routing JSON\n{src,dst,rate,paths}", BLUE)
box(6.8, 6.2, 2.5, 2.2, "3.  run_ns3_phase2.py",
    "hand routing JSON\nto ns-3 — once for\nOSPF, once for GNN\n(torch-free).", BLUE)
box(9.9, 6.2, 2.8, 2.2, "4.  abilene-validate.cc\n(ns-3 simulator)",
    "builds real network;\nsends real UDP packets\nat 'rate' along the\nexact path; queues drop\noverflow.", GREEN)

arrow(2.9, 7.3, 3.5, 7.3)
arrow(6.2, 7.3, 6.8, 7.3)
arrow(9.3, 7.3, 9.9, 7.3)

# --- ns-3 down to measurement ---
arrow(11.3, 6.2, 11.3, 5.0, color=GREEN)
box(9.4, 3.0, 3.3, 2.0, "5.  FlowMonitor measures",
    "loss% = (tx−rx)/tx\ndelay = Σ(arr−send)/rx\nthroughput = rx·8/t\nPER-PACKET, not a formula.", GREEN)

box(9.4, 0.5, 3.3, 1.9, "RESULT (state JSON)",
    "matrix 1005:\nOSPF 10.49% loss, 140ms\nGNN  9.44% loss, 124ms\n= packets actually dropped", RED)
arrow(11.05, 3.0, 11.05, 2.4, color=RED)

# --- training branch (left, separate) ---
box(0.3, 3.0, 4.2, 2.2, "TRAINING (separate, fast)",
    "train_gnn_qos.py + analytical model:\nmax link util = traffic / capacity.\nFast estimate of congestion — used to\nLEARN the policy. Does NOT simulate\nqueues, so it CANNOT give real loss/delay.", GREY)
arrow(2.4, 6.2, 2.4, 5.2, "policy learned here", GREY)
arrow(4.5, 4.1, 9.4, 4.1, "trained GNN judged in ns-3 →", BLUE)

ax.text(6.5, 8.75, "From generated traffic to measured packet loss & delay",
        ha="center", fontsize=14, fontweight="bold")
ax.text(6.5, 0.05,
        "Train on the fast analytical estimate; REPORT only what ns-3 measures packet-by-packet. "
        "The analytical model proposes, ns-3 disposes.",
        ha="center", fontsize=9, style="italic", color="#333")

fig.tight_layout()
fig.savefig("results/fig_pipeline.png", dpi=150)
print("[saved] results/fig_pipeline.png")
