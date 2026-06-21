#!/usr/bin/env python3
"""3-way comparison figure: OSPF vs single-agent GNN (centralized) vs MARL
(decentralized MAPPO), ns-3 packet-level QoS by congestion regime.
Reads two ns-3 summaries evaluated on IDENTICAL held-out seeds."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
sa_dir = sys.argv[1] if len(sys.argv) > 1 else "ns3_eval_sa3way"
ma_dir = sys.argv[2] if len(sys.argv) > 2 else "ns3_eval_marlv2_seed0"
title = sys.argv[3] if len(sys.argv) > 3 else "Abilene (load 4.0)"
out = sys.argv[4] if len(sys.argv) > 4 else "fig_3way_abilene.png"
caption = sys.argv[5] if len(sys.argv) > 5 else (
    "Both learned methods beat OSPF under congestion. MARL (local-only, no central "
    "controller) lands within ~1pt of the centralized GNN — a small coordination cost.")

sa = json.loads((R / sa_dir / "summary.json").read_text())["by_regime"]
ma = json.loads((R / ma_dir / "summary.json").read_text())["by_regime"]

OSPF_C, SA_C, MA_C = "#c44e52", "#55a868", "#4c72b0"
regimes = ["overload", "feasible"]
reg_lab = ["Overload\n(>100% offered)", "Feasible\n(<100% offered)"]
x = np.arange(2); w = 0.26

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, key_o, key_g, ylabel in [
        (axes[0], "ospf_loss", "gnn_loss", "Packet loss (%)"),
        (axes[1], "ospf_delay_ms", "gnn_delay_ms", "Mean delay (ms)")]:
    ospf = [sa[r][key_o] for r in regimes]
    saw  = [sa[r][key_g] for r in regimes]
    marl = [ma[r][key_g] for r in regimes]
    b0 = ax.bar(x - w, ospf, w, color=OSPF_C, label="OSPF")
    b1 = ax.bar(x,     saw,  w, color=SA_C, label="single-agent GNN (centralized)")
    b2 = ax.bar(x + w, marl, w, color=MA_C, label="MARL / MAPPO (decentralized)")
    for bars, vals in [(b0, ospf), (b1, saw), (b2, marl)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(reg_lab)
    ax.set_ylabel(ylabel); ax.set_title(ylabel.split(" (")[0], fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(loc="upper right", fontsize=8.5)
fig.suptitle(f"OSPF vs single-agent GNN vs MARL — {title}, ns-3 packet-level QoS",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005, caption, ha="center", fontsize=9, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(R / out, dpi=150)
print(f"[saved] {R/out}")
