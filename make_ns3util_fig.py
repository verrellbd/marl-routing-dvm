#!/usr/bin/env python3
"""§4.1 utilisation figure — now ns-3 PACKET-LEVEL (max link-util measured from bytes
carried per device), NOT analytical. Two panels: overload | feasible regime, three real
topologies, OSPF vs SA-GNN vs MARL, mean ± std over model seeds 0/1/2 × test matrices.

Reads results/ns3_util_summary_{overload,feasible}.json (from aggregate_ns3_util.py).
Utilisation is true (active-window corrected). ns-3 caps at 100% (link cannot carry more
than capacity) — so OSPF pins at 100% under overload; the excess offered load appears as
packet loss (see the QoS / loss figures), not as >100% utilisation.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
NETS = [("abilene_sndlib", "Abilene\n(12n)"),
        ("geant_sndlib", "GÉANT\n(22n)"),
        ("germany50_sndlib", "Germany50\n(50n)")]
OSPF_C, SA_C, MA_C = "#c44e52", "#55a868", "#4c72b0"

data = {reg: json.load(open(f"{R}/ns3_util_summary_{reg}.json"))
        for reg in ("overload", "feasible")}
x = np.arange(len(NETS)); w = 0.26

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
for ax, reg, sub in [(axes[0], "overload", "Overload regime (offered load > capacity)"),
                     (axes[1], "feasible", "Feasible regime (offered load < capacity)")]:
    d = data[reg]
    def col(k):
        return ([d[n][k]["mean"] for n, _ in NETS],
                [d[n][k]["std"] for n, _ in NETS])
    ospf, ospf_e = col("ospf"); sa, sa_e = col("sa"); ma, ma_e = col("marl")
    ax.bar(x - w, ospf, w, yerr=ospf_e, capsize=4, color=OSPF_C, label="OSPF")
    ax.bar(x, sa, w, yerr=sa_e, capsize=4, color=SA_C, label="single-agent GNN (centralized)")
    ax.bar(x + w, ma, w, yerr=ma_e, capsize=4, color=MA_C, label="MARL / MAPPO (decentralized)")
    ax.axhline(100, color="black", ls="--", lw=1, alpha=0.6)
    ax.text(-0.42, 108, "100% = link saturated", fontsize=8, ha="left")
    for i in range(len(NETS)):
        for off, v, e in [(-w, ospf[i], ospf_e[i]), (0, sa[i], sa_e[i]), (w, ma[i], ma_e[i])]:
            ax.text(i + off, v + e + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in NETS])
    ax.set_title(sub, fontweight="bold", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Max link utilisation (%) — ns-3 packet-level")
axes[0].legend(loc="lower left", fontsize=8.5)
axes[0].set_ylim(0, 118)
fig.suptitle("ns-3 max link-utilisation: OSPF vs single-agent GNN vs MARL\n"
             "mean ± std over model seeds 0/1/2 × held-out test matrices",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{R}/fig_ns3util_multiseed.png", dpi=150)
print(f"[saved] {R}/fig_ns3util_multiseed.png")
