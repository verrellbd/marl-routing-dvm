#!/usr/bin/env python3
"""Multi-seed error-bar figure: max link-util (overload regime) across the three real
topologies, OSPF vs SA-GNN vs MARL, mean +/- std over model seeds 0/1/2."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/multiseed_analytical.json"))
labels = {"abilene_sndlib": "Abilene", "geant_sndlib": "GÉANT", "germany50_sndlib": "Germany50"}
topos = [t for t in labels if t in d and "overload" in d[t]]
x = np.arange(len(topos)); w = 0.26
OSPF_C, SA_C, MA_C = "#c44e52", "#55a868", "#4c72b0"

fig, ax = plt.subplots(figsize=(9, 5.5))
ospf = [d[t]["overload"]["ospf"] for t in topos]
sa = [d[t]["overload"]["sa_gnn"][0] for t in topos]; sa_e = [d[t]["overload"]["sa_gnn"][1] for t in topos]
ma = [d[t]["overload"]["marl"][0] for t in topos]; ma_e = [d[t]["overload"]["marl"][1] for t in topos]

ax.bar(x - w, ospf, w, color=OSPF_C, label="OSPF")
ax.bar(x, sa, w, yerr=sa_e, capsize=5, color=SA_C, label="single-agent GNN (centralized)")
ax.bar(x + w, ma, w, yerr=ma_e, capsize=5, color=MA_C, label="MARL / MAPPO (decentralized)")
ax.axhline(100, color="black", ls="--", lw=1, alpha=0.6)
ax.text(len(topos) - 0.5, 102, "100% = link saturated", fontsize=8, ha="right")
for i, t in enumerate(topos):
    ax.text(i - w, ospf[i] + 1, f"{ospf[i]:.0f}", ha="center", fontsize=8)
    ax.text(i, sa[i] + sa_e[i] + 1, f"{sa[i]:.0f}±{sa_e[i]:.0f}", ha="center", fontsize=8)
    ax.text(i + w, ma[i] + ma_e[i] + 1, f"{ma[i]:.0f}±{ma_e[i]:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([labels[t] for t in topos])
ax.set_ylabel("Max link utilisation (%)")
ax.set_title("Multi-seed robustness (overload regime, real traffic)\n"
             "max link-util, mean ± std over model seeds 0/1/2", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results/fig_multiseed_overload.png", dpi=150)
print("[saved] results/fig_multiseed_overload.png")
