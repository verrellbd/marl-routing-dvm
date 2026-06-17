#!/usr/bin/env python3
"""Heatmaps of one Abilene and one GÉANT gravity-model traffic matrix, showing
the realistic hot-spot structure (a few node pairs carry most of the demand).
GÉANT panel marks the column for node 22 (RU) — the capacity-limited bottleneck."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from marl_routing.traffic import generate_matrix

specs = [("abilene", 1.0, 1003, "Abilene  (12 nodes, all 10G)"),
         ("geant",   1.5, 1005, "GÉANT  (23 nodes, mixed 2.5G/10G)")]

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
for ax, (topo, load, seed, title) in zip(axes, specs):
    T = generate_matrix(topo, load, seed=seed).copy()
    np.fill_diagonal(T, np.nan)  # self-pairs are 0/undefined
    im = ax.imshow(T, cmap="inferno", aspect="auto")
    ax.set_title(f"{title}\nseed {seed}, load {load} — total {np.nansum(T):.0f} Mbps",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("destination node"); ax.set_ylabel("source node")
    ax.set_xticks(range(T.shape[0])); ax.set_yticks(range(T.shape[0]))
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("demand (Mbps)")
    if topo == "geant":
        ax.axvline(22, color="#39e0ff", lw=1.6, alpha=0.9)
        ax.annotate("col 22 = RU\n(2.5G bottleneck:\n3,893 Mbps → 156%)",
                    xy=(22, 2), xytext=(13.5, -3.2), color="#0a6a85", fontsize=8,
                    fontweight="bold", ha="left",
                    arrowprops=dict(arrowstyle="->", color="#39a0c5"))

fig.suptitle("Gravity-model traffic: every node pair carries demand, but a few "
             "hot-spot pairs dominate (bright cells)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/fig_traffic_heatmap.png", dpi=150)
print("[saved] results/fig_traffic_heatmap.png")
