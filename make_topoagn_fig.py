#!/usr/bin/env python3
"""Zero-shot topology transfer figure. One shared-weight GNN policy trained ONLY on
Abilene + GEANT, then evaluated on all three networks -- including Germany50, which it
never saw. Bars = max link utilisation (analytical), OSPF vs the single policy,
mean +/- std over seeds 0/1/2. The held-out network is visually set apart.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
OSPF_C, GNN_C, HOLD_C = "#c44e52", "#4c72b0", "#dd8452"
NAMES = {"abilene_sndlib": "Abilene\n(12n) — seen",
         "geant_sndlib": "GÉANT\n(22n) — seen",
         "germany50_sndlib": "Germany50\n(50n) — ZERO-SHOT"}
ORDER = ["abilene_sndlib", "geant_sndlib", "germany50_sndlib"]

agg = json.load(open(f"{R}/topoagn_3seed_summary.json"))["by_topology"]
x = np.arange(len(ORDER)); w = 0.36
fig, ax = plt.subplots(figsize=(9.5, 5.6))

ospf = [agg[t]["ospf"] for t in ORDER]
gnn = [agg[t]["learned_mean"] for t in ORDER]
gstd = [agg[t]["learned_std"] for t in ORDER]

ax.bar(x - w / 2, ospf, w, color=OSPF_C, label="OSPF (per-network baseline)")
bar = ax.bar(x + w / 2, gnn, w, yerr=gstd, capsize=5,
             color=[HOLD_C if t == "germany50_sndlib" else GNN_C for t in ORDER],
             label="single topology-agnostic GNN")
for xi, v in zip(x - w / 2, ospf):
    ax.text(xi, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
for xi, v, e in zip(x + w / 2, gnn, gstd):
    ax.text(xi, v + e + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

ax.axhline(100, color="#333", lw=1.1, ls="--", zorder=0)
ax.text(len(ORDER) - 0.5, 100, " capacity", color="#333", fontsize=8, va="bottom", ha="right")
ax.axvline(1.5, color="#999", lw=1, ls=":")
ax.text(2.0, ax.get_ylim()[1] * 0.96 if False else 168, "trained on ⟵    ⟶ never trained on",
        ha="center", va="top", fontsize=8.5, color="#555", style="italic")

ax.set_xticks(x); ax.set_xticklabels([NAMES[t] for t in ORDER])
ax.set_ylabel("Max link utilisation (%)", fontweight="bold")
ax.set_ylim(0, 175)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_title("Zero-shot topology transfer: one GNN policy, trained only on Abilene+GÉANT\n"
             "beats OSPF on an unseen 50-node network (mean ± std, seeds 0/1/2)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{R}/fig_topoagn_zeroshot.png", dpi=150)
print(f"[saved] {R}/fig_topoagn_zeroshot.png")
