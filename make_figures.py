#!/usr/bin/env python3
"""Generate the OSPF-vs-GNN comparison figures for the thesis.
Reads the ns-3 evaluation summaries and produces:
  results/fig_ospf_vs_gnn.png   - main result: loss+delay by regime, both topologies
  results/fig_geant_headroom.png - GEANT capacity-limited proof (OSPF vs optimal greedy)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OSPF_C, GNN_C = "#c44e52", "#4c72b0"   # red = OSPF baseline, blue = GNN

# ---------- load Abilene (robust multi-seed, already aggregated) ----------
ab = json.loads((R / "ns3_eval_multiseed_robust.json").read_text())["by_regime"]

def ab_vals(regime, key):
    d = ab[regime][key]
    return d["mean"], d["std"]

# ---------- load GEANT (aggregate the 3 model seeds here) ----------
geant_seeds = [json.loads((R / f"ns3_eval_geantStress_seed{s}" / "summary.json").read_text())
               for s in (0, 1, 2)]

def ge_vals(regime, key):
    v = [g["by_regime"][regime][key] for g in geant_seeds]
    return float(np.mean(v)), float(np.std(v))

# ============================================================
# FIGURE 1 — main OSPF vs GNN comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
regimes = ["overload", "feasible"]
reg_labels = ["Overload\n(>100% offered)", "Feasible\n(<100% offered)"]
x = np.arange(len(regimes)); w = 0.36

panels = [
    (axes[0, 0], "Abilene", "loss",  ab_vals, "Packet loss (%)",  "ospf_loss",  "gnn_loss"),
    (axes[0, 1], "Abilene", "delay", ab_vals, "Mean delay (ms)",  "ospf_delay_ms", "gnn_delay_ms"),
    (axes[1, 0], "GÉANT",   "loss",  ge_vals, "Packet loss (%)",  "ospf_loss",  "gnn_loss"),
    (axes[1, 1], "GÉANT",   "delay", ge_vals, "Mean delay (ms)",  "ospf_delay_ms", "gnn_delay_ms"),
]

for ax, topo, _kind, getter, ylabel, ok, gk in panels:
    om = [getter(r, ok)[0] for r in regimes]; oe = [getter(r, ok)[1] for r in regimes]
    gm = [getter(r, gk)[0] for r in regimes]; ge = [getter(r, gk)[1] for r in regimes]
    b1 = ax.bar(x - w/2, om, w, yerr=oe, capsize=4, color=OSPF_C, label="OSPF")
    b2 = ax.bar(x + w/2, gm, w, yerr=ge, capsize=4, color=GNN_C, label="GNN")
    ax.set_xticks(x); ax.set_xticklabels(reg_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{topo} — {ylabel.split(' (')[0]}", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for b, m in list(zip(b1, om)) + list(zip(b2, gm)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{m:.2f}" if m < 10 else f"{m:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.legend(loc="upper right", fontsize=9)

axes[0, 0].annotate("GNN cuts loss ~10x\nunder congestion",
                    xy=(0 + w/2, 0.74), xytext=(0.15, 4.5), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="gray"))
axes[1, 0].annotate("GNN ≈ OSPF\n(capacity-limited)",
                    xy=(0 + w/2, 5.5), xytext=(0.3, 4.0), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="gray"))

fig.suptitle("Learned routing (GNN) vs OSPF — ns-3 packet-level QoS, held-out traffic",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.005,
         "Abilene (uniform 10G, well-connected): GNN strictly dominates under congestion.   "
         "GÉANT (mixed 2.5G/10G, bottleneck cuts): GNN safely matches OSPF.",
         ha="center", fontsize=9, style="italic")
fig.tight_layout(rect=[0, 0.025, 1, 0.96])
fig.savefig(R / "fig_ospf_vs_gnn.png", dpi=150)
print(f"[saved] {R/'fig_ospf_vs_gnn.png'}")

# ============================================================
# FIGURE 2 — GEANT headroom (why GNN can't beat OSPF there)
# ============================================================
# analytical max link-util (%) at load 1.5, from geant_headroom.py
headroom = {
    "seed 1005": {"OSPF": 156, "greedy k=3": 156, "greedy k=5": 156, "greedy k=8": 156},
    "seed 1008": {"OSPF": 112, "greedy k=3": 107, "greedy k=5": 107, "greedy k=8": 107},
    "seed 1013": {"OSPF": 102, "greedy k=3": 102, "greedy k=5": 102, "greedy k=8": 102},
}
methods = ["OSPF", "greedy k=3", "greedy k=5", "greedy k=8"]
colors = [OSPF_C, "#8c8c8c", "#55a868", "#4c72b0"]
seeds = list(headroom.keys())
xs = np.arange(len(seeds)); bw = 0.2

fig2, ax = plt.subplots(figsize=(9, 5.5))
for i, m in enumerate(methods):
    vals = [headroom[s][m] for s in seeds]
    bars = ax.bar(xs + (i - 1.5) * bw, vals, bw, color=colors[i], label=m)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v}",
                ha="center", va="bottom", fontsize=8)
ax.axhline(100, color="black", ls="--", lw=1, alpha=0.7)
ax.text(-0.45, 103, "100% = link saturated", fontsize=8, ha="left", color="black")
ax.set_xticks(xs); ax.set_xticklabels(seeds)
ax.set_ylabel("Max link utilisation (%)")
ax.set_title("GÉANT is capacity-limited: even optimal rerouting (greedy, up to k=8)\n"
             "cannot relieve the overloaded cut — so the GNN cannot beat OSPF",
             fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig2.tight_layout()
fig2.savefig(R / "fig_geant_headroom.png", dpi=150)
print(f"[saved] {R/'fig_geant_headroom.png'}")

# ---------- print the aggregated numbers used ----------
print("\n=== numbers plotted ===")
for r in regimes:
    print(f"ABILENE {r:8}: loss OSPF {ab_vals(r,'ospf_loss')[0]:.2f} -> GNN {ab_vals(r,'gnn_loss')[0]:.2f}"
          f" | delay {ab_vals(r,'ospf_delay_ms')[0]:.1f} -> {ab_vals(r,'gnn_delay_ms')[0]:.1f}")
for r in regimes:
    print(f"GEANT   {r:8}: loss OSPF {ge_vals(r,'ospf_loss')[0]:.2f}±{ge_vals(r,'ospf_loss')[1]:.2f}"
          f" -> GNN {ge_vals(r,'gnn_loss')[0]:.2f}±{ge_vals(r,'gnn_loss')[1]:.2f}"
          f" | delay {ge_vals(r,'ospf_delay_ms')[0]:.1f} -> {ge_vals(r,'gnn_delay_ms')[0]:.1f}±{ge_vals(r,'gnn_delay_ms')[1]:.1f}")
