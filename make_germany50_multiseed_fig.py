#!/usr/bin/env python3
"""Germany50 packet-level MULTI-SEED figure (honest): OSPF vs centralized GNN vs
hop-capped MARL, overload regime, loss + delay. MARL shown as mean±std over model
seeds 0/1/2 WITH per-seed points overlaid, so the high loss-variance (seed 2 worse
than OSPF) is visible rather than hidden by the mean."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIRS = {0: "results/ns3_eval_realmarl_opt3b_germany50_sndlib",
        1: "results/ns3_eval_realmarl_opt3b_s1_germany50_sndlib",
        2: "results/ns3_eval_realmarl_opt3b_s2_germany50_sndlib"}
marl_loss, marl_delay = [], []
for s, d in DIRS.items():
    o = json.load(open(f"{d}/summary.json"))["by_regime"]["overload"]
    marl_loss.append(o["gnn_loss"]); marl_delay.append(o["gnn_delay_ms"])
# OSPF / SA-GNN references (seed-0 ns-3 anchor; OSPF identical across seeds)
OSPF = {"loss": 3.16, "delay": 17.4}
SA = {"loss": 0.03, "delay": 1.9}
OSPF_C, SA_C, MA_C = "#c44e52", "#55a868", "#4c72b0"

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, key, vals, ylab, title in [
    (axes[0], "loss", marl_loss, "Packet loss (%)", "Packet loss (overload)"),
    (axes[1], "delay", marl_delay, "Mean delay (ms)", "Mean delay (overload)")]:
    x = [0, 1, 2]
    ax.bar(0, OSPF[key], 0.6, color=OSPF_C, label="OSPF")
    ax.bar(1, SA[key], 0.6, color=SA_C, label="single-agent GNN (centralized, seed 0)")
    ax.bar(2, np.mean(vals), 0.6, yerr=np.std(vals), capsize=6, color=MA_C,
           label="MARL hop-capped (decentralized, mean±std seeds 0/1/2)")
    # per-seed scatter on the MARL bar
    ax.scatter([2] * 3, vals, color="black", zorder=5, s=35)
    for s, v in zip(DIRS, vals):
        ax.annotate(f"s{s}", (2, v), textcoords="offset points", xytext=(8, -2), fontsize=8)
    ax.axhline(OSPF[key], color=OSPF_C, ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(["OSPF", "GNN", "MARL"])
    ax.set_ylabel(ylab); ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Germany50 (real DFN traffic, 50 nodes) — hop-capped MARL, multi-seed\n"
             "delay reliably beats OSPF; loss is high-variance (seed 2 > OSPF) — the "
             "coordination cost at scale shows as VARIANCE", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("results/fig_germany50_multiseed.png", dpi=150)
print("[saved] results/fig_germany50_multiseed.png")
print(f"  loss {np.mean(marl_loss):.2f}±{np.std(marl_loss):.2f}%  "
      f"delay {np.mean(marl_delay):.1f}±{np.std(marl_delay):.1f}ms")
