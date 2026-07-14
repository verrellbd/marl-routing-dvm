#!/usr/bin/env python3
"""Fresh 3-way ns-3 QoS figures (OSPF vs single-agent GNN vs MARL), per network,
multi-seed (0/1/2) mean ± std, matching RESULTS_SUMMARY.md §2. Loss + delay panels,
overload vs feasible regime. Reads the per-seed summary.json files directly (no re-sim):
  results/ns3_eval_realsa_fresh_<net>_s<seed>/summary.json   (SA-GNN routing + OSPF)
  results/ns3_eval_realmarl_fresh_<net>_s<seed>/summary.json (MARL routing)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
SEEDS = [0, 1, 2]
NETS = [("abilene_sndlib", "Abilene (12n) — real Zhang TM"),
        ("geant_sndlib", "GÉANT (22n) — real Uhlig/TOTEM TM"),
        ("germany50_sndlib", "Germany50 (50n) — real DFN TM")]
OSPF_C, SA_C, MA_C = "#c44e52", "#55a868", "#4c72b0"
REGIMES = ["overload", "feasible"]
REG_LAB = ["Overload\n(offered > capacity)", "Feasible\n(offered < capacity)"]


def gather(meth, net, regime, key):
    vals = []
    for s in SEEDS:
        fp = f"{R}/ns3_eval_{meth}_fresh_{net}_s{s}/summary.json"
        if not os.path.exists(fp):
            continue
        b = json.load(open(fp))["by_regime"][regime]
        if b.get(key) is not None:
            vals.append(b[key])
    return np.array(vals)


def mean_std(a):
    return (float(a.mean()), float(a.std())) if len(a) else (0.0, 0.0)


def make(net, title):
    x = np.arange(len(REGIMES)); w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, ok, gk, ylabel in [
            (axes[0], "ospf_loss", "gnn_loss", "Packet loss (%)"),
            (axes[1], "ospf_delay_ms", "gnn_delay_ms", "Mean delay (ms)")]:
        ospf = [mean_std(gather("realsa", net, r, ok)) for r in REGIMES]   # OSPF from SA dirs
        sa = [mean_std(gather("realsa", net, r, gk)) for r in REGIMES]
        ma = [mean_std(gather("realmarl", net, r, gk)) for r in REGIMES]
        for off, series, c, lab in [(-w, ospf, OSPF_C, "OSPF"),
                                    (0, sa, SA_C, "single-agent GNN (centralized)"),
                                    (w, ma, MA_C, "MARL / MAPPO (decentralized)")]:
            m = [v[0] for v in series]; e = [v[1] for v in series]
            ax.bar(x + off, m, w, yerr=e, capsize=4, color=c, label=lab)
            for xi, (mv, ev) in zip(x + off, series):
                ax.text(xi, mv + ev, f"{mv:.2f}" if mv < 10 else f"{mv:.1f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(REG_LAB)
        ax.set_ylabel(ylabel); ax.set_title(ylabel.split(" (")[0], fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(f"OSPF vs single-agent GNN vs MARL — {title}\n"
                 "ns-3 packet-level QoS, mean ± std over model seeds 0/1/2",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    short = net.replace("_sndlib", "")
    fig.savefig(f"{R}/fig_real3way_{short}.png", dpi=150)
    print(f"[saved] {R}/fig_real3way_{short}.png")


if __name__ == "__main__":
    for net, title in NETS:
        make(net, title)
