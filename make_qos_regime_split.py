#!/usr/bin/env python3
"""QoS figures reorganised: all three topologies on the SAME graph, split by regime.
Two SEPARATE figures — one for packet loss, one for mean delay. Each is a 1x2 grid
(columns = overload | feasible). Within each panel: x-axis = topology, grouped bars =
OSPF / single-agent GNN / MARL, mean +/- std over model seeds 0/1/2. Reads the per-seed
summary.json files (no re-sim). Colours match fig_real3way_* for thesis consistency.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
SEEDS = [0, 1, 2]
NETS = [("abilene_sndlib", "Abilene\n(12n)"),
        ("geant_sndlib", "GÉANT\n(22n)"),
        ("germany50_sndlib", "Germany50\n(50n)")]
OSPF_C, SA_C, MA_C = "#c44e52", "#55a868", "#4c72b0"
REGIMES = ["overload", "feasible"]
REG_TITLE = ["Overload  (OSPF drives a link ≥100%)",
             "Feasible  (traffic fits under OSPF)"]


def gather(meth, net, regime, key):
    vals = []
    for s in SEEDS:
        fp = f"{R}/ns3_eval_{meth}_fresh_{net}_s{s}/summary.json"
        if os.path.exists(fp):
            b = json.load(open(fp))["by_regime"][regime]
            if b.get(key) is not None:
                vals.append(b[key])
    a = np.array(vals)
    return (float(a.mean()), float(a.std())) if len(a) else (0.0, 0.0)


def make(metric, ok, gk, ylab, out):
    x = np.arange(len(NETS))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ci, regime in enumerate(REGIMES):
        ax = axes[ci]
        ospf = [gather("realsa", n, regime, ok) for n, _ in NETS]
        sa = [gather("realsa", n, regime, gk) for n, _ in NETS]
        ma = [gather("realmarl", n, regime, gk) for n, _ in NETS]
        for off, series, c, lab in [(-w, ospf, OSPF_C, "OSPF"),
                                    (0, sa, SA_C, "single-agent GNN"),
                                    (w, ma, MA_C, "MARL")]:
            m = [v[0] for v in series]
            e = [v[1] for v in series]
            ax.bar(x + off, m, w, yerr=e, capsize=4, color=c, label=lab)
            for xi, (mv, ev) in zip(x + off, series):
                txt = f"{mv:.2f}" if mv < 10 else f"{mv:.1f}"
                ax.text(xi, mv + ev, txt, ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in NETS])
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(REG_TITLE[ci], fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(v[0] + v[1] for v in ospf + sa + ma) * 1.22)
        if ci == 0:
            ax.set_ylabel(ylab, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"OSPF vs single-agent GNN vs MARL — ns-3 {ylab.split(' (')[0].lower()} "
                 "across three real backbones\n"
                 "mean ± std over model seeds 0/1/2; overload | feasible, all topologies on one axis",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


def main():
    make("loss", "ospf_loss", "gnn_loss", "Packet loss (%)",
         f"{R}/fig_qos_loss_regime.png")
    make("delay", "ospf_delay_ms", "gnn_delay_ms", "Mean delay (ms)",
         f"{R}/fig_qos_delay_regime.png")


if __name__ == "__main__":
    main()
