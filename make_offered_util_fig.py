#!/usr/bin/env python3
"""Link-utilisation figure using OFFERED LOAD (uncapped) rather than ns-3 carried
utilisation. Offered load = sum of the flow rates a routing places on a link / capacity,
maxed over links = the true demand on the bottleneck. Unlike ns-3 carried utilisation
(which caps at 100% because a real link cannot exceed capacity — the excess becomes loss),
offered load exceeds 100% under overload, so it shows exactly how far past capacity OSPF
pushes its worst link (e.g. 120%) versus how much headroom the learned routers keep.

Computed exactly from the per-matrix routing files (verified == stored ospf_util):
  results/ns3_eval_realsa_fresh_<net>_s<seed>/routing_seed*.json   (ospf_path + SA gnn_path)
  results/ns3_eval_realmarl_fresh_<net>_s<seed>/routing_seed*.json (MARL gnn_path)
Two panels: overload | feasible. x = topology, bars = OSPF / SA-GNN / MARL, mean ± std
over model seeds 0/1/2 x matrices. A dashed line marks the 100% capacity ceiling.
"""
import json
import os
import numpy as np
from collections import defaultdict
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


def caps(net):
    t = json.load(open(f"topologies/{net}.json"))
    c = {}
    for l in t["links"]:
        c[(l["src"], l["dst"])] = l["capacity"]
        c[(l["dst"], l["src"])] = l["capacity"]
    return c


def offered_maxutil(flows, pathkey, cap):
    load = defaultdict(float)
    for f in flows:
        p = f[pathkey]
        for a, b in zip(p, p[1:]):
            load[(a, b)] += f["rate_mbps"]
    return max(load[e] / cap[e] * 100 for e in load)


def collect(net):
    """-> {regime: {method: [util,...]}} over all seeds x matrices."""
    cap = caps(net)
    out = {r: {m: [] for m in ("ospf", "sa", "marl")} for r in REGIMES}
    for s in SEEDS:
        # OSPF + SA-GNN from the realsa dirs
        d = f"{R}/ns3_eval_realsa_fresh_{net}_s{s}"
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.startswith("routing_seed"):
                    r = json.load(open(f"{d}/{fn}"))
                    reg = r["regime"]
                    out[reg]["ospf"].append(offered_maxutil(r["flows"], "ospf_path", cap))
                    out[reg]["sa"].append(offered_maxutil(r["flows"], "gnn_path", cap))
        # MARL from the realmarl dirs
        d = f"{R}/ns3_eval_realmarl_fresh_{net}_s{s}"
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.startswith("routing_seed"):
                    r = json.load(open(f"{d}/{fn}"))
                    out[r["regime"]]["marl"].append(offered_maxutil(r["flows"], "gnn_path", cap))
    return out


def mean_std(a):
    a = np.array(a)
    return (float(a.mean()), float(a.std())) if len(a) else (0.0, 0.0)


def main():
    data = {net: collect(net) for net, _ in NETS}
    x = np.arange(len(NETS))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ci, regime in enumerate(REGIMES):
        ax = axes[ci]
        series = {m: [mean_std(data[net][regime][m]) for net, _ in NETS]
                  for m in ("ospf", "sa", "marl")}
        for off, m, c, lab in [(-w, "ospf", OSPF_C, "OSPF"),
                               (0, "sa", SA_C, "single-agent GNN"),
                               (w, "marl", MA_C, "MARL")]:
            mv = [v[0] for v in series[m]]
            ev = [v[1] for v in series[m]]
            ax.bar(x + off, mv, w, yerr=ev, capsize=4, color=c, label=lab)
            for xi, (m0, e0) in zip(x + off, series[m]):
                ax.text(xi, m0 + e0, f"{m0:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.axhline(100, color="#333333", lw=1.2, ls="--", zorder=0)
        ax.text(ax.get_xlim()[1], 100, " capacity", color="#333333",
                fontsize=8, va="bottom", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in NETS])
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(REG_TITLE[ci], fontsize=11, fontweight="bold")
        top = max(v[0] + v[1] for m in series for v in series[m])
        ax.set_ylim(0, max(top * 1.15, 110))
        if ci == 0:
            ax.set_ylabel("Max offered load on bottleneck link (% of capacity)",
                          fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Offered load on the bottleneck link — OSPF vs single-agent GNN vs MARL\n"
                 "true demand as % of capacity (uncapped; >100% = OSPF overloads a link); "
                 "mean ± std over seeds 0/1/2",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = f"{R}/fig_offered_util_regime.png"
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")
    # also dump the numbers
    for net, _ in NETS:
        for regime in REGIMES:
            row = {m: mean_std(data[net][regime][m]) for m in ("ospf", "sa", "marl")}
            print(f"{net:18s} {regime:9s} "
                  + "  ".join(f"{m} {v[0]:.0f}±{v[1]:.0f}%" for m, v in row.items()))


if __name__ == "__main__":
    main()
