#!/usr/bin/env python3
"""
Aggregate ns-3 PACKET-LEVEL max-link-utilization (max_offered_util_pct, measured from
bytes enqueued per device in the ns-3 sim) into the §4.1 utilisation table — so §4.1 is
ns-3-sourced, not analytical. The per-sim JSONs already carry this field; run_ns3_phase2.py
just never propagated it. This reads the existing files, no re-simulation.

Layout it reads (produced by eval_ns3_all_monaco.sh):
  results/ns3_eval_realsa_fresh_<net>_s<seed>/     -> SA-GNN routing  (ns3_gnn_*)  + OSPF (ns3_ospf_*)
  results/ns3_eval_realmarl_fresh_<net>_s<seed>/   -> MARL routing    (ns3_gnn_*)  + OSPF (ns3_ospf_*)
Each dir: routing_seed<M>.json (has "regime") + ns3_{ospf,gnn}_<M>.json (has max_offered_util_pct).

Aggregation: per (network, method) collect max_offered_util_pct over the 3 training seeds ×
the overload test matrices, report mean ± std. OSPF is routing-independent, so it is pooled
from the realsa dirs (identical routing in realmarl).
"""
import json
import glob
import os
import numpy as np

# ns-3 measures bytes carried over window=simTime(8s), but flows are active only [2s, 9s]
# = 7s (sim ends at simTime+1). So a link at 100% line rate reads 7/8 = 0.875. Divide by
# this exact, uniform factor (all flows share [2,18] clipped to 9s) to report TRUE
# utilisation. Verified: distinct (start,stop) across every flow = {(2.0, 18.0)}.
ACTIVE_FRAC = 7.0 / 8.0

NETS = [("abilene_sndlib", "Abilene (12n)"),
        ("geant_sndlib", "GEANT (22n)"),
        ("germany50_sndlib", "Germany50 (50n)")]
SEEDS = [0, 1, 2]
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def regime_of(d, m):
    return json.load(open(f"{d}/routing_seed{m}.json")).get("regime", "feasible")


def util(fp):
    return json.load(open(fp))["max_offered_util_pct"] / ACTIVE_FRAC  # -> true utilisation


def collect(net, meth_tag, who, regime):
    """max_offered_util_pct list over seeds×matrices for one (net, method, regime)."""
    vals = []
    for s in SEEDS:
        d = f"{RESULTS}/ns3_eval_{meth_tag}_fresh_{net}_s{s}"
        for rf in glob.glob(f"{d}/routing_seed*.json"):
            m = os.path.basename(rf).replace("routing_seed", "").replace(".json", "")
            if regime_of(d, m) != regime:
                continue
            vals.append(util(f"{d}/ns3_{who}_{m}.json"))
    return np.array(vals)


def ms(a):
    return f"{a.mean():5.1f} ± {a.std():4.1f}" if len(a) else "   n/a"


def main():
    for regime in ("overload", "feasible"):
        print(f"\n### ns-3 max-link-utilisation (%), {regime} regime "
              f"(mean ± std over seeds×matrices)")
        print(f"| {'Network':16} | {'OSPF':>12} | {'SA-GNN':>12} | {'MARL':>12} |")
        print(f"|{'-'*18}|{'-'*14}|{'-'*14}|{'-'*14}|")
        table = {}
        for net, label in NETS:
            ospf = collect(net, "realsa", "ospf", regime)      # OSPF pooled from SA dirs
            sa = collect(net, "realsa", "gnn", regime)          # SA-GNN routing
            marl = collect(net, "realmarl", "gnn", regime)      # MARL routing
            table[net] = {"ospf": ospf, "sa": sa, "marl": marl}
            print(f"| {label:16} | {ms(ospf):>12} | {ms(sa):>12} | {ms(marl):>12} |")
        # machine-readable dump
        out = {net: {k: {"mean": float(v.mean()) if len(v) else None,
                         "std": float(v.std()) if len(v) else None,
                         "n": int(len(v))} for k, v in d.items()}
               for net, d in table.items()}
        (open(f"{RESULTS}/ns3_util_summary_{regime}.json", "w")
         .write(json.dumps(out, indent=2)))
    print(f"\n-> wrote results/ns3_util_summary_overload.json, _feasible.json")


if __name__ == "__main__":
    main()
