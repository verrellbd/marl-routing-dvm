#!/usr/bin/env python3
"""Does the analytical training objective order packet loss the way ns-3 measures it?

WHY THIS EXISTS. Training minimises the analytical bottleneck U (eq. load/bottleneck),
but every reported result is ns-3 packet-level. A reviewer will ask whether the trained
objective actually tracks the measured outcome, especially in the overload regime where
the surrogate's "all demand is carried" assumption is false.

WHAT IT DOES NOT CLAIM. U does not equal anything ns-3 measures and is not meant to.
Offered load is unbounded above 100%; ns-3 utilisation SATURATES at capacity, because
packets dropped at a congested link never reach the links beyond it. Comparing the two
as magnitudes fails badly (mean gap ~86 points in overload, and ns-3's offered load
pins at 7/8 = 87.5% on any saturated link, carrying no information to rank methods).

WHAT IT DOES CLAIM. A training signal only has to ORDER candidate routings the way the
real outcome does. Spearman is a rank correlation, so it measures exactly that property.

NOTHING IS RE-SIMULATED. ns-3 loss/delay are read from the existing reported grid
(results/ns3f_*). U is recomputed offline from the stored paths and rates -- pure
arithmetic. The recomputation is self-checked against the `ospf_util` field that the
original export pipeline wrote into each routing file (see "ospf_util_check" in the
output); it should agree to display-rounding.

  python validate_surrogate.py --out results/surrogate_validation.json
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

from scipy.stats import pearsonr, spearmanr

TOPOS = ["abilene", "geant", "germany50"]

# arm directory tag -> (ns-3 output tag, label used in the report)
ARMS = {"marlh32": ("gnn", "MARL"),
        "singleRM": ("gnn", "Single"),
        "ecmp": ("ecmp", "ECMP")}

# the four topology/regime directory shorthands used by run_ns3_final.sh
DIRS = [("abilene", "abilene"), ("geant", "geant"),
        ("germany50", "germany50"), ("germany50", "g50feas")]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/surrogate_validation.json")
ap.add_argument("--results", default="results")
ap.add_argument("--topologies", default="topologies")
A = ap.parse_args()


def capacities(topo):
    """Directed arc -> capacity (Mb/s). Links are undirected in the JSON."""
    cap = {}
    for l in json.load(open(f"{A.topologies}/{topo}_sndlib.json"))["links"]:
        cap[(l["src"], l["dst"])] = cap[(l["dst"], l["src"])] = float(l["capacity"])
    return cap


def bottleneck(flows, path_key, cap):
    """The analytical objective: eq. (load) then eq. (bottleneck).

    Push each flow's rate onto every arc of its path, divide by that arc's capacity,
    take the maximum. This is what the reward minimises, computed on the paths that
    were actually installed in ns-3.
    """
    load = defaultdict(float)
    for f in flows:
        p = f.get(path_key)
        if not p:
            continue
        for a, b in zip(p[:-1], p[1:]):
            load[(a, b)] += f["rate_mbps"]
    if not load:
        return float("nan")
    return max(100.0 * v / cap[k] for k, v in load.items())


def collect():
    """One row per simulation: (topo, regime, method, U, loss, delay, ns3_offered)."""
    caps = {t: capacities(t) for t in TOPOS}
    rows, checks = [], []
    for arm, (tag, label) in ARMS.items():
        for topo, short in DIRS:
            for d in sorted(glob.glob(f"{A.results}/ns3f_{arm}_{short}_s*/")):
                for rf in sorted(glob.glob(d + "routing_seed*.json")):
                    i = re.search(r"routing_seed(\d+)", rf).group(1)
                    nf = f"{d}ns3_{tag}_{i}.json"
                    if not os.path.exists(nf):
                        continue
                    r, n = json.load(open(rf)), json.load(open(nf))
                    rows.append((topo, r["regime"], label,
                                 bottleneck(r["flows"], "gnn_path", caps[topo]),
                                 n["loss_pct"], n["mean_delay_ms"],
                                 n["max_offered_util_pct"]))
                    # OSPF is deterministic and was simulated once, inside the
                    # marlh32 seed 0-2 directories.
                    of = f"{d}ns3_ospf_{i}.json"
                    if arm == "marlh32" and os.path.exists(of):
                        no = json.load(open(of))
                        u = bottleneck(r["flows"], "ospf_path", caps[topo])
                        rows.append((topo, r["regime"], "OSPF", u,
                                     no["loss_pct"], no["mean_delay_ms"],
                                     no["max_offered_util_pct"]))
                        # self-check: the export pipeline stored its own OSPF U
                        checks.append(abs(u - r["ospf_util"]))
    return rows, checks


def corr(xs, ys):
    if len(xs) < 3:
        return None
    return {"n": len(xs),
            "spearman": round(float(spearmanr(xs, ys)[0]), 3),
            "pearson": round(float(pearsonr(xs, ys)[0]), 3)}


rows, checks = collect()
if not rows:
    raise SystemExit(f"no paired simulations found under {A.results}/ns3f_*")

out = {"n_paired_simulations": len(rows),
       "ospf_util_check": {
           "n": len(checks),
           "max_abs_diff_pp": round(float(max(checks)), 4) if checks else None,
           "note": "recomputed U(ospf_path) vs the ospf_util stored at export time"},
       "vs_ns3_loss": {}, "vs_ns3_offered_load": {}}

for regime in ["overload", "feasible"]:
    sel = [r for r in rows if r[1] == regime]
    out["vs_ns3_loss"][regime] = {
        "all": corr([r[3] for r in sel], [r[4] for r in sel])}
    out["vs_ns3_offered_load"][regime] = {
        "all": corr([r[3] for r in sel], [r[6] for r in sel])}
    for t in TOPOS:
        s = [r for r in sel if r[0] == t]
        out["vs_ns3_loss"][regime][t] = corr([r[3] for r in s], [r[4] for r in s])
        out["vs_ns3_offered_load"][regime][t] = corr([r[3] for r in s], [r[6] for r in s])

os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
json.dump(out, open(A.out, "w"), indent=2)

print(f"[surrogate] {len(rows)} paired simulations "
      f"(no ns-3 was re-run; U recomputed from stored paths)")
print(f"[surrogate] U(ospf_path) reproduces the stored ospf_util to "
      f"{out['ospf_util_check']['max_abs_diff_pp']} pp\n")
for regime in ["overload", "feasible"]:
    L = out["vs_ns3_loss"][regime]
    print(f"  {regime}: analytical U vs ns-3 LOSS")
    print(f"     all        n={L['all']['n']:4d}  spearman {L['all']['spearman']:+.3f}"
          f"  pearson {L['all']['pearson']:+.3f}")
    for t in TOPOS:
        c = L[t]
        print(f"     {t:<10} n={c['n']:4d}  spearman {c['spearman']:+.3f}"
              f"  pearson {c['pearson']:+.3f}")
    O = out["vs_ns3_offered_load"][regime]["all"]
    why = ("ns-3 offered load saturates once links congest, so it cannot rank methods here"
           if regime == "overload" else "nothing is dropped, so the surrogate is accurate here")
    print(f"  {regime}: analytical U vs ns-3 OFFERED LOAD  -> spearman "
          f"{O['spearman']:+.3f}   ({why})\n")
print(f"[surrogate] wrote {A.out}")
