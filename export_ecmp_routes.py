#!/usr/bin/env python3
"""Export per-flow ECMP routing JSONs so the ECMP baseline can be judged at packet level
by the same ns-3 scenario as OSPF / single / MARL (abilene-validate.cc).

The scenario installs one explicit path per flow, so this exports **flow-level (hashed)
ECMP**: each flow is assigned one of its equal-cost shortest paths, chosen by a seeded
hash. That is what real routers do (5-tuple hashing keeps a flow on one path to preserve
packet ordering). It differs from the ANALYTICAL ecmp_max_util, which splits each demand
fractionally across all equal-cost paths (a fluid model) — state this in the write-up.

The ECMP path is written under the key "gnn_path" purely so the existing scenario reads
it unchanged; ospf_path is also written, so ns-3 OSPF numbers can be re-derived if wanted.

  python export_ecmp_routes.py --topo geant_sndlib --loads 3,5,7 --seed 0 \
      --out ns3m_ecmp_geant_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx

from marl_routing.graph_routing_env import GraphSeqRoutingEnv
from marl_routing.real_traffic import real_matrices
from marl_routing.ospf_metric import (weighted_graph, shortest_path,
                                      all_shortest_paths, max_util)
from marl_routing.topology import load as load_topology

ap = argparse.ArgumentParser()
ap.add_argument("--topo", default="geant_sndlib")
ap.add_argument("--loads", default="3,5,7")
ap.add_argument("--n-per-scale", type=int, default=6)
ap.add_argument("--max-flows", type=int, default=200)
ap.add_argument("--n-overload", type=int, default=3)
ap.add_argument("--n-feasible", type=int, default=3)
ap.add_argument("--seed", type=int, default=0, help="ECMP hash seed (path tie-breaking)")
ap.add_argument("--metric", choices=["hop", "weighted"], default="hop",
                help="cost metric for equal-cost sets AND stratification. Real ECMP\nsplits among paths of equal OSPF COST, not equal hop count")
ap.add_argument("--out", required=True)
A = ap.parse_args()


def filt_rates(r, k):
    r = np.asarray(r, dtype=float)
    if k and (r > 0).sum() > k:
        thr = np.sort(r)[::-1][k - 1]
        r = np.where(r >= thr, r, 0.0)
    return r


def main():
    n = load_topology(A.topo).n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    loads = [float(x) for x in A.loads.split(",")]
    mats = real_matrices(A.topo, pairs, loads, n_per_scale=A.n_per_scale, split="test")
    cand = {i: filt_rates(mats[i], A.max_flows) for i in range(len(mats))}

    env = GraphSeqRoutingEnv([(A.topo, pairs, mats)], k_paths=3, seed=0, metric=A.metric)
    b = env.bundles[0]
    G = b.topo.graph

    # identical stratification to the other arms -> same matrices judged
    W = weighted_graph(G)
    util = {i: round(max_util(G, W, pairs, b.arc_index, b.cap, r, A.metric), 1)
            for i, r in cand.items()}
    by = sorted(util, key=lambda i: -util[i])
    overload = [i for i in by if util[i] >= 100][:A.n_overload]
    feasible = [i for i in by if util[i] < 100][:A.n_feasible]
    sel = overload + feasible
    print(f"[stratify] overload {[(i, util[i]) for i in overload]}")
    print(f"[stratify] feasible {[(i, util[i]) for i in feasible]}")

    # precompute equal-cost path sets once per pair (expensive on big graphs)
    rng = np.random.RandomState(A.seed)
    ecmp_choice = {}
    for pi, (s, d) in enumerate(pairs):
        paths = all_shortest_paths(G, W, s, d, A.metric)
        ecmp_choice[pi] = paths[rng.randint(len(paths))] if len(paths) > 1 else paths[0]
    multi = sum(1 for pi, (s, d) in enumerate(pairs)
                if len(all_shortest_paths(G, W, s, d, A.metric)) > 1)
    print(f"[ecmp] {multi}/{len(pairs)} pairs have >1 equal-cost path (seed {A.seed})")

    out = Path(f"results/{A.out}"); out.mkdir(parents=True, exist_ok=True)
    for i in sel:
        rates = cand[i]
        flows = []
        for pi, (s, d) in enumerate(pairs):
            if rates[pi] <= 0:
                continue
            flows.append({
                "src": int(s), "dst": int(d), "rate_mbps": float(rates[pi]),
                "start": 2.0, "stop": 18.0,
                "gnn_path": [int(x) for x in ecmp_choice[pi]],   # = ECMP path
                "ospf_path": [int(x) for x in shortest_path(G, W, s, d, A.metric)],
            })
        regime = "overload" if util[i] >= 100 else "feasible"
        (out / f"routing_seed{i}.json").write_text(json.dumps(
            {"seed": i, "regime": regime, "ospf_util": util[i], "routing_kind": "ecmp",
             "flows": flows}, indent=2))
        print(f"  [{regime:8}] matrix {i} (OSPF {util[i]:.0f}%): {len(flows)} flows exported")
    print(f"\n[done] wrote {len(sel)} ECMP routing JSONs to {out}")


if __name__ == "__main__":
    main()
