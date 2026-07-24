#!/usr/bin/env python3
"""Export per-flow routing JSONs for a TOPOLOGY-AGNOSTIC policy on a held-out network,
so the zero-shot policy can be judged at packet level by run_ns3_phase2.py exactly like
the per-topology models were.

Writes results/<out>/routing_seed<N>.json  (same schema as evaluate_ns3.py:
seed, regime, ospf_util, flows[{src,dst,rate_mbps,start,stop,gnn_path,ospf_path}]).
Then:  python run_ns3_phase2.py --dir results/<out> --topo <topo> --ratescale 20

Load band is wide on purpose so BOTH regimes (overload & feasible) are populated,
matching the main results' stratification.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx
from stable_baselines3 import PPO

from marl_routing.graph_routing_env import GraphSeqRoutingEnv
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_gnn_extractor import TopoAgnosticGNNExtractor  # noqa: F401 (registers class)
from marl_routing.topology import load as load_topology

ap = argparse.ArgumentParser()
ap.add_argument("--topo", default="germany50_sndlib", help="held-out topology to judge")
ap.add_argument("--model", required=True, help="path to topoagn policy .zip")
ap.add_argument("--loads", default="10,15,20,35,50,65",
                help="magnitude scales on real demand (wide -> both regimes)")
ap.add_argument("--n-per-scale", type=int, default=5)
ap.add_argument("--max-flows", type=int, default=200,
                help="keep top-N flows by rate (germany50 has 2450 pairs; ns-3 tractability)")
ap.add_argument("--n-overload", type=int, default=3)
ap.add_argument("--n-feasible", type=int, default=3)
ap.add_argument("--k-paths", type=int, default=3)
ap.add_argument("--out", required=True, help="output dir under results/")
A = ap.parse_args()


def filt_rates(r, k):
    r = np.asarray(r, dtype=float)
    if k and (r > 0).sum() > k:
        thr = np.sort(r)[::-1][k - 1]
        r = np.where(r >= thr, r, 0.0)
    return r


def arcpath_to_nodes(arc_path, arcs):
    """[a0,a1,...] (arc indices) -> [n0,n1,...] node path."""
    if not arc_path:
        return []
    nodes = [arcs[arc_path[0]][0]]
    for a in arc_path:
        nodes.append(arcs[a][1])
    return nodes


def main():
    n = load_topology(A.topo).n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    loads = [float(x) for x in A.loads.split(",")]
    mats = real_matrices(A.topo, pairs, loads, n_per_scale=A.n_per_scale, split="test")
    cand = {i: filt_rates(mats[i], A.max_flows) for i in range(len(mats))}

    env = GraphSeqRoutingEnv([(A.topo, pairs, mats)], k_paths=A.k_paths, seed=0)
    b = env.bundles[0]

    # stratify by OSPF analytical bottleneck (same rule as evaluate_ns3.py)
    util = {i: round(b.ospf_max_util(r), 1) for i, r in cand.items()}
    by = sorted(util, key=lambda i: -util[i])
    overload = [i for i in by if util[i] >= 100][:A.n_overload]
    feasible = [i for i in by if util[i] < 100][:A.n_feasible]
    sel = overload + feasible
    print(f"[stratify] overload {[(i, util[i]) for i in overload]}")
    print(f"[stratify] feasible {[(i, util[i]) for i in feasible]}")

    model = PPO.load(A.model, device="cpu")
    out = Path(f"results/{A.out}"); out.mkdir(parents=True, exist_ok=True)

    for i in sel:
        rates = cand[i]
        env.set_matrix(rates, 0)
        obs, _ = env.reset()
        chosen = {}
        done = False
        while not done:
            pi = env.order[env.pos]
            a, _ = model.predict(obs, deterministic=True)
            chosen[pi] = int(a) % A.k_paths
            obs, _, done, _, _ = env.step(int(a))

        flows = []
        for pi, (s, d) in enumerate(pairs):
            if rates[pi] <= 0:
                continue
            k = min(chosen.get(pi, 0), len(b.pair_arc_paths[pi]) - 1)
            gnn_nodes = arcpath_to_nodes(b.pair_arc_paths[pi][k], b.arcs)
            flows.append({
                "src": int(s), "dst": int(d), "rate_mbps": float(rates[pi]),
                "start": 2.0, "stop": 18.0,
                "gnn_path": [int(x) for x in gnn_nodes],
                "ospf_path": [int(x) for x in nx.shortest_path(b.topo.graph, s, d)],
            })
        regime = "overload" if util[i] >= 100 else "feasible"
        (out / f"routing_seed{i}.json").write_text(json.dumps(
            {"seed": i, "regime": regime, "ospf_util": util[i], "flows": flows}, indent=2))
        print(f"  [{regime:8}] matrix {i} (OSPF {util[i]:.0f}%): {len(flows)} flows exported")

    print(f"\n[done] wrote {len(sel)} routing JSONs to {out}")
    print(f"  now run: python run_ns3_phase2.py --dir {out} --topo {A.topo} --ratescale 20")


if __name__ == "__main__":
    main()
