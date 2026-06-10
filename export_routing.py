#!/usr/bin/env python3
"""
Export per-flow routing (GNN-chosen path + OSPF shortest path) for one traffic
matrix, so ns-3 can install the exact paths and measure utilization + packet loss.

Picks a held-out test matrix where OSPF overloads (>100% analytical max-util), so
ns-3 will show real packet loss under OSPF that the GNN avoids.
"""
import json
from pathlib import Path

import numpy as np
import networkx as nx
from stable_baselines3 import PPO

from marl_routing.traffic import generate_matrix
from marl_routing.topology import load as load_topology
from marl_routing.sequential_routing_env import MultiTrafficSequentialEnv, CAND_FEATS
from marl_routing.gnn_routing_agent import compute_ksp

TOPO = "abilene"
LOAD_FACTOR = 3.0
K_PATHS = 3
MODEL = "results/generalization/gnn_generalist"
OUT = Path("results/generalization/ns3_routing.json")


def main():
    topo = load_topology(TOPO)
    n = topo.n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    # node-sequence candidates per pair (same order the env uses)
    pair_paths = [compute_ksp(topo.graph, s, d, k=K_PATHS) for (s, d) in pairs]

    # pick the MOST overloaded held-out test matrix (seeds 1000-1019) for the
    # strongest OSPF-vs-GNN loss contrast in ns-3
    env = MultiTrafficSequentialEnv(TOPO, pairs, [np.zeros(len(pairs))], k_paths=K_PATHS)
    chosen_seed, rates, best_mu = None, None, -1.0
    for seed in range(1000, 1020):
        T = generate_matrix(TOPO, LOAD_FACTOR, seed=seed)
        r = np.array([T[s, d] for (s, d) in pairs])
        mu = env.ospf_max_util(r)
        if mu > best_mu:
            best_mu, chosen_seed, rates = mu, seed, r

    ospf_mu = env.ospf_max_util(rates)
    print(f"Selected test seed {chosen_seed}: OSPF analytical max-util={ospf_mu:.1f}%")

    # run the GNN deterministically, record chosen path index per processed pair
    model = PPO.load(MODEL, device="cpu")
    env.set_matrix(rates)
    obs, _ = env.reset()
    base = env.n_arcs + env.topo.n_nodes ** 2
    chosen_idx = {}
    done = False
    while not done:
        pair_i = env.order[env.pos]
        a, _ = model.predict(obs, deterministic=True)
        chosen_idx[pair_i] = int(a) % K_PATHS
        obs, _, done, _, info = env.step(a)
    gnn_mu = info["max_util"]
    print(f"GNN analytical max-util={gnn_mu:.1f}%")

    # build flow list with both routings (only non-zero flows)
    flows = []
    for pi, (s, d) in enumerate(pairs):
        if rates[pi] <= 0:
            continue
        gnn_path = pair_paths[pi][chosen_idx.get(pi, 0)]
        ospf_path = nx.shortest_path(topo.graph, s, d)
        flows.append({
            "src": int(s), "dst": int(d), "rate_mbps": float(rates[pi]),
            "start": 2.0, "stop": 18.0,
            "gnn_path": [int(x) for x in gnn_path],
            "ospf_path": [int(x) for x in ospf_path],
        })

    OUT.write_text(json.dumps({
        "seed": chosen_seed, "load_factor": LOAD_FACTOR,
        "analytical": {"ospf_max_util": round(ospf_mu, 2), "gnn_max_util": round(gnn_mu, 2)},
        "n_flows": len(flows), "flows": flows,
    }, indent=2))
    print(f"Wrote {len(flows)} flows (both routings) -> {OUT}")


if __name__ == "__main__":
    main()
