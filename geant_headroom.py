#!/usr/bin/env python3
"""Does giving more candidate paths (k) open routing headroom on GEANT?
Compares OSPF vs a greedy best-response path selector at k=3,5,8 on the
overload matrices. If greedy@k5/k8 drops max-util well below OSPF, retraining
with more paths is worth it. If even k=8 ties OSPF, GEANT is capacity-limited.
Vectorized candidate precompute + numpy load accumulation for speed."""
import numpy as np
import networkx as nx
from marl_routing.traffic import generate_matrix
from marl_routing.topology import load as load_topology
from marl_routing.gnn_routing_agent import compute_ksp

topo = load_topology('geant'); n = topo.n_nodes
pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
arcs = list(topo.graph.edges()); ai = {a: k for k, a in enumerate(arcs)}
cap = np.array([topo.graph[u][v]['capacity'] for u, v in arcs], float)
nA = len(arcs)


def arc_path(p):
    return [ai[(p[i], p[i + 1])] for i in range(len(p) - 1)]


def ospf_util(rates):
    load = np.zeros(nA)
    for k, (s, d) in enumerate(pairs):
        if rates[k] <= 0:
            continue
        for a in arc_path(nx.shortest_path(topo.graph, s, d)):
            load[a] += rates[k]
    return (100 * load / cap).max()


def greedy_util(rates, K, cand_cache):
    cand = cand_cache[K]
    active = [k for k in range(len(pairs)) if rates[k] > 0]
    load = np.zeros(nA)
    ch = {}
    # init: first candidate
    for k in active:
        ch[k] = 0
        for a in cand[k][0]:
            load[a] += rates[k]
    for _ in range(6):
        improved = False
        for k in active:
            # remove current
            for a in cand[k][ch[k]]:
                load[a] -= rates[k]
            best_j, best_m = ch[k], np.inf
            for j in range(len(cand[k])):
                tmp = load.copy()
                for a in cand[k][j]:
                    tmp[a] += rates[k]
                m = (100 * tmp / cap).max()
                if m < best_m - 1e-9:
                    best_m, best_j = m, j
            ch[k] = best_j
            for a in cand[k][best_j]:
                load[a] += rates[k]
            if best_j != ch[k]:
                improved = True
        if not improved:
            break
    return (100 * load / cap).max()


# precompute candidate arc-paths per k once (shared across seeds)
cand_cache = {}
for K in (3, 5, 8):
    cand = []
    for (s, d) in pairs:
        ps = compute_ksp(topo.graph, s, d, k=K)
        cp = [arc_path(p) for p in ps]
        while len(cp) < K:
            cp.append(cp[-1])
        cand.append(cp)
    cand_cache[K] = cand
    print(f"[precomputed k={K}]", flush=True)

print("\nGEANT headroom @ load 1.5 (analytical max link-util %):")
print("  seed   OSPF   greedy@k3  greedy@k5  greedy@k8")
for seed in [1005, 1008, 1013]:
    r = np.array([generate_matrix('geant', 1.5, seed=seed)[a, b] for a, b in pairs])
    o = ospf_util(r)
    g3 = greedy_util(r, 3, cand_cache)
    g5 = greedy_util(r, 5, cand_cache)
    g8 = greedy_util(r, 8, cand_cache)
    print(f"  {seed}  {o:5.0f}   {g3:7.0f}   {g5:8.0f}   {g8:8.0f}", flush=True)
