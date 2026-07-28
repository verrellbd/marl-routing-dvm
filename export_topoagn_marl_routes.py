#!/usr/bin/env python3
"""Export per-flow routing JSONs for a TOPOLOGY-AGNOSTIC *MARL* policy on a held-out
network, so the zero-shot MARL policy can be judged at packet level by run_ns3_phase2.py
exactly like the single-agent topoagn policy (export_topoagn_routes.py).

Same output schema as export_topoagn_routes.py / evaluate_ns3.py:
  results/<out>/routing_seed<N>.json
  {seed, regime, ospf_util, flows[{src,dst,rate_mbps,start,stop,gnn_path,ospf_path}]}
("gnn_path" keeps the established key name so run_ns3_phase2.py works unchanged; here it
carries the MARL-chosen hop-by-hop route.)

  python export_topoagn_marl_routes.py --topo germany50_sndlib \
      --model results/marlgnn_tier2m15h64_seed0/policy.pt --hidden 64 --rounds 3 \
      --out ns3_m15h64_germany50_s0
  python run_ns3_phase2.py --dir results/ns3_m15h64_germany50_s0 --topo germany50_sndlib
"""
import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx

from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.ospf_metric import weighted_graph, shortest_path, max_util
from marl_routing.topology import load as load_topology

ap = argparse.ArgumentParser()
ap.add_argument("--topo", default="germany50_sndlib", help="held-out topology to judge")
ap.add_argument("--model", required=True, help="path to MARL policy.pt")
ap.add_argument("--loads", default="10,15,20,35,50,65",
                help="magnitude scalings of the measured matrices")
ap.add_argument("--n-per-scale", type=int, default=5)
ap.add_argument("--max-flows", type=int, default=200,
                help="keep top-K flows by rate (ns-3 tractability); 0 = all")
ap.add_argument("--n-overload", type=int, default=3)
ap.add_argument("--n-feasible", type=int, default=3)
ap.add_argument("--hidden", type=int, default=32, help="must match the trained policy")
ap.add_argument("--rounds", type=int, default=3, help="must match the trained policy")
ap.add_argument("--stretch", type=int, default=2)
ap.add_argument("--metric", choices=["hop", "weighted"], default="hop",
                help="OSPF cost metric for the baseline path + stratification")
ap.add_argument("--out", required=True, help="output dir under results/")
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

    env = TopoAgnosticMARLEnv([(A.topo, pairs, mats)], seed=0, stretch=A.stretch,
                              metric=A.metric,
                              normalize_reward=True)
    b = env.bundles[0]

    # stratify by OSPF analytical bottleneck (same rule as export_topoagn_routes.py)
    W = weighted_graph(b.G)
    util = {i: round(max_util(b.G, W, pairs, b.arc_index, b.cap, r, A.metric), 1)
            for i, r in cand.items()}
    by = sorted(util, key=lambda i: -util[i])
    overload = [i for i in by if util[i] >= 100][:A.n_overload]
    feasible = [i for i in by if util[i] < 100][:A.n_feasible]
    sel = overload + feasible
    print(f"[stratify] overload {[(i, util[i]) for i in overload]}")
    print(f"[stratify] feasible {[(i, util[i]) for i in feasible]}")

    mappo = GNNMAPPO(env, hidden=A.hidden, rounds=A.rounds, seed=0)
    mappo.load(A.model)
    policy = mappo.act_fn(True)

    out = Path(f"results/{A.out}"); out.mkdir(parents=True, exist_ok=True)

    for i in sel:
        rates = cand[i]
        env.set_matrix(rates, 0)
        obs, mask, _ = env.reset()
        done = False; info = {}
        while not done:
            obs, mask, _, _, done, info = env.step(policy(env))
        paths = info["paths"]          # {pair_index: [node, node, ...]}

        flows = []
        for pi, (s, d) in enumerate(pairs):
            if rates[pi] <= 0:
                continue
            marl_nodes = paths.get(pi)
            if not marl_nodes:          # pair not routed this episode -> fall back to SP
                marl_nodes = nx.shortest_path(b.G, s, d)
            flows.append({
                "src": int(s), "dst": int(d), "rate_mbps": float(rates[pi]),
                "start": 2.0, "stop": 18.0,
                "gnn_path": [int(x) for x in marl_nodes],
                "ospf_path": [int(x) for x in shortest_path(b.G, W, s, d, A.metric)],
            })
        regime = "overload" if util[i] >= 100 else "feasible"
        (out / f"routing_seed{i}.json").write_text(json.dumps(
            {"seed": i, "regime": regime, "ospf_util": util[i], "flows": flows}, indent=2))
        print(f"  [{regime:8}] matrix {i} (OSPF {util[i]:.0f}%): {len(flows)} flows exported")

    print(f"\n[done] wrote {len(sel)} routing JSONs to {out}")
    print(f"  now run: python run_ns3_phase2.py --dir {out} --topo {A.topo}")


if __name__ == "__main__":
    main()
