#!/usr/bin/env python3
"""
M4 — export the trained MARL (MAPPO) policy's per-flow paths for ns-3 validation.

Mirrors evaluate_ns3.py's stratify+export, but the paths come from rolling out the
multi-agent policy (each node-agent's hop choices) instead of an SB3 model. Writes the
same routing-JSON schema the ns-3 scenario already reads — the MARL path goes in the
`gnn_path` slot, OSPF shortest path in `ospf_path` — so run_ns3_phase2.py /
abilene-validate.cc run UNCHANGED.

  python export_marl_routing.py --topo abilene --load 4.0 \
      --model results/abilene_marl_v1_seed0/mappo_actor_critic.pt --tag _marl_seed0
  python run_ns3_phase2.py --dir results/ns3_eval_marl_seed0 --topo abilene
"""
import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx

from marl_routing.traffic import generate_matrix
from marl_routing.topology import load as load_topology
from marl_routing.multiagent_routing_env import MultiAgentRoutingEnv
from marl_routing.mappo import MAPPO

_ap = argparse.ArgumentParser()
_ap.add_argument("--topo", default="abilene")
_ap.add_argument("--load", type=float, default=4.0)
_ap.add_argument("--model", required=True, help="path to mappo_actor_critic.pt")
_ap.add_argument("--tag", default="_marl")
_ap.add_argument("--candidate-seeds", default="1000-1019")
_ap.add_argument("--n-overload", type=int, default=3)
_ap.add_argument("--n-feasible", type=int, default=3)
_ap.add_argument("--stretch", type=int, default=1)
_ap.add_argument("--max-flows", type=int, default=0,
                 help="keep only top-N flows by rate (0=all); for big topologies (germany50)")
_ARGS = _ap.parse_args()


def filt_rates(r, k):
    r = np.asarray(r, dtype=float)
    if k and (r > 0).sum() > k:
        thr = np.sort(r)[::-1][k - 1]
        r = np.where(r >= thr, r, 0.0)
    return r

TOPO = _ARGS.topo
lo, hi = (int(x) for x in _ARGS.candidate_seeds.split("-"))
CAND = list(range(lo, hi + 1))
OUT = Path(f"~/thesis/results/ns3_eval{_ARGS.tag}").expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)


def main():
    topo = load_topology(TOPO); n = topo.n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    env = MultiAgentRoutingEnv(TOPO, pairs, [np.zeros(len(pairs))], stretch=_ARGS.stretch)
    mappo = MAPPO(env); mappo.load(_ARGS.model)
    act = mappo.act_fn(True)

    # rate vector per seed (top-N filtered) + OSPF util (for stratification)
    rates = {s: filt_rates(np.array([generate_matrix(TOPO, _ARGS.load, seed=s)[a, b]
                                     for a, b in pairs]), _ARGS.max_flows)
             for s in CAND}
    util = {s: round(env.ospf_max_util(rates[s]), 1) for s in CAND}
    by = sorted(CAND, key=lambda s: -util[s])
    overload = [s for s in by if util[s] >= 100][:_ARGS.n_overload]
    feasible = [s for s in by if util[s] < 100][:_ARGS.n_feasible]  # most-stressed feasible
    test_seeds = overload + feasible
    print(f"[stratify] overload {[(s, util[s]) for s in overload]}  "
          f"feasible {[(s, util[s]) for s in feasible]}")

    for seed in test_seeds:
        r = rates[seed]
        paths = env.rollout_paths(act, r)              # MARL hop-by-hop paths
        regime = "overload" if util[seed] >= 100 else "feasible"
        flows = []
        for pi, (s, d) in enumerate(pairs):
            if r[pi] <= 0:
                continue
            flows.append({
                "src": int(s), "dst": int(d), "rate_mbps": float(r[pi]),
                "start": 2.0, "stop": 18.0,
                "gnn_path": [int(x) for x in paths[pi]],            # <- MARL path
                "ospf_path": [int(x) for x in nx.shortest_path(topo.graph, s, d)],
            })
        rf = OUT / f"routing_seed{seed}.json"
        rf.write_text(json.dumps({"seed": seed, "regime": regime,
                                  "ospf_util": util[seed], "marl_util": round(env.cur_max, 1),
                                  "flows": flows}, indent=2))
        print(f"  seed {seed} [{regime}] OSPF {util[seed]:.0f}% -> MARL {env.cur_max:.0f}%  "
              f"({len(flows)} flows) -> {rf.name}")
    print(f"\n[export] {len(test_seeds)} routing JSONs to {OUT}")
    print(f"  now run: python run_ns3_phase2.py --dir {OUT} --topo {TOPO}")


if __name__ == "__main__":
    main()
