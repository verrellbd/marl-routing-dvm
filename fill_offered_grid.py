#!/usr/bin/env python3
"""Fill the two empty cells of the offered-load table so both regimes exist for all
three evaluation networks.

At the trainers' TEST_LOADS, Abilene is entirely feasible (a correctly costed OSPF is
never overloaded there below ~16) and Germany50 entirely overloaded. The packet-level
grid already solves this by measuring Abilene at 16/22/28 and Germany50's feasible
regime at 15-30; this script applies the same alternate loads to the ANALYTICAL
objective so the two tables cover the same six cells.

  python fill_offered_grid.py --out results/offered_grid.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from marl_routing.graph_routing_env import GraphSeqRoutingEnv
from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.ospf_metric import all_shortest_paths
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topology import load as load_topology

# (topo, loads, which regime this run is for) -- covers all six cells
CELLS = [
    ("abilene_sndlib",   [8.0, 12.0, 16.0],        "feasible"),
    ("abilene_sndlib",   [16.0, 22.0, 28.0],       "overload"),
    ("geant_sndlib",     [3.0, 5.0, 7.0],          "both"),
    ("germany50_sndlib", [35.0, 50.0, 65.0],       "overload"),
    ("germany50_sndlib", [15.0, 20.0, 25.0, 30.0], "feasible"),
]
ARMS = [("single",  "results/single_singleH64gRM_seed{s}/policy.zip", None),
        ("marlh32", "results/marlgnn_tier2m15cm_seed{s}/policy.pt",     32),
        ("marlh64", "results/marlgnn_tier2m15h64cm_seed{s}/policy.pt",  64)]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/offered_grid.json")
ap.add_argument("--max-flows", type=int, default=200)
# arms with fewer trained seeds than this are skipped per-seed by the exists() check
ap.add_argument("--seeds", type=int, default=3)
A = ap.parse_args()


def pairs_of(t):
    n = load_topology(t).n_nodes
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def cap(mats, k):
    out = []
    for m in mats:
        r = np.asarray(m, float).copy()
        if (r > 0).sum() > k:
            r[r < np.sort(r)[::-1][k - 1]] = 0.0
        out.append(r)
    return out


def main():
    acc = {}   # (topo, regime, arm) -> list of per-seed means
    for topo, loads, want in CELLS:
        p = pairs_of(topo)
        mats = cap(real_matrices(topo, p, loads, n_per_scale=6, split="test"), A.max_flows)
        ref = GraphSeqRoutingEnv([(topo, p, mats)], k_paths=3, seed=0, metric="weighted",
                                 reward_form="marl")
        ospf = np.array([ref.ospf_max_util(m, 0) for m in mats])
        # ECMP at FLOW granularity (one tied path per demand, chosen by a hash seed),
        # matching what is installed in ns-3. The fluid ecmp_max_util() splits each
        # demand fractionally and is a different, stronger policy -- using it here would
        # make the analytical and packet-level tables describe different mechanisms.
        b = ref.bundles[0]
        allsp = [all_shortest_paths(b.topo.graph, b.W, s, d, "weighted") for (s, d) in p]
        # One hash seed per reported seed, so ECMP carries the same dispersion here as
        # in the packet-level grid. Averaging the hash seeds into a single value (as an
        # earlier version did) made ECMP the only arm quoted without a spread.
        ecmp_seeds = []
        for hseed in range(A.seeds):
            rng = np.random.RandomState(hseed)
            arcp = []
            for ps in allsp:
                path = ps[rng.randint(len(ps))] if len(ps) > 1 else ps[0]
                arcp.append([b.arc_index[(path[j], path[j + 1])] for j in range(len(path) - 1)])
            vals = []
            for m in mats:
                load = np.zeros(b.n_arcs)
                for i, ap in enumerate(arcp):
                    if m[i] > 0:
                        load[ap] += m[i]
                vals.append(float((100.0 * load / b.cap).max()))
            ecmp_seeds.append(np.array(vals))
        short = topo.replace("_sndlib", "")
        for regime, sel in [("overload", ospf >= 100), ("feasible", ospf < 100)]:
            if want not in (regime, "both") or sel.sum() == 0:
                continue
            acc[(short, regime, "ospf")] = [float(ospf[sel].mean())]
            acc[(short, regime, "ecmp")] = [float(e[sel].mean()) for e in ecmp_seeds]

        for arm, tmpl, hidden in ARMS:
            per_seed = {"overload": [], "feasible": []}
            for s in range(A.seeds):
                mp = Path(tmpl.format(s=s))
                if not mp.exists():
                    print(f"[skip] {mp}"); continue
                vals = []
                if hidden is None:
                    env = GraphSeqRoutingEnv([(topo, p, mats)], k_paths=3, seed=1 + s,
                                             metric="weighted", reward_form="marl")
                    model = PPO.load(str(mp), device="cpu")
                    for m in mats:
                        env.set_matrix(m, 0)
                        obs, _ = env.reset(); done = False; info = {}
                        while not done:
                            a, _ = model.predict(obs, deterministic=True)
                            obs, _, done, _, info = env.step(int(a))
                        vals.append(info["max_util"])
                else:
                    env = TopoAgnosticMARLEnv([(topo, p, mats)], seed=1 + s,
                                              normalize_reward=True, metric="weighted")
                    mappo = GNNMAPPO(env, hidden=hidden, rounds=3, seed=s)
                    mappo.ac.load_state_dict(torch.load(mp, map_location="cpu"))
                    act = mappo.act_fn(True)
                    for m in mats:
                        env.set_matrix(m, 0); env.reset()
                        done, info = False, {}
                        while not done:
                            _, _, _, _, done, info = env.step(act(env))
                        vals.append(info["max_util"])
                vals = np.array(vals)
                for regime, sel in [("overload", ospf >= 100), ("feasible", ospf < 100)]:
                    if want in (regime, "both") and sel.sum():
                        per_seed[regime].append(float(vals[sel].mean()))
            for regime, v in per_seed.items():
                # a grid built at N seeds carries only arms that have N seeds
                if v and len(v) < A.seeds:
                    print(f"  [drop] {short}/{regime} {arm}: only {len(v)} seeds")
                elif v:
                    acc[(short, regime, arm)] = v
                    print(f"  {short}/{regime:8s} {arm:8s} {np.mean(v):7.1f} +/- {np.std(v):4.1f} "
                          f"(OSPF {acc[(short,regime,'ospf')][0]:6.1f})", flush=True)

    out = {}
    for (t, r, a), v in acc.items():
        out.setdefault(f"{t}/{r}", {})[a] = {"mean": round(float(np.mean(v)), 1),
                                             "sd": round(float(np.std(v)), 1),
                                             "seeds": len(v)}
    Path(A.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {A.out}")


if __name__ == "__main__":
    main()
