#!/usr/bin/env python3
"""Decompose the MARL episode return into its congestion and delay terms.

This is the source for tab:reward-decomp, which previously had no committed script.

The split is read straight out of the telescoping identity (eq:telescope):

    G = -100 * U_T / U_OSPF   -   beta * D

so with the episode return G and the final bottleneck U_T both known, the congestion
term is computed directly and the delay term is whatever is left over. The detour count
follows as D = -delay / beta. Nothing is re-derived from the paths, so the numbers
cannot drift from the reward the agent actually optimised -- and the identity is
checked rather than assumed: D must come out an integer, and --validate asserts it.

Evaluation is the trainers' own TEST_LOADS with n_per_scale=6, i.e. eighteen measured
matrices per network, on the three held-out backbones. This is a wider set than the
three-per-regime the packet-level tables use, because the decomposition describes the
policy's behaviour rather than a regime comparison.

  python decompose_reward.py --seeds 10 --out results/reward_decomp.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topology import load as load_topology

TEST_LOADS = {
    "abilene_sndlib":   [8.0, 12.0, 16.0],
    "geant_sndlib":     [3.0, 5.0, 7.0],
    "germany50_sndlib": [35.0, 50.0, 65.0],
}
MODEL = "results/marlgnn_tier2m15cm_seed{s}/policy.pt"
BETA = 0.5

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--max-flows", type=int, default=200)
ap.add_argument("--out", default="results/reward_decomp.json")
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


out = {}
for topo, loads in TEST_LOADS.items():
    p = pairs_of(topo)
    mats = cap(real_matrices(topo, p, loads, n_per_scale=6, split="test"), A.max_flows)
    short = topo.replace("_sndlib", "")
    per_seed = []
    for s in range(A.seeds):
        mp = Path(MODEL.format(s=s))
        if not mp.exists():
            print(f"[skip] {mp}")
            continue
        env = TopoAgnosticMARLEnv([(topo, p, mats)], seed=1 + s,
                                  normalize_reward=True, metric="weighted")
        mappo = GNNMAPPO(env, hidden=32, rounds=3, seed=s)
        mappo.ac.load_state_dict(torch.load(mp, map_location="cpu"))
        act = mappo.act_fn(True)
        rows = []
        for m in mats:
            env.set_matrix(m, 0)
            obs, mask, _ = env.reset()
            G, done, info = 0.0, False, {}
            while not done:
                obs, mask, _, r, done, info = env.step(act(env))
                G += r
            u_t, u_ospf = info["max_util"], env.ospf_ref
            # info["paths"] is Dict[flow_index, [nodes]]
            hops = sum(len(q) - 1 for q in info["paths"].values())
            congestion = -100.0 * u_t / u_ospf
            delay = G - congestion                  # eq:telescope, by construction
            detours = -delay / BETA
            # the identity says this must be a whole number of hops
            assert abs(detours - round(detours)) < 1e-6, f"non-integer detours {detours}"
            rows.append((G, congestion, delay, round(detours), hops, u_t / u_ospf))
        per_seed.append(np.mean(rows, axis=0))
        print(f"  {short:10} seed {s}: G={per_seed[-1][0]:8.1f}  "
              f"cong={per_seed[-1][1]:7.1f}  delay={per_seed[-1][2]:7.1f}  "
              f"detours={per_seed[-1][3]:6.1f}", flush=True)

    a = np.array(per_seed)
    out[short] = {
        "return":        round(float(a[:, 0].mean()), 1),
        "return_sd":     round(float(a[:, 0].std()), 1),
        "congestion":    round(float(a[:, 1].mean()), 1),
        "delay":         round(float(a[:, 2].mean()), 1),
        "detour_hops":   round(float(a[:, 3].mean()), 1),
        "detour_pct":    round(100.0 * float((a[:, 3] / a[:, 4]).mean()), 1),
        "util_ratio":    round(float(a[:, 5].mean()), 2),
        "seeds":         len(per_seed),
    }
    print(f"  -> {short}: {out[short]}\n", flush=True)

Path(A.out).write_text(json.dumps(out, indent=1))
print(f"wrote {A.out}")
