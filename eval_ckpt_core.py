#!/usr/bin/env python3
"""Evaluate saved checkpoints of a zoo run on the 3 core backbones (overload) to trace
how zero-shot performance evolved over training — plateau check + Germany50 diagnosis."""
import argparse, glob, re
import numpy as np
from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topology import load as load_topology

TEST = {"abilene_sndlib": [8.0, 12.0, 16.0], "geant_sndlib": [3.0, 5.0, 7.0],
        "germany50_sndlib": [35.0, 50.0, 65.0]}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/marlgnn_zoo_seed0")
ap.add_argument("--max-flows", type=int, default=0, help="cap eval flows (0=unfiltered)")
ap.add_argument("--topos", default="germany50_sndlib")
A = ap.parse_args()


def pairs_of(t):
    n = load_topology(t).n_nodes
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def cap(m, k):
    if not k: return m
    r = np.asarray(m, float).copy()
    if (r > 0).sum() > k: r[r < np.sort(r)[::-1][k - 1]] = 0.0
    return r


evs = {}
for t in A.topos.split(","):
    p = pairs_of(t)
    mats = [cap(m, A.max_flows) for m in real_matrices(t, p, TEST[t], n_per_scale=6, split="test")]
    ev = TopoAgnosticMARLEnv([(t, p, mats)], seed=1, normalize_reward=True)
    evs[t] = (ev, mats)

ckpts = sorted(glob.glob(f"{A.dir}/ckpts/ckpt_upd*.pt"),
               key=lambda s: int(re.search(r"upd(\d+)", s).group(1)))
ckpts.append(f"{A.dir}/policy.pt")

# build a throwaway mappo to hold the net arch (uses one core env)
t0 = list(evs)[0]
holder = GNNMAPPO(evs[t0][0], hidden=32, rounds=3, seed=0)

for ck in ckpts:
    holder.load(ck)
    pol = holder.act_fn(True)
    tag = re.search(r"(upd\d+|policy)", ck).group(1)
    parts = []
    for t, (ev, mats) in evs.items():
        o, g = [], []
        for m in mats:
            ospf = ev.ospf_max_util(m, 0); ev.set_matrix(m, 0)
            obs, mask, _ = ev.reset(); done = False; info = {}
            while not done:
                obs, mask, _, _, done, info = ev.step(pol(ev))
            if ospf >= 100: o.append(ospf); g.append(info["max_util"])
        if o:
            o, g = np.array(o), np.array(g)
            parts.append(f"{t.split('_')[0]}:{(o-g).mean():+.1f}pt(win{(g<o).mean()*100:.0f}%)")
    print(f"  {tag:10s}  " + "   ".join(parts), flush=True)
