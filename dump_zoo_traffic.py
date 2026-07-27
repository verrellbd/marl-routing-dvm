#!/usr/bin/env python3
"""Persist the TMgen training traffic for a given seed as a reproducible artifact.

Regenerates the EXACT training traffic used by train_marl_gnn_zoo.py (deterministic from
seed) and writes it to results/marlgnn_zoo_seed<seed>/traffic/. Use this to snapshot the
traffic for a run that was launched before --save-traffic existed.

  python dump_zoo_traffic.py --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from marl_routing.tmgen_traffic import tmgen_matrices, save_traffic
from marl_routing.topology import load as load_topology

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-patterns", type=int, default=3)
ap.add_argument("--loads", default="0.6,0.8,1.0,1.2,1.5")
ap.add_argument("--max-flows", type=int, default=300)
ap.add_argument("--tag", default="_zoo")
A = ap.parse_args()

load_scales = tuple(float(x) for x in A.loads.split(","))
names = json.load(open("topologies/zoo_manifest.json"))


def cap_flows(mats, k):
    if not k:
        return mats
    out = []
    for m in mats:
        r = np.asarray(m, float).copy()
        if (r > 0).sum() > k:
            r[r < np.sort(r)[::-1][k - 1]] = 0.0
        out.append(r)
    return out


specs = []
for i, t in enumerate(names):
    n = load_topology(t).n_nodes
    p = [(s, d) for s in range(n) for d in range(n) if s != d]
    mats = cap_flows(tmgen_matrices(t, p, n_patterns=A.n_patterns,
                                    load_scales=load_scales, seed=A.seed), A.max_flows)
    specs.append((t, p, mats))
    if (i + 1) % 50 == 0:
        print(f"  regenerated {i+1}/{len(names)}", flush=True)

out = Path(f"results/marlgnn{A.tag}_seed{A.seed}/traffic")
meta = {"generator": "tmgen.modulated_gravity_tm", "seed": A.seed,
        "n_patterns": A.n_patterns, "load_scales": load_scales,
        "max_flows": A.max_flows, "target_util": 100.0,
        "spatial_variance": 600.0, "mean_traffic": 1000.0,
        "n_topos": len(names), "note": "regenerated post-hoc; identical to training (seed-deterministic)"}
save_traffic(specs, out, meta)
print(f"[done] wrote {len(specs)} topologies' training traffic to {out}")
print(f"       total size: ", end="")
sz = sum(f.stat().st_size for f in out.glob('*.npz'))
print(f"{sz/1e6:.1f} MB")
