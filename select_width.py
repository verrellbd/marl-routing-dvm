#!/usr/bin/env python3
"""Select the MARL hidden width on the TRAINING topologies, not the test backbones.

Choosing h32 over h64 because it scores better on abilene/geant/germany50 would be
test-set selection, and those three networks carry the zero-shot claim. This script
instead evaluates both widths on HELD-OUT TMgen matrices drawn from the 17 TRAINING
topologies (a different generator seed from the one used in training), which is a
legitimate model-selection signal.

  python select_width.py --seeds 0,1,2 --out results/width_selection.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.tmgen_traffic import tmgen_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topology import load as load_topology

TRAIN_TOPOS = [
    "atlanta_sndlib", "cost266_sndlib", "dfn-bwin_sndlib", "dfn-gwin_sndlib",
    "di-yuan_sndlib", "france_sndlib", "india35_sndlib", "janos-us_sndlib",
    "newyork_sndlib", "nobel-eu_sndlib", "nobel-germany_sndlib", "nobel-us_sndlib",
    "norway_sndlib", "pdh_sndlib", "polska_sndlib", "ta1_sndlib", "zib54_sndlib",
]
ARMS = [("h32", "results/marlgnn_tier2m15cm_seed{s}/policy.pt", 32),
        ("h64", "results/marlgnn_tier2m15h64cm_seed{s}/policy.pt", 64)]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2")
# SELECTION SEED: must differ from the training seeds (0/1/2) so the matrices are
# genuinely held out rather than the ones the policy was fitted on.
ap.add_argument("--gen-seed", type=int, default=777)
ap.add_argument("--n-patterns", type=int, default=2)
ap.add_argument("--max-flows", type=int, default=500)
ap.add_argument("--out", default="results/width_selection.json")
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
    seeds = [int(x) for x in A.seeds.split(",")]
    res = {"gen_seed": A.gen_seed, "topos": TRAIN_TOPOS, "per_arm": {}}

    # matrices are identical for every arm and seed -> paired comparison
    specs = {}
    for t in TRAIN_TOPOS:
        p = pairs_of(t)
        mats = cap(tmgen_matrices(t, p, n_patterns=A.n_patterns,
                                  load_scales=(0.8, 1.0, 1.3), seed=A.gen_seed), A.max_flows)
        specs[t] = (p, mats)

    for name, tmpl, hidden in ARMS:
        per_seed = []
        for s in seeds:
            mp = Path(tmpl.format(s=s))
            if not mp.exists():
                print(f"[skip] {mp} missing", flush=True)
                continue
            deltas, ratios = [], []
            for t in TRAIN_TOPOS:
                p, mats = specs[t]
                env = TopoAgnosticMARLEnv([(t, p, mats)], seed=1000 + s,
                                          normalize_reward=True, metric="weighted")
                mappo = GNNMAPPO(env, hidden=hidden, rounds=3, seed=s)
                mappo.ac.load_state_dict(torch.load(mp, map_location="cpu"))
                act = mappo.act_fn(True)
                for m in mats:
                    o = env.ospf_max_util(m, 0)
                    # act_fn consumes the env itself (see GNNMAPPO.act_fn), so drive the
                    # episode the same way the trainer's own evaluate() does.
                    env.set_matrix(m, 0)
                    env.reset()
                    done, info = False, {}
                    while not done:
                        _, _, _, _, done, info = env.step(act(env))
                    g = info["max_util"]
                    deltas.append(o - g)          # percentage points saved vs OSPF
                    ratios.append(g / max(o, 1e-9))
            per_seed.append({"seed": s, "mean_delta_pt": float(np.mean(deltas)),
                             "mean_ratio": float(np.mean(ratios)), "n": len(deltas)})
            print(f"[{name} seed {s}] mean {np.mean(deltas):+.2f}pt vs OSPF "
                  f"(ratio {np.mean(ratios):.3f}, n={len(deltas)})", flush=True)
        if per_seed:
            d = [x["mean_delta_pt"] for x in per_seed]
            res["per_arm"][name] = {"per_seed": per_seed,
                                    "mean_delta_pt": float(np.mean(d)),
                                    "std_delta_pt": float(np.std(d))}

    Path(A.out).write_text(json.dumps(res, indent=2))
    print("\n=== SELECTION ON TRAINING TOPOLOGIES (higher = better) ===")
    for k, v in res["per_arm"].items():
        print(f"  MARL {k}: {v['mean_delta_pt']:+.2f} +/- {v['std_delta_pt']:.2f} pt vs OSPF")
    if len(res["per_arm"]) == 2:
        win = max(res["per_arm"], key=lambda k: res["per_arm"][k]["mean_delta_pt"])
        print(f"  -> select {win}")
    print(f"-> {A.out}")


if __name__ == "__main__":
    main()
