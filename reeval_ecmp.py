#!/usr/bin/env python3
"""Re-evaluate a saved tier2 policy on the 3 core backbones with OSPF + ECMP + MARL,
overwriting its summary.json. Used to backfill ECMP into seed-0 (trained before ECMP was
added to the eval). Policy-loading + eval only — no training.

  python reeval_ecmp.py --dir results/marlgnn_tier2tmgen_seed0
"""
import argparse, json
from pathlib import Path
import numpy as np
from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topology import load as load_topology

TEST_LOADS = {"abilene_sndlib": [8., 12., 16.], "geant_sndlib": [3., 5., 7.],
              "germany50_sndlib": [35., 50., 65.]}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
A = ap.parse_args()


def pairs_of(t):
    n = load_topology(t).n_nodes
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def evaluate(env, mats, policy):
    ospf, ecmp, marl = [], [], []
    for m in mats:
        ospf.append(env.ospf_max_util(m, 0)); ecmp.append(env.ecmp_max_util(m, 0))
        env.set_matrix(m, 0)
        obs, mask, _ = env.reset(); done = False; info = {}
        while not done:
            obs, mask, _, _, done, info = env.step(policy(env))
        marl.append(info["max_util"])
    ospf, ecmp, marl = map(np.array, (ospf, ecmp, marl))
    out = {}
    for regime, sel in [("overload", ospf >= 100), ("feasible", ospf < 100)]:
        if sel.sum() == 0:
            continue
        o, e, g = ospf[sel], ecmp[sel], marl[sel]
        out[regime] = {"ospf_mean": float(o.mean()), "ecmp_mean": float(e.mean()),
                       "marl_mean": float(g.mean()),
                       "delta_ospf_pt": float(o.mean() - g.mean()),
                       "delta_ecmp_pt": float(e.mean() - g.mean()),
                       "win_ospf_pct": float((g < o).mean() * 100),
                       "win_ecmp_pct": float((g < e).mean() * 100), "n": int(sel.sum())}
        print(f"  [{regime:8s}] OSPF {o.mean():6.1f}% ECMP {e.mean():6.1f}% MARL {g.mean():6.1f}%"
              f"  vs-OSPF {o.mean()-g.mean():+.1f} vs-ECMP {e.mean()-g.mean():+.1f} (n={sel.sum()})",
              flush=True)
    return out


d = Path(A.dir)
summ = json.loads((d / "summary.json").read_text())
holder = None
for t in TEST_LOADS:
    p = pairs_of(t)
    mats = real_matrices(t, p, TEST_LOADS[t], n_per_scale=6, split="test")
    ev = TopoAgnosticMARLEnv([(t, p, mats)], seed=summ.get("seed", 0) + 1, normalize_reward=True)
    if holder is None:
        holder = GNNMAPPO(ev, hidden=32, rounds=3, seed=0)
        holder.load(d / "policy.pt")
    print(f"[{t}]")
    summ["zero_shot"][t] = evaluate(ev, mats, holder.act_fn(True))
(d / "summary.json").write_text(json.dumps(summ, indent=2))
print(f"[saved] {d/'summary.json'} (now includes ECMP)")
