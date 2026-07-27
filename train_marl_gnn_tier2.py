#!/usr/bin/env python3
"""Tier 2 generalisation: train the GNN-actor MARL on MANY SNDlib topologies (real nominal
demand, load-scaled, no synthetic noise), then test ZERO-SHOT on the 3 core backbones using
their REAL MEASURED dynamic traffic.

Design rationale: the nominal SNDlib matrices give structural diversity for TRAINING (where
demand-quality matters least); the pristine measured time-series (Abilene-Zhang, GEANT-Uhlig,
Germany50-DFN) are reserved for TEST (where rigour matters most). The policy never sees a
single measured matrix in training, yet is judged on hundreds of them.

  python train_marl_gnn_tier2.py --seed 0 --updates 200 --tag _tier2
"""
import argparse
import json
from pathlib import Path

import numpy as np

from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.nominal_traffic import nominal_matrices
from marl_routing.real_traffic import real_matrices
from marl_routing.tmgen_traffic import tmgen_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topology import load as load_topology

# 17 SNDlib topologies that fit the env padding (<=56 nodes, <=176 arcs, <=12 degree)
TRAIN_TOPOS = [
    "atlanta_sndlib", "cost266_sndlib", "dfn-bwin_sndlib", "dfn-gwin_sndlib",
    "di-yuan_sndlib", "france_sndlib", "india35_sndlib", "janos-us_sndlib",
    "newyork_sndlib", "nobel-eu_sndlib", "nobel-germany_sndlib", "nobel-us_sndlib",
    "norway_sndlib", "pdh_sndlib", "polska_sndlib", "ta1_sndlib", "zib54_sndlib",
]
# zero-shot test: the 3 core backbones with REAL MEASURED dynamic traffic
TEST_TOPOS = ["abilene_sndlib", "geant_sndlib", "germany50_sndlib"]
TEST_LOADS = {  # magnitude scaling of the measured matrices, into feasible+overload
    "abilene_sndlib": [8.0, 12.0, 16.0],
    "geant_sndlib": [3.0, 5.0, 7.0],
    "germany50_sndlib": [35.0, 50.0, 65.0],
}

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--updates", type=int, default=200)
ap.add_argument("--rollout", type=int, default=4096)
ap.add_argument("--hidden", type=int, default=32)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--stretch", type=int, default=2)
ap.add_argument("--delay-penalty", type=float, default=0.5)
ap.add_argument("--max-flows", type=int, default=500,
                help="cap flows/matrix to top-N by rate for tractable episodes (0=off)")
ap.add_argument("--traffic", choices=["nominal", "tmgen"], default="nominal",
                help="training traffic source on the 17 SNDlib topos: real nominal demand "
                     "(default) or TMgen modulated-gravity (for the ablation isolating "
                     "traffic-type from topo-set)")
ap.add_argument("--tag", default="_tier2")
A = ap.parse_args()


def pairs_of(topo):
    n = load_topology(topo).n_nodes
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def _cap(mats, k):
    if not k:
        return mats
    out = []
    for m in mats:
        r = np.asarray(m, float).copy()
        if (r > 0).sum() > k:
            r[r < np.sort(r)[::-1][k - 1]] = 0.0
        out.append(r)
    return out


def evaluate(env, mats, policy, label):
    ospf, ecmp, marl = [], [], []
    for m in mats:
        o = env.ospf_max_util(m, 0)
        e = env.ecmp_max_util(m, 0)
        env.set_matrix(m, 0)
        obs, mask, _ = env.reset()
        done = False; info = {}
        while not done:
            obs, mask, _, _, done, info = env.step(policy(env))
        ospf.append(o); ecmp.append(e); marl.append(info["max_util"])
    ospf = np.array(ospf); ecmp = np.array(ecmp); marl = np.array(marl)
    out = {}
    for regime, sel in [("overload", ospf >= 100), ("feasible", ospf < 100)]:
        if sel.sum() == 0:
            continue
        o = ospf[sel]; e = ecmp[sel]; g = marl[sel]
        out[regime] = {"ospf_mean": float(o.mean()), "ecmp_mean": float(e.mean()),
                       "marl_mean": float(g.mean()),
                       "delta_ospf_pt": float(o.mean() - g.mean()),
                       "delta_ecmp_pt": float(e.mean() - g.mean()),
                       "win_ospf_pct": float((g < o).mean() * 100),
                       "win_ecmp_pct": float((g < e).mean() * 100), "n": int(sel.sum())}
        print(f"  [{label:20s} {regime:8s}] OSPF {o.mean():6.1f}%  ECMP {e.mean():6.1f}%  "
              f"MARL {g.mean():6.1f}%  vs-OSPF {o.mean()-g.mean():+.1f}pt  "
              f"vs-ECMP {e.mean()-g.mean():+.1f}pt  (n={sel.sum()})", flush=True)
    return out


def main():
    out = Path(f"results/marlgnn{A.tag}_seed{A.seed}"); out.mkdir(parents=True, exist_ok=True)

    # ---- training specs: real nominal demand or TMgen gravity, load-scaled ----
    train_specs = []
    for t in TRAIN_TOPOS:
        p = pairs_of(t)
        if A.traffic == "tmgen":
            raw = tmgen_matrices(t, p, n_patterns=3,
                                 load_scales=(0.6, 0.8, 1.0, 1.2, 1.5), seed=A.seed)
        else:
            raw = nominal_matrices(t, p, seed=A.seed)
        train_specs.append((t, p, _cap(raw, A.max_flows)))
    env = TopoAgnosticMARLEnv(train_specs, seed=A.seed, delay_penalty=A.delay_penalty,
                              stretch=A.stretch, normalize_reward=True)
    print(f"[tier2] train on {len(TRAIN_TOPOS)} topos ({A.traffic} traffic), "
          f"zero-shot test on {TEST_TOPOS}; seed={A.seed} updates={A.updates}", flush=True)

    mappo = GNNMAPPO(env, hidden=A.hidden, rounds=A.rounds, rollout_steps=A.rollout,
                     n_epochs=6, minibatch=512, seed=A.seed)
    mappo.learn(total_steps=A.rollout * A.updates, log_every=5,
                ckpt_dir=out / "ckpts", ckpt_every=40)
    mappo.save(out / "policy.pt")
    print(f"[saved] {out/'policy.pt'}", flush=True)

    # ---- zero-shot eval on the 3 core backbones, REAL MEASURED traffic ----
    policy = mappo.act_fn(True)
    results = {"train_topos": TRAIN_TOPOS, "test_topos": TEST_TOPOS,
               "seed": A.seed, "zero_shot": {}}
    print("\n[eval] ZERO-SHOT on core backbones (real measured test traffic):", flush=True)
    for t in TEST_TOPOS:
        p = pairs_of(t)
        mats = real_matrices(t, p, TEST_LOADS[t], n_per_scale=6, split="test")
        ev = TopoAgnosticMARLEnv([(t, p, mats)], seed=A.seed + 1,
                                 delay_penalty=A.delay_penalty, stretch=A.stretch,
                                 normalize_reward=True)
        results["zero_shot"][t] = evaluate(ev, mats, policy, t)

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
