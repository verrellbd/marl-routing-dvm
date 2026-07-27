#!/usr/bin/env python3
"""Large-scale generalisation: train the GNN-actor MARL on the FULL Topology Zoo (245
networks, <=100 nodes, all except the 3 evaluation backbones) with modulated-gravity
traffic (TMgen), then test ZERO-SHOT on Abilene / GEANT / Germany50 using their REAL
MEASURED dynamic traffic.

This is the strong generalisation protocol: hundreds of real topologies with realistic
synthetic traffic for TRAINING; pristine measured traffic on 3 held-out real backbones
for TEST. The policy sees none of the 3 eval networks and no measured matrices in training.

  python train_marl_gnn_zoo.py --seed 0 --updates 300 --tag _zoo
"""
import argparse
import json
from pathlib import Path

import numpy as np

from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.real_traffic import real_matrices
from marl_routing.tmgen_traffic import tmgen_matrices, save_traffic
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topology import load as load_topology

TEST_TOPOS = ["abilene_sndlib", "geant_sndlib", "germany50_sndlib"]
TEST_LOADS = {
    "abilene_sndlib": [8.0, 12.0, 16.0],
    "geant_sndlib": [3.0, 5.0, 7.0],
    "germany50_sndlib": [35.0, 50.0, 65.0],
}

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--updates", type=int, default=300)
ap.add_argument("--rollout", type=int, default=4096)
ap.add_argument("--hidden", type=int, default=32)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--stretch", type=int, default=2)
ap.add_argument("--delay-penalty", type=float, default=0.5)
ap.add_argument("--n-patterns", type=int, default=3, help="modulated-gravity draws / topo")
ap.add_argument("--loads", default="0.6,0.8,1.0,1.2,1.5",
                help="load-scale sweep; matches the earlier runs' feasible->overload span "
                     "(OSPF ~60%..150%)")
ap.add_argument("--max-flows", type=int, default=300,
                help="cap flows/matrix to top-N by rate (dense gravity TMs -> huge episodes)")
ap.add_argument("--max-topos", type=int, default=0, help=">0 to subsample topos (debug)")
ap.add_argument("--save-traffic", action="store_true",
                help="persist the generated training traffic to results/<out>/traffic/ "
                     "as a reproducible artifact")
ap.add_argument("--tag", default="_zoo")
A = ap.parse_args()


def pairs_of(topo):
    n = load_topology(topo).n_nodes
    return [(s, d) for s in range(n) for d in range(n) if s != d]


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


def evaluate(env, mats, policy, label):
    ospf, marl = [], []
    for m in mats:
        o = env.ospf_max_util(m, 0)
        env.set_matrix(m, 0)
        obs, mask, _ = env.reset()
        done = False; info = {}
        while not done:
            obs, mask, _, _, done, info = env.step(policy(env))
        ospf.append(o); marl.append(info["max_util"])
    ospf = np.array(ospf); marl = np.array(marl)
    out = {}
    for regime, sel in [("overload", ospf >= 100), ("feasible", ospf < 100)]:
        if sel.sum() == 0:
            continue
        o = ospf[sel]; g = marl[sel]
        out[regime] = {"ospf_mean": float(o.mean()), "marl_mean": float(g.mean()),
                       "delta_pt": float(o.mean() - g.mean()),
                       "win_pct": float((g < o).mean() * 100), "n": int(sel.sum())}
        print(f"  [{label:18s} {regime:8s}] OSPF {o.mean():6.1f}% -> MARL {g.mean():6.1f}%"
              f"  delta {o.mean()-g.mean():+6.1f}pt  win {out[regime]['win_pct']:.0f}%"
              f" (n={sel.sum()})", flush=True)
    return out


def main():
    out = Path(f"results/marlgnn{A.tag}_seed{A.seed}"); out.mkdir(parents=True, exist_ok=True)
    load_scales = tuple(float(x) for x in A.loads.split(","))

    names = json.load(open("topologies/zoo_manifest.json"))
    if A.max_topos:
        names = names[:A.max_topos]
    print(f"[zoo] building {len(names)} training topologies (modulated-gravity)...", flush=True)
    train_specs = []
    for i, t in enumerate(names):
        p = pairs_of(t)
        mats = cap_flows(tmgen_matrices(t, p, n_patterns=A.n_patterns,
                                        load_scales=load_scales, seed=A.seed), A.max_flows)
        train_specs.append((t, p, mats))
        if (i + 1) % 50 == 0:
            print(f"  built {i+1}/{len(names)}", flush=True)

    if A.save_traffic:
        meta = {"generator": "tmgen.modulated_gravity_tm", "seed": A.seed,
                "n_patterns": A.n_patterns, "load_scales": load_scales,
                "max_flows": A.max_flows, "target_util": 100.0,
                "spatial_variance": 600.0, "mean_traffic": 1000.0,
                "n_topos": len(names), "note": "regenerable from seed via tmgen_matrices"}
        d = save_traffic(train_specs, out / "traffic", meta)
        print(f"[save-traffic] wrote reproducible training traffic to {d}", flush=True)

    env = TopoAgnosticMARLEnv(train_specs, seed=A.seed, delay_penalty=A.delay_penalty,
                              stretch=A.stretch, normalize_reward=True)
    print(f"[zoo] train on {len(names)} Zoo topos, zero-shot test on {TEST_TOPOS}; "
          f"seed={A.seed} updates={A.updates}", flush=True)

    mappo = GNNMAPPO(env, hidden=A.hidden, rounds=A.rounds, rollout_steps=A.rollout,
                     n_epochs=6, minibatch=512, seed=A.seed)
    mappo.learn(total_steps=A.rollout * A.updates, log_every=10,
                ckpt_dir=out / "ckpts", ckpt_every=50)
    mappo.save(out / "policy.pt")
    print(f"[saved] {out/'policy.pt'}", flush=True)

    policy = mappo.act_fn(True)
    results = {"n_train_topos": len(names), "test_topos": TEST_TOPOS, "seed": A.seed,
               "zero_shot": {}}
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
