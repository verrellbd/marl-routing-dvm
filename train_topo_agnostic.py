#!/usr/bin/env python3
"""Train ONE topology-agnostic GNN routing policy across several networks.

Unlike train_gnn_qos.py (one model per topology, flat n_nodes^2 adjacency input),
this trains a single shared-weight policy on a MIX of topologies and can then be
evaluated ZERO-SHOT on a network it never saw -- the cross-topology generalisation
experiment.

  python train_topo_agnostic.py --train-topos abilene_sndlib,geant_sndlib \
      --holdout germany50_sndlib --seed 0 --timesteps 500000 --tag _topoagn

Saves to results/topoagn{tag}_seed{seed}/policy.zip
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from marl_routing.graph_routing_env import GraphSeqRoutingEnv
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_gnn_extractor import TopoAgnosticGNNExtractor
from marl_routing.topology import load as load_topology

# per-network demand scaling that puts each into a congesting regime
# (same values as train_all_monaco.sh so results stay comparable)
LOADS = {
    "abilene_sndlib": [8.0, 12.0, 16.0],
    "geant_sndlib": [3.0, 5.0, 7.0],
    "germany50_sndlib": [35.0, 50.0, 65.0],
}

ap = argparse.ArgumentParser()
ap.add_argument("--train-topos", default="abilene_sndlib,geant_sndlib")
ap.add_argument("--holdout", default="germany50_sndlib",
                help="topology never trained on; evaluated zero-shot")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--timesteps", type=int, default=500000)
ap.add_argument("--n-per-scale", type=int, default=20)
ap.add_argument("--k-paths", type=int, default=3)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--rounds", type=int, default=3, help="message-passing rounds")
ap.add_argument("--delay-penalty", type=float, default=0.5)
ap.add_argument("--tag", default="_topoagn")
A = ap.parse_args()


def pairs_of(topo_name):
    n = load_topology(topo_name).n_nodes
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def spec(topo_name, split, n_per_scale):
    p = pairs_of(topo_name)
    mats = real_matrices(topo_name, p, LOADS[topo_name],
                         n_per_scale=n_per_scale, split=split)
    return (topo_name, p, mats)


def evaluate(env, topo_idx, mats, model, label):
    """Analytical max-util: OSPF vs learned, on held-out matrices."""
    rows = []
    for m in mats:
        ospf = env.ospf_max_util(m, topo_idx)
        env.set_matrix(m, topo_idx)
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(a))
        rows.append((ospf, info["max_util"]))
    o = np.array([r[0] for r in rows]); g = np.array([r[1] for r in rows])
    win = float((g < o).mean() * 100)
    print(f"  [{label}] OSPF {o.mean():6.1f}%  ->  learned {g.mean():6.1f}%   "
          f"delta {o.mean()-g.mean():+6.1f}pt   beats OSPF on {win:.0f}% of matrices")
    return {"ospf_mean": float(o.mean()), "learned_mean": float(g.mean()),
            "delta_pt": float(o.mean() - g.mean()), "win_pct": win,
            "n_matrices": len(rows)}


def main():
    train_topos = [t for t in A.train_topos.split(",") if t]
    out = Path(f"results/topoagn{A.tag}_seed{A.seed}")
    out.mkdir(parents=True, exist_ok=True)

    train_specs = [spec(t, "train", A.n_per_scale) for t in train_topos]
    env = GraphSeqRoutingEnv(train_specs, k_paths=A.k_paths, seed=A.seed,
                             delay_penalty=A.delay_penalty)
    print(f"[train] topos={train_topos} holdout={A.holdout} seed={A.seed} "
          f"steps={A.timesteps} rounds={A.rounds}")

    policy_kwargs = {
        "features_extractor_class": TopoAgnosticGNNExtractor,
        "features_extractor_kwargs": {"k_paths": A.k_paths, "hidden_dim": A.hidden,
                                      "rounds": A.rounds},
    }
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=256,
                n_epochs=10, gamma=0.995, gae_lambda=0.95, ent_coef=0.01,
                seed=A.seed, device="cpu", policy_kwargs=policy_kwargs, verbose=0)
    model.learn(total_timesteps=A.timesteps, progress_bar=False)
    model.save(out / "policy")
    print(f"[saved] {out/'policy'}.zip")

    # ---- evaluation: seen topologies (unseen traffic) + zero-shot topology ----
    results = {"train_topos": train_topos, "holdout": A.holdout, "seed": A.seed,
               "seen": {}, "zero_shot": {}}
    print("\n[eval] SEEN topologies, held-out traffic (test split):")
    test_specs = [spec(t, "test", 5) for t in train_topos]
    eval_env = GraphSeqRoutingEnv(test_specs, k_paths=A.k_paths, seed=A.seed,
                                  delay_penalty=A.delay_penalty)
    for i, t in enumerate(train_topos):
        results["seen"][t] = evaluate(eval_env, i, test_specs[i][2], model, t)

    if A.holdout:
        print(f"\n[eval] ZERO-SHOT on {A.holdout} (never trained on):")
        hs = spec(A.holdout, "test", 5)
        zenv = GraphSeqRoutingEnv([hs], k_paths=A.k_paths, seed=A.seed,
                                  delay_penalty=A.delay_penalty)
        results["zero_shot"][A.holdout] = evaluate(zenv, 0, hs[2], model, A.holdout)

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out/'summary.json'}")


if __name__ == "__main__":
    main()
