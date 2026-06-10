#!/usr/bin/env python3
"""
Generalization experiment: train ONE GNN policy on a distribution of gravity-model
traffic matrices, then test on UNSEEN matrices. Compares, on held-out traffic:
  - GNN (single trained policy, no recomputation)
  - OSPF (min-hop, the deployed baseline)
  - per-matrix greedy/myopic (recomputes for every matrix — an oracle-ish reference)

This is the experiment that justifies a learned GNN over a heuristic: if the GNN
beats OSPF on traffic it never saw, it's a deployable adaptive policy (the
'dynamic traffic' research question), not just a per-instance optimizer.
"""
import json
from pathlib import Path

import numpy as np
import networkx as nx
from stable_baselines3 import PPO

from marl_routing.traffic import generate_matrix
from marl_routing.topology import load as load_topology
from marl_routing.sequential_routing_env import MultiTrafficSequentialEnv, CAND_FEATS
from marl_routing.gnn_extractor import SeqGNNExtractor

TOPO = "abilene"
LOAD_FACTOR = 3.0
N_TRAIN, N_TEST = 30, 20
TRAIN_SEEDS = list(range(0, N_TRAIN))
TEST_SEEDS = list(range(1000, 1000 + N_TEST))   # disjoint from train
K_PATHS = 3
TIMESTEPS = 120_000
RESULTS = Path("results/generalization")
RESULTS.mkdir(parents=True, exist_ok=True)


def matrix_to_rates(T, pairs):
    return np.array([T[s, d] for (s, d) in pairs], dtype=np.float64)


def eval_gnn_on(model, env, rates, episodes=4):
    best = float("inf")
    for ep in range(episodes):
        env.set_matrix(rates)
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=(ep == 0))
            obs, _, done, _, info = env.step(a)
        best = min(best, info["max_util"])
    return best


def main():
    topo = load_topology(TOPO)
    n = topo.n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    train_mats = [matrix_to_rates(generate_matrix(TOPO, LOAD_FACTOR, seed=s), pairs)
                  for s in TRAIN_SEEDS]
    test_mats = [matrix_to_rates(generate_matrix(TOPO, LOAD_FACTOR, seed=s), pairs)
                 for s in TEST_SEEDS]

    print("=" * 72)
    print(f"GENERALIZATION — train on {N_TRAIN} matrices, test on {N_TEST} unseen "
          f"(load_factor={LOAD_FACTOR})")
    print("=" * 72)

    env = MultiTrafficSequentialEnv(TOPO, pairs, train_mats, k_paths=K_PATHS, seed=0)

    policy_kwargs = {
        "features_extractor_class": SeqGNNExtractor,
        "features_extractor_kwargs": {
            "n_nodes": n, "n_arcs": env.n_arcs, "arc_list": env.arcs, "hidden_dim": 64,
        },
    }
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=256,
                n_epochs=10, gamma=0.995, gae_lambda=0.95, ent_coef=0.01,
                device="cpu", policy_kwargs=policy_kwargs, verbose=0)
    print(f"[train] {TIMESTEPS} timesteps over the train distribution (CPU)...")
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    model.save(RESULTS / "gnn_generalist")

    # ---- evaluate on held-out matrices ----
    print("\n[eval] on unseen test matrices...")
    rows = []
    for k, rates in enumerate(test_mats):
        ospf = env.ospf_max_util(rates)
        myopic = env.myopic_max_util(rates)
        gnn = eval_gnn_on(model, env, rates)
        rows.append({"test_idx": k, "ospf": round(ospf, 2), "gnn": round(gnn, 2),
                     "myopic": round(myopic, 2),
                     "gnn_vs_ospf": round(ospf - gnn, 2),
                     "gnn_beats_ospf": gnn < ospf})

    def col(name): return np.array([r[name] for r in rows], float)
    ospf, gnn, myo = col("ospf"), col("gnn"), col("myopic")
    win = ospf - gnn
    summary = {
        "load_factor": LOAD_FACTOR, "n_train": N_TRAIN, "n_test": N_TEST,
        "ospf_mean": round(ospf.mean(), 2), "ospf_std": round(ospf.std(), 2),
        "gnn_mean": round(gnn.mean(), 2), "gnn_std": round(gnn.std(), 2),
        "myopic_mean": round(myo.mean(), 2), "myopic_std": round(myo.std(), 2),
        "gnn_vs_ospf_mean_pts": round(win.mean(), 2),
        "gnn_vs_ospf_std_pts": round(win.std(), 2),
        "gnn_beats_ospf_frac": round(float((gnn < ospf).mean()), 3),
        "ospf_overloaded_frac": round(float((ospf > 100).mean()), 3),
        "gnn_overloaded_frac": round(float((gnn > 100).mean()), 3),
        "per_test": rows,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("GENERALIZATION RESULT (held-out, unseen traffic)")
    print("=" * 72)
    print(f"  OSPF      max-util: {summary['ospf_mean']:.1f}% ± {summary['ospf_std']:.1f}"
          f"   (overloaded on {summary['ospf_overloaded_frac']*100:.0f}% of matrices)")
    print(f"  GNN       max-util: {summary['gnn_mean']:.1f}% ± {summary['gnn_std']:.1f}"
          f"   (overloaded on {summary['gnn_overloaded_frac']*100:.0f}% of matrices)")
    print(f"  myopic    max-util: {summary['myopic_mean']:.1f}% ± {summary['myopic_std']:.1f}")
    print(f"  GNN beats OSPF by {summary['gnn_vs_ospf_mean_pts']:.1f} ± "
          f"{summary['gnn_vs_ospf_std_pts']:.1f} pts, "
          f"on {summary['gnn_beats_ospf_frac']*100:.0f}% of unseen matrices")
    print(f"\n  -> written to {RESULTS/'summary.json'}")


if __name__ == "__main__":
    main()
