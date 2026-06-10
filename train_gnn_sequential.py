#!/usr/bin/env python3
"""
Train GNN-PPO on the SequentialRoutingEnv (per-flow placement, Discrete(k) action,
potential-based reward) across a load sweep. Compare vs OSPF, myopic-sequential
heuristic, greedy coordinate-descent, and random.

This is the credit-assignment-friendly reformulation: even a myopic policy here
beats the old one-shot GNN, so a trained GNN should dominate OSPF consistently.
"""
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from marl_routing.sequential_routing_env import SequentialRoutingEnv, CAND_FEATS
from marl_routing.gnn_extractor import SeqGNNExtractor

TRAFFIC = Path("results/traffic_abilene_α1.5_min30.json")
LOAD_SCALES = [1.0, 2.0, 3.0]
K_PATHS = 3
TIMESTEPS = 60_000
RESULTS = Path("results/sequential_gnn")
RESULTS.mkdir(parents=True, exist_ok=True)


def myopic_max_util(env):
    obs, _ = env.reset()
    done = False
    base = env.n_arcs + env.topo.n_nodes ** 2
    while not done:
        feats = obs[base: base + env.k_paths * CAND_FEATS]
        obs, _, done, _, info = env.step(int(np.argmin(feats[0::CAND_FEATS])))
    return info["max_util"]


def random_max_util(env, tries=500):
    best = float("inf")
    for _ in range(tries):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, done, _, info = env.step(env.action_space.sample())
        best = min(best, info["max_util"])
    return best


def eval_gnn(model, env, episodes=10):
    best = float("inf")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=(ep == 0))
            obs, _, done, _, info = env.step(a)
        best = min(best, info["max_util"])
    return best


def main():
    print("=" * 72)
    print("GNN-PPO SEQUENTIAL — load sweep vs OSPF / myopic / random")
    print("=" * 72)
    summary = []

    for scale in LOAD_SCALES:
        print(f"\n{'#'*72}\n# load_scale = {scale}\n{'#'*72}")
        env = SequentialRoutingEnv(
            traffic_file=TRAFFIC, k_paths=K_PATHS, load_scale=scale
        )
        ospf = env.ospf_max_util
        myopic = myopic_max_util(env)
        rand = random_max_util(env)

        policy_kwargs = {
            "features_extractor_class": SeqGNNExtractor,
            "features_extractor_kwargs": {
                "n_nodes": env.topo.n_nodes, "n_arcs": env.n_arcs,
                "arc_list": env.arcs, "hidden_dim": 64,
            },
        }
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, ent_coef=0.01,
            device="cpu", policy_kwargs=policy_kwargs, verbose=0,
        )
        print(f"[train] {TIMESTEPS} timesteps (sequential, Discrete({K_PATHS}), CPU)...")
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
        model.save(RESULTS / f"gnn_scale{scale}")
        gnn = eval_gnn(model, env)

        row = {
            "load_scale": scale, "n_flows": env.n_flows,
            "ospf_max_util": round(ospf, 2), "gnn_max_util": round(gnn, 2),
            "myopic_max_util": round(myopic, 2), "random_max_util": round(rand, 2),
            "gnn_vs_ospf_pts": round(ospf - gnn, 2),
            "gnn_beats_ospf": gnn < ospf,
            "gnn_beats_myopic": gnn < myopic,
        }
        summary.append(row)
        print(f"\n  RESULT  OSPF={ospf:.2f}%  GNN={gnn:.2f}%  myopic={myopic:.2f}%  "
              f"random={rand:.2f}%   GNN beats OSPF by {ospf-gnn:.2f} pts")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 72 + f"\nSEQUENTIAL SWEEP COMPLETE — {RESULTS/'summary.json'}\n" + "=" * 72)
    print(f"\n{'scale':>6}{'flows':>7}{'OSPF%':>9}{'GNN%':>9}{'myopic%':>9}"
          f"{'rand%':>9}{'GNN-win':>9}")
    for r in summary:
        print(f"{r['load_scale']:>6.1f}{r['n_flows']:>7}{r['ospf_max_util']:>9.2f}"
              f"{r['gnn_max_util']:>9.2f}{r['myopic_max_util']:>9.2f}"
              f"{r['random_max_util']:>9.2f}{r['gnn_vs_ospf_pts']:>9.2f}")


if __name__ == "__main__":
    main()
