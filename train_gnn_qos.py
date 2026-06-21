#!/usr/bin/env python3
"""
Train a QoS-aware GNN generalist: reward = -(marginal max-util) - delay_penalty*(extra hops).
The delay term makes the policy prefer SHORT paths when the network is uncongested
(matching OSPF delay) and detour only to relieve congestion -> aim for strict
dominance over OSPF (big loss/delay cuts under overload, no penalty when feasible).

Heavier run than the first generalist: more timesteps. Trains on a distribution of
gravity matrices (seeds 0-29), saves to results/generalization_qos/.
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from stable_baselines3 import PPO

from marl_routing.traffic import generate_matrix
from marl_routing.topology import load as load_topology
from marl_routing.sequential_routing_env import MultiTrafficSequentialEnv
from marl_routing.gnn_extractor import SeqGNNExtractor

K_PATHS = 3

_ap = argparse.ArgumentParser()
_ap.add_argument("--topo", default="abilene", help="abilene | geant")
_ap.add_argument("--loads", default="2,3,4", help="comma-list of training load factors")
_ap.add_argument("--seed", type=int, default=0, help="PPO/env seed (multi-seed runs)")
_ap.add_argument("--threads", type=int, default=8, help="torch CPU threads (parallel-friendly)")
_ap.add_argument("--timesteps", type=int, default=500_000)
_ap.add_argument("--delay-penalty", type=float, default=0.5,
                 help="penalty per extra hop over shortest. Lower on large/long-diameter "
                      "topologies (e.g. 0.1 on germany50) so useful detours aren't over-penalised.")
_ap.add_argument("--tag", default="", help="output dir suffix (e.g. _robust)")
_ARGS, _ = _ap.parse_known_args()
torch.set_num_threads(_ARGS.threads)
DELAY_PENALTY = _ARGS.delay_penalty

TOPO = _ARGS.topo
# Domain-randomize over load so the policy PRACTISES congestion relief across regimes.
TRAIN_LOADS = [float(x) for x in _ARGS.loads.split(",")]
TRAIN_SEEDS = list(range(0, 20))
RESULTS = Path(f"results/{TOPO}_qos{_ARGS.tag}_seed{_ARGS.seed}"); RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    topo = load_topology(TOPO); n = topo.n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    train_mats = [np.array([generate_matrix(TOPO, lf, seed=s)[a, b] for a, b in pairs])
                  for lf in TRAIN_LOADS for s in TRAIN_SEEDS]

    env = MultiTrafficSequentialEnv(TOPO, pairs, train_mats, k_paths=K_PATHS,
                                    seed=_ARGS.seed, delay_penalty=DELAY_PENALTY)
    print(f"[train] seed={_ARGS.seed} QoS reward (delay_penalty={DELAY_PENALTY}), "
          f"{_ARGS.timesteps} steps, {len(train_mats)} train matrices")

    policy_kwargs = {
        "features_extractor_class": SeqGNNExtractor,
        "features_extractor_kwargs": {
            "n_nodes": n, "n_arcs": env.n_arcs, "arc_list": env.arcs, "hidden_dim": 64},
    }
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=256,
                n_epochs=10, gamma=0.995, gae_lambda=0.95, ent_coef=0.01,
                seed=_ARGS.seed, device="cpu", policy_kwargs=policy_kwargs, verbose=0)
    model.learn(total_timesteps=_ARGS.timesteps, progress_bar=False)
    model.save(RESULTS / "gnn_generalist_qos")
    print(f"\n[saved] {RESULTS/'gnn_generalist_qos'}.zip")

    # quick analytical check on a few held-out matrices vs OSPF
    print("\n[analytical check on held-out seeds]")
    check_load = TRAIN_LOADS[-1]  # heaviest trained load (most congested held-out check)
    for seed in [1009, 1018, 1004, 1011]:
        T = generate_matrix(TOPO, check_load, seed=seed)
        rates = np.array([T[a, b] for a, b in pairs])
        ospf = env.ospf_max_util(rates)
        env.set_matrix(rates); obs, _ = env.reset(); done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True); obs, _, done, _, info = env.step(a)
        print(f"  seed {seed}: OSPF util {ospf:.1f}%  ->  GNN-QoS util {info['max_util']:.1f}%")


if __name__ == "__main__":
    main()
