#!/usr/bin/env python3
"""
Train GNN-PPO routing agents on the fast analytical environment, across a load
sweep, and compare against OSPF (min-hop), greedy coordinate-descent (achievable
reference), and random search.

This replaces ns-3-in-the-loop training (too slow + truncated rewards). ns-3 is
used afterwards only to validate the learned routing at high fidelity.
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
from stable_baselines3 import PPO

from marl_routing.topology import load as load_topology
from marl_routing.gnn_routing_agent import compute_ksp
from marl_routing.analytical_routing_env import AnalyticalRoutingEnv, route_max_util
from marl_routing.gnn_extractor import SimpleGNNExtractor

TRAFFIC = Path("results/traffic_abilene_α1.5_min30.json")
LOAD_SCALES = [1.0, 2.0, 3.0]
K_PATHS = 3
TIMESTEPS = 30_000
RESULTS = Path("results/analytical_gnn")
RESULTS.mkdir(parents=True, exist_ok=True)


def greedy_reference(env, iters=8):
    """Coordinate descent from OSPF start — an achievable lower bound on max-util."""
    cands = [env.flow_paths[i] for i in range(env.n_flows)]
    choice = [0] * env.n_flows

    def mu(ch):
        chosen = [cands[i][ch[i]] for i in range(env.n_flows)]
        _, m = route_max_util(env.topo, env.flows, env.arc_index, chosen)
        return m

    cur = mu(choice)
    for _ in range(iters):
        improved = False
        for i in range(env.n_flows):
            best_k, best_m = choice[i], cur
            for k in range(len(cands[i])):
                if k == choice[i]:
                    continue
                ch = choice.copy(); ch[i] = k
                m = mu(ch)
                if m < best_m - 1e-9:
                    best_m, best_k = m, k
            if best_k != choice[i]:
                choice[i], cur = best_k, best_m
                improved = True
        if not improved:
            break
    return cur


def random_reference(env, tries=2000):
    best = float("inf")
    env.reset()
    for _ in range(tries):
        _, _, _, _, i = env.step(env.action_space.sample())
        best = min(best, i["max_util"])
        if i["step"] >= env.episode_len:
            env.reset()
    return best


def eval_policy(model, env, episodes=20):
    """Best max-util the trained policy achieves (deterministic + a few stochastic)."""
    best = float("inf")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=(ep == 0))
            obs, _, done, _, info = env.step(action)
            best = min(best, info["max_util"])
    return best


def main():
    print("=" * 72)
    print("GNN-PPO on analytical environment — load sweep vs OSPF/greedy/random")
    print("=" * 72)
    summary = []

    for scale in LOAD_SCALES:
        print(f"\n{'#'*72}\n# load_scale = {scale}\n{'#'*72}")
        env = AnalyticalRoutingEnv(
            traffic_file=TRAFFIC, k_paths=K_PATHS, episode_len=20, load_scale=scale
        )
        ospf = env.ospf_max_util
        greedy = greedy_reference(env)
        rand = random_reference(env)

        policy_kwargs = {
            "features_extractor_class": SimpleGNNExtractor,
            "features_extractor_kwargs": {"n_nodes": env.topo.n_nodes, "hidden_dim": 64},
        }
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
            device="cpu", policy_kwargs=policy_kwargs, verbose=0,
        )
        print(f"[train] {TIMESTEPS} timesteps (analytical, CPU)...")
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

        model_path = RESULTS / f"gnn_scale{scale}"
        model.save(model_path)
        gnn = eval_policy(model, env)

        row = {
            "load_scale": scale, "n_flows": env.n_flows,
            "ospf_max_util": round(ospf, 2), "gnn_max_util": round(gnn, 2),
            "greedy_max_util": round(greedy, 2), "random_max_util": round(rand, 2),
            "gnn_vs_ospf_pts": round(ospf - gnn, 2),
            "gnn_beats_ospf": gnn < ospf,
            "model": str(model_path),
        }
        summary.append(row)
        print(f"\n  RESULT  OSPF={ospf:.2f}%  GNN={gnn:.2f}%  greedy={greedy:.2f}%  "
              f"random={rand:.2f}%   GNN beats OSPF by {ospf-gnn:.2f} pts")

    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 72)
    print(f"SWEEP COMPLETE — written to {out}")
    print("=" * 72)
    print(f"\n{'scale':>6}{'flows':>7}{'OSPF%':>9}{'GNN%':>9}{'greedy%':>9}"
          f"{'rand%':>9}{'GNN-win':>9}")
    for r in summary:
        print(f"{r['load_scale']:>6.1f}{r['n_flows']:>7}{r['ospf_max_util']:>9.2f}"
              f"{r['gnn_max_util']:>9.2f}{r['greedy_max_util']:>9.2f}"
              f"{r['random_max_util']:>9.2f}{r['gnn_vs_ospf_pts']:>9.2f}")


if __name__ == "__main__":
    main()
