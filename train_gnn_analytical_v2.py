#!/usr/bin/env python3
"""
v2: GNN-PPO on the analytical env with SHAPED reward (p-norm of utilizations) and
a larger timestep budget, to close the gap to greedy. Evaluation is still by true
max link utilization, so GNN vs OSPF/greedy/random stays apples-to-apples.
"""
import json
from pathlib import Path

from stable_baselines3 import PPO

from marl_routing.analytical_routing_env import AnalyticalRoutingEnv
from marl_routing.gnn_extractor import SimpleGNNExtractor
from train_gnn_analytical import greedy_reference, random_reference, eval_policy

TRAFFIC = Path("results/traffic_abilene_α1.5_min30.json")
LOAD_SCALES = [1.0, 2.0, 3.0]
K_PATHS = 3
TIMESTEPS = 120_000
RESULTS = Path("results/analytical_gnn_v2")
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 72)
    print("GNN-PPO v2 (shaped reward, 120k steps) — load sweep vs OSPF/greedy/random")
    print("=" * 72)
    summary = []

    for scale in LOAD_SCALES:
        print(f"\n{'#'*72}\n# load_scale = {scale}\n{'#'*72}")
        # Training env: shaped (dense) reward for better credit assignment
        train_env = AnalyticalRoutingEnv(
            traffic_file=TRAFFIC, k_paths=K_PATHS, episode_len=20,
            load_scale=scale, reward_mode="shaped", pnorm=6.0,
        )
        # Eval/reference env: true max-util objective
        eval_env = AnalyticalRoutingEnv(
            traffic_file=TRAFFIC, k_paths=K_PATHS, episode_len=20,
            load_scale=scale, reward_mode="neg_max",
        )
        ospf = eval_env.ospf_max_util
        greedy = greedy_reference(eval_env)
        rand = random_reference(eval_env)

        policy_kwargs = {
            "features_extractor_class": SimpleGNNExtractor,
            "features_extractor_kwargs": {"n_nodes": train_env.topo.n_nodes, "hidden_dim": 64},
        }
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.02,
            device="cpu", policy_kwargs=policy_kwargs, verbose=0,
        )
        print(f"[train] {TIMESTEPS} timesteps, shaped reward (p=6), CPU...")
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

        model_path = RESULTS / f"gnn_scale{scale}"
        model.save(model_path)
        gnn = eval_policy(model, eval_env, episodes=40)

        row = {
            "load_scale": scale, "n_flows": eval_env.n_flows,
            "ospf_max_util": round(ospf, 2), "gnn_max_util": round(gnn, 2),
            "greedy_max_util": round(greedy, 2), "random_max_util": round(rand, 2),
            "gnn_vs_ospf_pts": round(ospf - gnn, 2),
            "gnn_beats_ospf": gnn < ospf,
            "closed_gap_to_greedy_pct": round(
                100 * (ospf - gnn) / (ospf - greedy), 1) if ospf > greedy else None,
            "model": str(model_path),
        }
        summary.append(row)
        print(f"\n  RESULT  OSPF={ospf:.2f}%  GNN={gnn:.2f}%  greedy={greedy:.2f}%  "
              f"random={rand:.2f}%   GNN beats OSPF by {ospf-gnn:.2f} pts "
              f"(closed {row['closed_gap_to_greedy_pct']}% of OSPF→greedy gap)")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 72)
    print(f"v2 SWEEP COMPLETE — {RESULTS/'summary.json'}")
    print("=" * 72)
    print(f"\n{'scale':>6}{'flows':>7}{'OSPF%':>9}{'GNN%':>9}{'greedy%':>9}"
          f"{'rand%':>9}{'GNN-win':>9}{'gap-closed':>11}")
    for r in summary:
        gc = r["closed_gap_to_greedy_pct"]
        print(f"{r['load_scale']:>6.1f}{r['n_flows']:>7}{r['ospf_max_util']:>9.2f}"
              f"{r['gnn_max_util']:>9.2f}{r['greedy_max_util']:>9.2f}"
              f"{r['random_max_util']:>9.2f}{r['gnn_vs_ospf_pts']:>9.2f}"
              f"{(str(gc)+'%' if gc is not None else 'NA'):>11}")


if __name__ == "__main__":
    main()
