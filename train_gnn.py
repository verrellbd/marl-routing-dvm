#!/usr/bin/env python3
"""
GNN-based PPO training for routing optimization.
Compares GNN backbone vs flat MLP on same task.
"""
from pathlib import Path
import json

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from marl_routing.routing_env import RoutingEnv
from marl_routing.gnn_env import GraphRoutingEnv
from marl_routing.gnn_policy import GNNActorCriticPolicy


def train_and_eval(env, policy_name: str, total_timesteps: int = 5000):
    """Train PPO agent and evaluate.

    Args:
        env: Gym environment
        policy_name: "gnn" or "mlp"
        total_timesteps: Training timesteps

    Returns:
        dict with metrics
    """
    print(f"\n{'='*70}")
    print(f"Training {policy_name.upper()} Policy")
    print(f"{'='*70}")

    results_dir = Path(__file__).parent / "results" / f"ppo_models_{policy_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(__file__).parent / "results" / f"ppo_logs_{policy_name}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5,
        save_path=str(results_dir),
        name_prefix=f"abilene_{policy_name}",
        save_replay_buffer=False,
    )

    # Create agent
    print(f"\n[1] Creating {policy_name.upper()} PPO agent...")
    if policy_name == "gnn":
        # Use GNN policy
        model = PPO(
            GNNActorCriticPolicy,
            env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=str(logs_dir),
            policy_kwargs={"n_nodes": 12, "gnn_hidden": 64},
        )
    else:
        # Use default MLP policy
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=str(logs_dir),
        )

    # Train
    print(f"\n[2] Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # Save
    final_model_path = results_dir / f"abilene_{policy_name}_final"
    model.save(final_model_path)
    print(f"\n✅ Model saved to {final_model_path}")

    # Evaluate
    print(f"\n[3] Evaluating trained {policy_name.upper()} policy...")
    obs, info = env.reset()
    total_reward = 0.0
    episode_rewards = []

    for step in range(60):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break

    final_util = info["max_util"]
    episode_rewards.append(total_reward)

    print(f"    Episode reward: {total_reward:.2f}")
    print(f"    Final max link util: {final_util:.2f}%")

    return {
        "policy": policy_name,
        "final_reward": float(total_reward),
        "final_util": float(final_util),
        "training_steps": total_timesteps,
    }


def main():
    print("=" * 70)
    print("GNN vs MLP Comparison: Network Routing Optimization")
    print("=" * 70)

    # Create base environment
    print("\n[Setup] Creating base environment...")
    base_env = RoutingEnv(
        topo_name="abilene",
        traffic_file=Path(__file__).parent / "results" / "traffic_abilene_α0.3_min30.json",
    )
    print(f"    {base_env.n_flows} flows, {base_env.topo.n_nodes} nodes")

    # Train MLP baseline
    print("\n" + "="*70)
    print("BASELINE: Flat MLP Policy")
    print("="*70)
    mlp_results = train_and_eval(base_env, "mlp", total_timesteps=5000)

    # Create graph environment for GNN
    print("\n[Setup] Creating graph environment for GNN...")
    gnn_env = GraphRoutingEnv(base_env)
    print(f"    Graph: {gnn_env.n_nodes} nodes, {gnn_env.num_edges} edges")

    # Train GNN
    print("\n" + "="*70)
    print("NOVEL: Graph Neural Network Policy")
    print("="*70)
    gnn_results = train_and_eval(gnn_env, "gnn", total_timesteps=5000)

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print(f"\nMLP Baseline:")
    print(f"  Final reward: {mlp_results['final_reward']:.2f}")
    print(f"  Max link util: {mlp_results['final_util']:.2f}%")

    print(f"\nGNN Policy:")
    print(f"  Final reward: {gnn_results['final_reward']:.2f}")
    print(f"  Max link util: {gnn_results['final_util']:.2f}%")

    improvement = (
        (mlp_results["final_util"] - gnn_results["final_util"])
        / mlp_results["final_util"]
        * 100
    )
    print(f"\nImprovement: {improvement:.2f}%")

    if improvement > 0:
        print(f"✅ GNN improves routing! ({improvement:.2f}% reduction in max util)")
    else:
        print(f"❌ GNN does not improve routing")

    # Save comparison
    comparison = {
        "mlp": mlp_results,
        "gnn": gnn_results,
        "improvement_pct": improvement,
    }
    comparison_file = Path(__file__).parent / "results" / "gnn_vs_mlp_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {comparison_file}")

    # Clean up
    base_env.close()
    gnn_env.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
