#!/usr/bin/env python3
"""
Phase 3: Evaluate GNN routing agent on heavier traffic (α=1.0, α=1.5).

Tests if GNN can beat OSPF under congestion.
"""
import json
import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
from stable_baselines3 import PPO

from marl_routing.gnn_routing_agent import GNNRoutingAgentEnv


def evaluate_gnn_on_traffic(
    traffic_file: Union[Path, str],
    traffic_label: str,
    model_path: Union[Path, str],
    n_steps: int = 10,
    device: str = "cuda:3",
) -> Dict:
    """Evaluate trained GNN model on specific traffic scenario.

    Args:
        traffic_file: Path to traffic JSON
        traffic_label: Label for results (e.g., "α=1.0")
        model_path: Path to trained PPO model
        n_steps: Number of evaluation steps
        device: Device to run on (cuda:X or cpu)

    Returns:
        dict with evaluation metrics
    """
    print("\n" + "=" * 70)
    print(f"GNN Evaluation on {traffic_label}")
    print("=" * 70)

    # Create environment with specified traffic file
    print(f"\n[ENV] Creating GNNRoutingAgentEnv with {traffic_file}...")
    env = GNNRoutingAgentEnv(
        topo_name="abilene",
        traffic_file=traffic_file,
        k_paths=3,
    )
    n_flows = env.n_flows
    print(f"  Topology: {env.topo.n_nodes} nodes, {env.topo.n_directed_links} directed links")
    print(f"  Flows: {n_flows}")

    # Load trained model
    print(f"\n[MODEL] Loading trained PPO model from {model_path}...")
    model = PPO.load(model_path, env=env, device=device)
    print(f"  Device: {device}")

    # Evaluate
    print(f"\n[EVAL] Testing GNN agent ({n_steps}-step evaluation)...")
    obs, info = env.reset()

    max_utils = []
    rewards = []

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        max_util = info.get("max_util", 0.0)
        max_utils.append(max_util)
        rewards.append(reward)
        print(f"  Step {step+1:2d}: max_util={max_util:.2f}%, reward={reward:.2f}")

    avg_util = np.mean(max_utils)
    final_util = max_utils[-1]
    total_reward = np.sum(rewards)

    print(f"\n[RESULTS] {traffic_label}")
    print(f"  Final max util:   {final_util:.2f}%")
    print(f"  Avg max util:     {avg_util:.2f}%")
    print(f"  Total reward:     {total_reward:.2f}")

    return {
        "traffic_label": traffic_label,
        "traffic_file": str(traffic_file),
        "n_flows": n_flows,
        "n_steps": n_steps,
        "device": device,
        "final_util": final_util,
        "avg_util": avg_util,
        "total_reward": total_reward,
        "max_utils_history": max_utils,
        "rewards_history": rewards,
    }


def evaluate_ospf_baseline(
    traffic_file: Union[Path, str],
    traffic_label: str,
    n_steps: int = 10,
) -> Dict:
    """Evaluate OSPF baseline on specific traffic scenario.

    Uses uniform link weights (default OSPF behavior).

    Args:
        traffic_file: Path to traffic JSON
        traffic_label: Label for results
        n_steps: Number of evaluation steps

    Returns:
        dict with baseline metrics
    """
    print("\n" + "=" * 70)
    print(f"OSPF Baseline on {traffic_label}")
    print("=" * 70)

    # Create environment with specified traffic file
    print(f"\n[ENV] Creating GNNRoutingAgentEnv with {traffic_file}...")
    env = GNNRoutingAgentEnv(
        topo_name="abilene",
        traffic_file=traffic_file,
        k_paths=3,
    )
    n_flows = env.n_flows
    print(f"  Topology: {env.topo.n_nodes} nodes, {env.topo.n_directed_links} directed links")
    print(f"  Flows: {n_flows}")

    # Evaluate: always select path 0 (shortest path, OSPF behavior)
    print(f"\n[EVAL] Testing OSPF baseline ({n_steps}-step evaluation)...")
    obs, info = env.reset()

    max_utils = []
    rewards = []

    for step in range(n_steps):
        # OSPF: always select shortest path (path 0)
        action = np.zeros(n_flows, dtype=int)
        obs, reward, terminated, truncated, info = env.step(action)
        max_util = info.get("max_util", 0.0)
        max_utils.append(max_util)
        rewards.append(reward)
        print(f"  Step {step+1:2d}: max_util={max_util:.2f}%, reward={reward:.2f}")

    avg_util = np.mean(max_utils)
    final_util = max_utils[-1]
    total_reward = np.sum(rewards)

    print(f"\n[RESULTS] OSPF Baseline {traffic_label}")
    print(f"  Final max util:   {final_util:.2f}%")
    print(f"  Avg max util:     {avg_util:.2f}%")
    print(f"  Total reward:     {total_reward:.2f}")

    return {
        "traffic_label": traffic_label,
        "traffic_file": str(traffic_file),
        "n_flows": n_flows,
        "n_steps": n_steps,
        "baseline": "ospf",
        "final_util": final_util,
        "avg_util": avg_util,
        "total_reward": total_reward,
        "max_utils_history": max_utils,
        "rewards_history": rewards,
    }


def main():
    """Run Phase 3 evaluation."""
    print("\n" + "=" * 70)
    print("PHASE 3: GNN ROUTING EVALUATION ON HEAVIER TRAFFIC")
    print("=" * 70)

    # GPU setup - use cuda:0 (most compatible)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0"

    # Traffic scenarios to test
    scenarios = [
        ("α=1.0 (moderate)", "traffic_abilene_α1.0_min30.json"),
        ("α=1.5 (heavy)", "traffic_abilene_α1.5_min30.json"),
    ]

    # Paths
    thesis_dir = Path(__file__).parent
    results_dir = thesis_dir / "results"
    model_path = results_dir / "ppo_gnn_routing_agent" / "gnn_routing_agent_final"

    # Store all results
    all_results = {
        "phase": 3,
        "model": str(model_path),
        "device": device,
        "scenarios": {},
    }

    # Evaluate on each scenario
    for label, traffic_filename in scenarios:
        traffic_file = results_dir / traffic_filename

        print(f"\n{'#' * 70}")
        print(f"# Scenario: {label}")
        print(f"{'#' * 70}")

        # OSPF baseline
        ospf_result = evaluate_ospf_baseline(
            traffic_file=traffic_file,
            traffic_label=label,
            n_steps=10,
        )

        # GNN agent
        gnn_result = evaluate_gnn_on_traffic(
            traffic_file=traffic_file,
            traffic_label=label,
            model_path=model_path,
            n_steps=10,
            device=device,
        )

        # Compare
        ospf_util = ospf_result["avg_util"]
        gnn_util = gnn_result["avg_util"]
        delta = gnn_util - ospf_util
        pct_delta = (delta / ospf_util) * 100 if ospf_util > 0 else 0

        print(f"\n[COMPARISON] {label}")
        print(f"  OSPF avg util:  {ospf_util:.2f}%")
        print(f"  GNN avg util:   {gnn_util:.2f}%")
        print(f"  Delta:          {delta:+.2f}pp ({pct_delta:+.1f}%)")
        if gnn_util < ospf_util:
            print(f"  ✅ GNN WINS by {-delta:.2f}pp")
        else:
            print(f"  ❌ OSPF wins by {delta:.2f}pp")

        # Store results
        all_results["scenarios"][label] = {
            "ospf": ospf_result,
            "gnn": gnn_result,
            "comparison": {
                "ospf_avg_util": ospf_util,
                "gnn_avg_util": gnn_util,
                "delta_pp": delta,
                "delta_pct": pct_delta,
                "winner": "GNN" if gnn_util < ospf_util else "OSPF",
            },
        }

    # Save comprehensive results
    results_file = results_dir / "phase3_evaluation_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ All results saved to {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)
    for label in all_results["scenarios"]:
        comp = all_results["scenarios"][label]["comparison"]
        print(f"\n{label}:")
        print(f"  OSPF: {comp['ospf_avg_util']:.2f}%")
        print(f"  GNN:  {comp['gnn_avg_util']:.2f}%")
        print(f"  Winner: {comp['winner']} ({comp['delta_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
