#!/usr/bin/env python3
"""
Train GNN routing agent to optimize path selection.

GNN agent learns to select k-shortest paths for each flow to minimize max link utilization.
This is a direct comparison against OSPF baseline (14.60% max util).

The agent makes path selection decisions which are converted to link weight preferences.
"""
import os
from pathlib import Path
import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from marl_routing.gnn_routing_agent import GNNRoutingAgentEnv
from marl_routing.gnn_extractor import SimpleGNNExtractor


def train_gnn_routing_agent(total_timesteps: int = 5000):
    """Train GNN agent to select paths.

    Args:
        total_timesteps: Number of training steps

    Returns:
        dict with training results
    """
    print("=" * 70)
    print("GNN Routing Agent Training")
    print("=" * 70)
    print("Agent learns to select k-shortest paths for flows")
    print("Goal: Beat OSPF baseline (14.60% max utilization)")

    # Create environment
    print("\n[SETUP] Creating GNNRoutingAgentEnv...")
    env = GNNRoutingAgentEnv(topo_name="abilene", k_paths=3)
    n_nodes = env.topo.n_nodes
    n_flows = env.n_flows

    print(f"  Topology: {n_nodes} nodes")
    print(f"  Flows: {n_flows}")
    print(f"  Action space: {env.action_space} (path selection per flow)")
    print(f"  Observation space: {env.observation_space.shape}")

    # Setup results directory
    results_dir = Path(__file__).parent / "results" / "ppo_gnn_routing_agent"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(__file__).parent / "results" / "ppo_logs_gnn_routing_agent"
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=5,
        save_path=str(results_dir),
        name_prefix="gnn_routing_agent",
        save_replay_buffer=False,
    )

    # Create agent
    print("\n[AGENT] Creating PPO with GNN extractor...")
    policy_kwargs = {
        "features_extractor_class": SimpleGNNExtractor,
        "features_extractor_kwargs": {"n_nodes": n_nodes, "hidden_dim": 64},
    }

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log=str(logs_dir),
        policy_kwargs=policy_kwargs,
    )

    # Train
    print(f"\n[TRAINING] Learning for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_model_path = results_dir / "gnn_routing_agent_final"
    model.save(final_model_path)
    print(f"✅ Model saved to {final_model_path}")

    # Evaluate
    print(f"\n[EVALUATION] Testing trained agent...")
    obs, info = env.reset()
    total_reward = 0.0
    max_utils = []

    for step in range(60):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        max_util = info["max_util"]
        max_utils.append(max_util)

        if (step + 1) % 10 == 0 or step < 3:
            print(f"  Step {step+1}: max_util={max_util:.2f}%, reward={reward:.2f}, terminated={terminated}")

        if terminated:
            print(f"  [TERMINATED at step {step+1}]")
            break

    final_util = max_utils[-1] if max_utils else 0.0
    avg_util = sum(max_utils) / len(max_utils) if max_utils else 0.0

    print(f"    Total steps: {len(max_utils)}")
    print(f"    Episode reward: {total_reward:.2f}")
    print(f"    Final max util: {final_util:.2f}%")
    print(f"    Avg max util: {avg_util:.2f}%")

    env.close()

    return {
        "agent": "GNN Routing Agent",
        "final_util": float(final_util),
        "avg_util": float(avg_util),
        "episode_reward": float(total_reward),
    }


def compare_with_baseline():
    """Compare GNN agent results with OSPF baseline."""
    print("\n" + "=" * 70)
    print("COMPARISON: GNN ROUTING vs OSPF BASELINE")
    print("=" * 70)

    # Load baseline
    baseline_file = Path(__file__).parent / "results" / "ospf_baseline.json"
    if not baseline_file.exists():
        print(f"❌ Baseline not found at {baseline_file}")
        print("   Run measure_ospf_baseline.py first")
        return

    with open(baseline_file) as f:
        baseline = json.load(f)

    ospf_util = baseline["aggregate"]["final_util_mean"]
    print(f"\nOSPF Baseline: {ospf_util:.2f}% max utilization")

    # Load GNN agent results (from tensorboard or saved metrics)
    results_file = Path(__file__).parent / "results" / "gnn_routing_results.json"
    if not results_file.exists():
        print(f"⚠️  Results file not yet saved: {results_file}")
        print("   Results will be saved after training")
        return

    with open(results_file) as f:
        gnn_results = json.load(f)

    gnn_util = gnn_results.get("avg_util", 0.0)
    print(f"GNN Agent:     {gnn_util:.2f}% max utilization")

    improvement = (ospf_util - gnn_util) / ospf_util * 100

    print(f"\nImprovement: {improvement:.2f}%")

    if improvement > 1.0:
        print(f"✅ GNN beats OSPF by {improvement:.2f}%!")
        print("   Graph-based path selection is more effective than default OSPF")
    elif improvement > -1.0:
        print(f"≈️  GNN matches OSPF (within ±1%)")
        print("   Path selection equivalent to OSPF shortest paths")
    else:
        print(f"❌ OSPF beats GNN by {-improvement:.2f}%")
        print("   OSPF routing is superior to learned GNN routing")


def main():
    print("[STARTUP] Using CUDA_VISIBLE_DEVICES=2\n")

    # Phase 1: Train GNN routing agent
    gnn_results = train_gnn_routing_agent(total_timesteps=5000)

    # Save results
    results_file = Path(__file__).parent / "results" / "gnn_routing_results.json"
    with open(results_file, "w") as f:
        json.dump(gnn_results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")

    # Phase 2: Compare with baseline
    compare_with_baseline()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. If GNN > 1% improvement: Success! GNN routing works")
    print("2. If GNN ≈ OSPF: Path selection equivalent, OSPF already optimal")
    print("3. If GNN < OSPF: OSPF superior, RL not effective for routing")
    print("=" * 70)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()
