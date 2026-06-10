#!/usr/bin/env python3
"""
Train GNN routing agents for heavier traffic loads (α=1.0, α=1.5).

Each load factor gets its own trained model.
"""
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from marl_routing.gnn_routing_agent import GNNRoutingAgentEnv
from marl_routing.gnn_extractor import SimpleGNNExtractor


def train_gnn_for_load_factor(
    alpha: float,
    min_flow_mbps: int = 30,
    total_timesteps: int = 5000,
):
    """Train GNN agent for specific load factor.

    Args:
        alpha: Load factor (0.3, 1.0, 1.5)
        min_flow_mbps: Minimum flow threshold
        total_timesteps: Training steps

    Returns:
        Path to saved model
    """
    print("\n" + "=" * 70)
    print(f"Training GNN Routing Agent (α={alpha})")
    print("=" * 70)

    device = "cpu"
    print(f"[STARTUP] Using CPU (GNN + PPO optimized for CPU)")

    # Load traffic for this alpha (use min50 for faster initialization)
    traffic_file = (
        Path(__file__).parent
        / "results"
        / f"traffic_abilene_α{alpha}_min50.json"
    )

    print(f"\n[SETUP] Creating environment for α={alpha}")
    print(f"  Traffic file: {traffic_file}")

    env = GNNRoutingAgentEnv(
        topo_name="abilene",
        traffic_file=traffic_file,
        k_paths=2,
    )
    n_nodes = env.topo.n_nodes
    n_flows = env.n_flows

    print(f"  Flows: {n_flows}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")

    # Setup results directory
    results_dir = Path(__file__).parent / "results" / f"ppo_gnn_routing_agent_α{alpha}"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=5,
        save_path=str(results_dir),
        name_prefix="gnn_routing_agent",
        save_replay_buffer=False,
    )

    # Create agent
    print(f"\n[AGENT] Creating PPO with GNN extractor (device={device})...")
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
        device=device,
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

    # Quick evaluation
    print(f"\n[EVAL] Quick 5-step evaluation on α={alpha}...")
    obs, info = env.reset()
    max_utils = []

    for step in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        max_utils.append(info.get("max_util", 0.0))
        print(f"  Step {step+1}: max_util={info.get('max_util', 0.0):.2f}%")

    avg_util = sum(max_utils) / len(max_utils)
    print(f"  Avg utilization: {avg_util:.2f}%")

    return final_model_path


def main():
    """Train models for all load factors."""
    print("\n" + "=" * 70)
    print("TRAINING GNN AGENTS FOR HEAVIER TRAFFIC")
    print("=" * 70)

    # Check GPU availability
    import torch

    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA not available, using CPU")

    # Train for each load factor
    load_factors = [1.0, 1.5]
    models = {}

    for alpha in load_factors:
        try:
            print(f"\n{'#' * 70}")
            print(f"# Training for α={alpha}")
            print(f"{'#' * 70}")

            model_path = train_gnn_for_load_factor(
                alpha=alpha,
                min_flow_mbps=30,
                total_timesteps=5000,
            )
            models[alpha] = model_path
        except Exception as e:
            print(f"❌ Error training α={alpha}: {e}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    for alpha, path in models.items():
        print(f"  α={alpha}: {path}")


if __name__ == "__main__":
    main()
