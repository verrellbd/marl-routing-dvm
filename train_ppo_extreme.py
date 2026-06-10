#!/usr/bin/env python3
"""Train PPO agent on extreme traffic load (α=1.5).

At this load, network is heavily congested. Agent should have maximum
opportunity to optimize routing and beat OSPF.

Usage:
    CUDA_VISIBLE_DEVICES=2 python train_ppo_extreme.py
"""
import os

# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "2")

import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from marl_routing.routing_env import RoutingEnv

# Paths
THESIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THESIS_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PPO Training for Network Routing - EXTREME LOAD (α=1.5)")
print("=" * 80)

# Environment setup
print("\n[1/5] Creating gym environment (α=1.5 heavy load)...")
env = RoutingEnv(
    topo_name="abilene",
    traffic_file=THESIS_DIR / "results" / "traffic_abilene_α1.5_min30.json",
    ksp_file=THESIS_DIR / "results" / "ksp_abilene_k3.json",
    sim_time_sec=60.0,
)
print(f"  ✓ Env created: {env.n_flows} flows, {env.topo.n_nodes} nodes, k={env.k} paths")
print(f"  ✓ Traffic load: α=1.5 (heavy, ~10.9% OSPF baseline expected)")
print(f"  ✓ Action space: {env.action_space}")
print(f"  ✓ Observation space: {env.observation_space}")

# Model setup
print("\n[2/5] Creating PPO model...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
    device="cuda",
    tensorboard_log=RESULTS_DIR / "tb_logs",
)
print("  ✓ PPO model created")
print(f"  ✓ TensorBoard logs: {RESULTS_DIR / 'tb_logs'}")

# Training setup
print("\n[3/5] Setting up training callbacks...")
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=MODELS_DIR,
    name_prefix="ppo_routing_extreme",
    save_replay_buffer=False,
)
print(f"  ✓ Checkpoints: {MODELS_DIR}")

# Training
print("\n[4/5] Training for 50,000 timesteps (~833 episodes, extreme load)...")
print("      Expected: Heavy congestion forces agent to optimize hard...")
print()

try:
    model.learn(
        total_timesteps=50000,
        callback=checkpoint_callback,
        progress_bar=False,
        log_interval=100,
    )
    print("\n✅ Training completed successfully!")
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    try:
        partial_model_path = MODELS_DIR / "ppo_routing_extreme_partial"
        model.save(partial_model_path)
        print(f"\n⚠️ Partial model saved to: {partial_model_path}")
    except:
        pass
    raise

# Save final model
print("\n[5/5] Saving final model...")
final_model_path = MODELS_DIR / "ppo_routing_extreme_final"
model.save(final_model_path)
print(f"  ✓ Final model saved to: {final_model_path}")

# Quick evaluation
print("\n[Bonus] Quick evaluation (5 episodes on extreme load)...")
env.close()
eval_env = RoutingEnv(
    topo_name="abilene",
    traffic_file=THESIS_DIR / "results" / "traffic_abilene_α1.5_min30.json",
    ksp_file=THESIS_DIR / "results" / "ksp_abilene_k3.json",
)

episode_rewards = []
episode_utils = []
for ep in range(5):
    obs, _ = eval_env.reset()
    ep_reward = 0
    max_util = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        ep_reward += reward
        max_util = max(max_util, info['max_util'])
        if terminated or truncated:
            episode_rewards.append(ep_reward)
            episode_utils.append(max_util)
            print(f"  Episode {ep+1}: reward={ep_reward:.2f}, max_util={max_util:.2f}%")
            break

mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
mean_util = np.mean(episode_utils)
std_util = np.std(episode_utils)
print(f"\n  Mean episode reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"  Mean max link util: {mean_util:.2f}% ± {std_util:.2f}%")
print(f"  (Target: Beat OSPF baseline of ~10.9%)")

eval_env.close()

print("\n" + "=" * 80)
print("✅ Training Complete!")
print("=" * 80)
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"  - Light load model (α=0.3): {MODELS_DIR}/ppo_routing_final.zip")
print(f"  - Medium load model (α=1.0): {MODELS_DIR}/ppo_routing_heavy_final.zip")
print(f"  - Heavy load model (α=1.5): {final_model_path}.zip")
print(f"  - Check mean max_util above vs OSPF baseline")
print()
