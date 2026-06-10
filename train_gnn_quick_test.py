#!/usr/bin/env python3
"""Quick test training on α=1.0 with minimal timesteps."""
from pathlib import Path
from stable_baselines3 import PPO
from marl_routing.gnn_routing_agent import GNNRoutingAgentEnv
from marl_routing.gnn_extractor import SimpleGNNExtractor

# Use cuda:0 - GPU selection via command-line CUDA_VISIBLE_DEVICES
device = "cuda:0"

print("="*70)
print("Quick Training Test: α=1.0 on GPU 2")
print("="*70)

# Create environment for α=1.0
traffic_file = Path.home() / "thesis" / "results" / "traffic_abilene_α1.0_min30.json"
print(f"\n[1/5] Creating environment with α=1.0 traffic...")
print(f"      Traffic file: {traffic_file}")

try:
    env = GNNRoutingAgentEnv(
        topo_name="abilene",
        traffic_file=traffic_file,
        k_paths=3,
    )
    print(f"✓ Environment created: {env.n_flows} flows, {env.topo.n_nodes} nodes")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    exit(1)

# Create model
print(f"\n[2/5] Creating PPO model...")
try:
    policy_kwargs = {
        "features_extractor_class": SimpleGNNExtractor,
        "features_extractor_kwargs": {"n_nodes": env.topo.n_nodes, "hidden_dim": 64},
    }
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    print(f"✓ Model created")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    exit(1)

# Train for just 500 timesteps as a quick test
print(f"\n[3/5] Training for 500 timesteps (quick test)...")
try:
    model.learn(total_timesteps=500, progress_bar=True)
    print(f"✓ Training complete")
except Exception as e:
    print(f"✗ Training failed: {e}")
    exit(1)

# Save
print(f"\n[4/5] Saving model...")
model_path = Path.home() / "thesis" / "results" / "ppo_gnn_routing_agent_α1.0" / "gnn_routing_agent_final"
model_path.parent.mkdir(parents=True, exist_ok=True)
model.save(model_path)
print(f"✓ Model saved to {model_path}")

# Quick eval
print(f"\n[5/5] Quick evaluation...")
obs, _ = env.reset()
max_utils = []
for step in range(3):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, _, info = env.step(action)
    max_utils.append(info.get('max_util', 0.0))
    print(f"  Step {step+1}: max_util={info.get('max_util', 0.0):.2f}%")

print(f"\n✅ Success! Avg util: {sum(max_utils)/len(max_utils):.2f}%")
