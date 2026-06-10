#!/usr/bin/env python3
"""Quick test on CPU with even lighter setup to validate code."""
from pathlib import Path
from stable_baselines3 import PPO
from marl_routing.gnn_routing_agent import GNNRoutingAgentEnv
from marl_routing.gnn_extractor import SimpleGNNExtractor

print("=" * 70)
print("CPU Test: Validate code without GPU overhead")
print("=" * 70)

# Use CPU for quick validation (no CUDA overhead, k-shortest paths computation is the bottleneck)
device = "cpu"

print(f"\n[1/4] Creating environment with α=1.0 traffic...")
traffic_file = Path.home() / "thesis" / "results" / "traffic_abilene_α1.0_min30.json"

try:
    env = GNNRoutingAgentEnv(
        topo_name="abilene",
        traffic_file=traffic_file,
        k_paths=3,
    )
    print(f"✓ Environment created: {env.n_flows} flows, {env.topo.n_nodes} nodes")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

print(f"\n[2/4] Creating PPO model on {device}...")
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
        verbose=0,
    )
    print(f"✓ Model created")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

print(f"\n[3/4] Training for 100 timesteps (ultra quick test)...")
try:
    model.learn(total_timesteps=100, progress_bar=False)
    print(f"✓ Training complete")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

print(f"\n[4/4] Quick evaluation...")
obs, _ = env.reset()
max_utils = []
for step in range(2):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, _, info = env.step(action)
    max_utils.append(info.get('max_util', 0.0))
    print(f"  Step {step+1}: max_util={info.get('max_util', 0.0):.2f}%")

print(f"\n✅ Code validated! Avg util: {sum(max_utils)/len(max_utils):.2f}%")
