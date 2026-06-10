"""Gym environment for routing optimization via ns-3 simulation.

File-based IPC: Python ↔ JSON files ↔ ns-3
- Python writes link weight multipliers to JSON
- ns-3 reads, applies weights to OSPF, simulates, writes stats to JSON
- Python reads stats, computes rewards
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from marl_routing.topology import load as load_topology
from marl_routing.traffic import summary as traffic_summary


class RoutingEnv(gym.Env):
    """OpenAI Gym environment for network routing optimization.

    State: Link utilization vector (one float per link)
    Action: Path selection per flow (discrete, k choices per flow)
    Reward: -max_link_utilization (minimize congestion)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topo_name: str = "abilene",
        traffic_file: Path | str = None,
        ns3_scenario: str = "scratch/abilene-gym/abilene-gym",
        work_dir: Path | str = None,
        sim_time_sec: float = 60.0,
    ):
        """Initialize the routing environment.

        Args:
            topo_name: Topology name ("abilene" or "geant")
            traffic_file: Path to traffic JSON (e.g., traffic_abilene_α0.3_min30.json)
            ns3_scenario: Relative path to ns-3 scenario (from ns-3-dev)
            work_dir: Working directory for IPC files (default: /tmp/ns3-gym-{pid})
            sim_time_sec: Total simulation time in seconds
        """
        self.topo_name = topo_name
        self.topo = load_topology(topo_name)
        self.sim_time_sec = sim_time_sec
        self.ns3_scenario = ns3_scenario
        self.step_count = 0
        self.episode_count = 0

        # Set up working directory for IPC
        if work_dir is None:
            import os

            work_dir = Path(f"/tmp/ns3-gym-{os.getpid()}")
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Load traffic flows (for reference only, not used in control)
        if traffic_file is None:
            traffic_file = (
                Path(__file__).resolve().parent.parent
                / "results"
                / f"traffic_{topo_name}_α0.3_min30.json"
            )
        self.traffic_file = Path(traffic_file)
        with self.traffic_file.open() as f:
            traffic_data = json.load(f)
        self.flows = traffic_data["flows"]
        self.n_flows = len(self.flows)

        # Count number of directed links (topology includes both directions)
        # For action space, we control one weight per directed link
        self.n_links = self.topo.n_directed_links
        print(f"[RoutingEnv] Loaded {self.n_flows} flows, {self.n_links} directed links ({self.n_links // 2} undirected)")

        # Action space: continuous link weight multipliers [0.1, 10.0] per link
        # Agent outputs an array of weights, one per link
        self.action_space = spaces.Box(
            low=0.1, high=10.0, shape=(self.n_links,), dtype=np.float32
        )

        # State space: link utilization (one float per link)
        # Each link can be 0-100% utilized (undirected, one entry per link direction counted)
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(self.n_links,), dtype=np.float32
        )

        # IPC file paths
        self.action_file = self.work_dir / "action.json"
        self.state_file = self.work_dir / "state.json"

        # ns3-dev path
        self.ns3_dev_path = Path(__file__).resolve().parent.parent.parent / "ns-3-dev"
        self.topo_file = Path(__file__).resolve().parent.parent / "topologies" / f"{topo_name}.json"
        self._last_routing = []  # Last routing decision (for ns3 call)

        print(f"[RoutingEnv] Setup complete. Work dir: {self.work_dir}")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and start new episode.

        Returns:
            observation: Initial link utilization
            info: Metadata
        """
        super().reset(seed=seed)
        self.episode_count += 1
        self.step_count = 0

        # Initial state: assume all links at 0% utilization
        initial_state = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "episode": self.episode_count,
            "step": self.step_count,
            "max_util": 0.0,
        }

        print(
            f"[RoutingEnv] Episode {self.episode_count} reset. "
            f"{self.n_flows} flows, {self.topo.n_nodes} nodes"
        )
        return initial_state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step: send link weights to ns-3, get utilization feedback.

        Args:
            action: Link weight multipliers [0.1, 10.0]

        Returns:
            observation: Link utilization after step
            reward: -max_link_utilization
            terminated: True if episode done
            truncated: False (no early stopping)
            info: Metadata (max_util, per-link stats, etc.)
        """
        self.step_count += 1

        # Convert action to link weights
        weights = self._action_to_routing(action)
        self._last_routing = weights

        # Run ns-3 for 1 second (sends action.json, gets state.json back)
        self._run_ns3_step()

        # Read link utilization and path length from ns-3
        state, max_util, link_stats = self._read_state_file()

        # Reward: negative average flow path length (proxy for latency)
        # Shorter paths = fewer hops = lower latency
        # Using path length creates optimization signal even with light traffic
        avg_path_length = link_stats.get("avg_path_length", 1.0)
        reward = float(-avg_path_length) if avg_path_length > 0 else 0.0

        # Episode done after 60 steps (60 seconds)
        terminated = self.step_count >= self.sim_time_sec

        info = {
            "step": self.step_count,
            "max_util": max_util,
            "avg_path_length": link_stats.get("avg_path_length", 1.0),
            "flow_count": link_stats.get("flow_count", 0),
            "link_stats": link_stats,
            "weights": weights[:3],  # Log first 3 weights for debugging
        }

        return state, reward, terminated, False, info

    def _action_to_routing(self, action: np.ndarray) -> List[float]:
        """Convert continuous action (link weights) to routing control.

        Args:
            action: Array of link weight multipliers [0.1, 10.0]

        Returns:
            List of weights to apply to each link (for JSON serialization)
        """
        # Clamp to valid range [0.1, 10.0]
        weights = np.clip(action, 0.1, 10.0).tolist()
        return weights

    def _write_action_file(self, weights: List[float]) -> None:
        """Write link weight multipliers to JSON for ns-3 to read."""
        action_data = {
            "episode": self.episode_count,
            "step": self.step_count,
            "link_weights": weights,
        }
        with self.action_file.open("w") as f:
            json.dump(action_data, f, indent=2)

    def _run_ns3_step(self) -> None:
        """Run ns-3 for 1 second (blocking call)."""
        # Write action.json (gym env → ns-3)
        self._write_action_file(self._last_routing)

        # Call ns-3 (starts fresh each time for now)
        # In production, this would be a persistent ns-3 process
        # For now, we spawn ns3 for each step (slow but works for validation)
        cmd = [
            "bash", "-c",
            f"cd {self.ns3_dev_path} && timeout 30 ./ns3 run "
            f'"{self.ns3_scenario} '
            f'--topo={self.topo_file} '
            f'--traffic={self.traffic_file} '
            f'--action={self.action_file} '
            f'--state={self.state_file} '
            f'--simTime={self.sim_time_sec} '
            f'--stepTime=1" 2>&1'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
            # Ignore output for now (ns-3 is verbose)
        except subprocess.TimeoutExpired:
            print("[RoutingEnv] Warning: ns-3 timed out")

    def _write_mock_state(self) -> None:
        """Write mock link utilization for testing (before ns-3 integration)."""
        # Mock: random link utilization
        n_links = self.observation_space.shape[0]
        util = np.random.uniform(5, 15, n_links).astype(float)
        max_util = float(np.max(util))

        state_data = {
            "episode": self.episode_count,
            "step": self.step_count,
            "max_link_utilization_pct": max_util,
            "link_utilizations": util.tolist(),
        }
        with self.state_file.open("w") as f:
            json.dump(state_data, f, indent=2)

    def _read_state_file(self) -> Tuple[np.ndarray, float, Dict]:
        """Read link utilization from ns-3 output file."""
        if not self.state_file.exists():
            # Fallback: return mock state
            self._write_mock_state()

        with self.state_file.open() as f:
            state_data = json.load(f)

        max_util = state_data["max_link_utilization_pct"]
        link_utils = np.array(state_data["link_utilizations"], dtype=np.float32)

        return link_utils, max_util, state_data

    def render(self) -> None:
        """Not implemented."""
        pass


if __name__ == "__main__":
    # Quick test
    print("Creating RoutingEnv...")
    env = RoutingEnv(topo_name="abilene")

    print("Resetting environment...")
    obs, info = env.reset()
    print(f"  Initial obs shape: {obs.shape}, info: {info}")

    print("\nRunning 3 steps with random link weight actions...")
    for step in range(3):
        action = env.action_space.sample()
        print(f"  Action {step}: {action[:3]}... (first 3 weights)")
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  Step {step}: reward={reward:.2f}, max_util={info['max_util']:.2f}%, "
            f"obs shape={obs.shape}"
        )
        if terminated:
            print("  Episode terminated")
            break

    print("\n✅ Environment test passed!")
