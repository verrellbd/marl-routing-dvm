#!/usr/bin/env python3
"""
GNN-based routing agent environment.

Agent directly controls per-flow routing by selecting paths from k-shortest paths.
Different from weight-based approach: agent makes direct routing decisions.

State: topology + link utilization + flow info
Action: select path for each flow
Reward: -max_link_utilization (same as OSPF baseline for fair comparison)
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from marl_routing.topology import load as load_topology


def compute_ksp(graph: nx.Graph, src: int, dst: int, k: int = 3) -> List[List[int]]:
    """Compute k-shortest paths using Yen's algorithm.

    Args:
        graph: NetworkX graph
        src: source node
        dst: destination node
        k: number of paths to compute

    Returns:
        list of paths, each path is list of node indices
    """
    if src == dst:
        return [[src]]

    try:
        # islice takes only the first k paths LAZILY from Yen's generator.
        # (Materializing the full generator enumerates every simple path in the
        # graph -> combinatorial blowup, ~0.8s/pair on dense topologies.)
        from itertools import islice
        paths = list(islice(nx.shortest_simple_paths(graph, src, dst, weight=None), k))
        return paths if paths else [[src, dst]]
    except nx.NetworkXNoPath:
        return [[src, dst]]


class GNNRoutingAgentEnv(gym.Env):
    """Gym environment for GNN-based routing agent.

    Agent directly selects paths for flows to minimize max link utilization.
    This is the correct comparison point against OSPF baseline.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topo_name: str = "abilene",
        traffic_file: Path | str = None,
        ns3_scenario: str = "scratch/abilene-gym/abilene-gym",
        work_dir: Path | str = None,
        sim_time_sec: float = 60.0,
        k_paths: int = 3,
    ):
        """Initialize routing agent environment.

        Args:
            topo_name: Topology name ("abilene" or "geant")
            traffic_file: Path to traffic JSON
            ns3_scenario: Path to ns-3 scenario
            work_dir: Working directory for IPC files
            sim_time_sec: Simulation time
            k_paths: Number of shortest paths per flow
        """
        self.topo_name = topo_name
        self.topo = load_topology(topo_name)
        self.sim_time_sec = sim_time_sec
        self.ns3_scenario = ns3_scenario
        self.k_paths = k_paths
        self.step_count = 0
        self.episode_count = 0

        # Working directory for IPC
        if work_dir is None:
            import os

            work_dir = Path(f"/tmp/ns3-gym-agent-{os.getpid()}")
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Load traffic
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

        # Precompute k-shortest paths for each flow
        print(f"[GNNRoutingAgentEnv] Computing {k_paths}-shortest paths for {self.n_flows} flows...")
        self.flow_paths = {}  # flow_id → list of paths
        for i, flow in enumerate(self.flows):
            src, dst = flow["src"], flow["dst"]
            paths = compute_ksp(self.topo.graph, src, dst, k=k_paths)
            self.flow_paths[i] = paths
            if i < 3:
                print(
                    f"  Flow {i} ({src}→{dst}): {len(paths)} paths computed "
                    f"(lengths: {[len(p) for p in paths]})"
                )

        # Action space: for each flow, select which path (0 to k-1)
        self.action_space = spaces.MultiDiscrete([k_paths] * self.n_flows)

        # Observation space: link utilization + adjacency matrix (for GNN extractor)
        n_directed_links = self.topo.n_directed_links
        n_nodes = self.topo.n_nodes
        obs_size = n_directed_links + (n_nodes * n_nodes)
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(obs_size,), dtype=np.float32
        )

        # IPC files
        self.action_file = self.work_dir / "action.json"
        self.state_file = self.work_dir / "state.json"

        # ns-3 paths
        # __file__ is marl_routing/gnn_routing_agent.py, so parent.parent = thesis/
        self.ns3_dev_path = Path(__file__).resolve().parent.parent / "ns-3-dev"
        self.topo_file = (
            Path(__file__).resolve().parent.parent / "topologies" / f"{topo_name}.json"
        )

        print(
            f"[GNNRoutingAgentEnv] Setup complete. "
            f"{self.n_flows} flows, {self.topo.n_nodes} nodes, {k_paths}-shortest paths"
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        self.episode_count += 1
        self.step_count = 0

        # Initial observation: zero utilization + adjacency matrix
        n_links = self.topo.n_directed_links
        adj_matrix = nx.to_numpy_array(self.topo.graph, dtype=np.float32)
        adj_flat = adj_matrix.flatten()
        link_utils = np.zeros(n_links, dtype=np.float32)
        initial_state = np.concatenate([link_utils, adj_flat]).astype(np.float32)

        info = {"episode": self.episode_count, "step": self.step_count, "max_util": 0.0}

        print(f"[GNNRoutingAgentEnv] Episode {self.episode_count} reset. {self.n_flows} flows")
        return initial_state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step.

        Args:
            action: Path selection for each flow (flow_id → path_idx in [0, k-1])

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # Convert action to routing decision (flow → selected path)
        routing = self._action_to_routing(action)

        # Write routing decision to action.json for ns-3
        self._write_routing_action(routing)

        # Run ns-3 simulation
        self._run_ns3_step()

        # Read results
        state, max_util, link_stats = self._read_state_file()

        # Reward: negative max link utilization (same as baseline)
        reward = float(-max_util)

        # Done after 60 steps
        terminated = self.step_count >= self.sim_time_sec

        info = {
            "step": self.step_count,
            "max_util": max_util,
            "link_stats": link_stats,
        }

        return state, reward, terminated, False, info

    def _action_to_routing(self, action: np.ndarray) -> Dict[int, int]:
        """Convert action to routing decision.

        Args:
            action: array of path indices for each flow

        Returns:
            dict: flow_id → selected_path_idx
        """
        routing = {}
        for flow_id in range(self.n_flows):
            path_idx = int(action[flow_id]) % len(self.flow_paths[flow_id])
            routing[flow_id] = path_idx
        return routing

    def _write_routing_action(self, routing: Dict[int, int]) -> None:
        """Convert routing decisions to link weights for ns-3.

        Strategy: Prefer links that are on selected paths by setting lower weights.
        - Selected path links: weight = 0.5 (preferred)
        - Non-selected path links: weight = 2.0 (avoid)
        - Default links: weight = 1.0 (neutral)

        This converts per-flow path selection into link weight preferences.
        """
        # Track which links are on selected paths
        preferred_links = set()  # (src, dst) tuples

        for flow_id, path_idx in routing.items():
            path = self.flow_paths[flow_id][path_idx]
            # Add all edges on this path to preferred set
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                preferred_links.add((src, dst))
                preferred_links.add((dst, src))  # Both directions

        # Get number of directed links (undirected links × 2)
        n_undirected = len(list(self.topo.graph.edges()))
        n_directed = n_undirected * 2

        # Initialize weights: 1.0 for all links
        link_weights = [1.0] * n_directed

        # Map edge indices to directed link indices
        edges = list(self.topo.graph.edges())
        for link_idx, (src, dst) in enumerate(edges):
            if (src, dst) in preferred_links:
                # Forward direction: prefer selected paths
                link_weights[link_idx] = 0.5
            if (dst, src) in preferred_links:
                # Reverse direction: prefer selected paths
                link_weights[link_idx + n_undirected] = 0.5

        # Write as link weights for ns-3
        action_data = {"link_weights": link_weights}

        with self.action_file.open("w") as f:
            json.dump(action_data, f, indent=2)

    def _run_ns3_step(self) -> None:
        """Run ns-3 simulation for one step."""
        cmd = [
            "bash",
            "-c",
            f"cd {self.ns3_dev_path} && timeout 30 ./ns3 run "
            f'"{self.ns3_scenario} '
            f'--topo={self.topo_file} '
            f'--traffic={self.traffic_file} '
            f'--action={self.action_file} '
            f'--state={self.state_file} '
            f'--simTime={self.sim_time_sec} '
            f'--stepTime=1" 2>&1',
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if self.step_count <= 2:
                print(f"[DEBUG ns3-step] Return code: {result.returncode}")
                if result.returncode != 0:
                    print(f"[DEBUG ns3-step] stderr: {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            print("[GNNRoutingAgentEnv] Warning: ns-3 timed out (120s)")

    def _read_state_file(self) -> Tuple[np.ndarray, float, Dict]:
        """Read state from ns-3 output and return graph-augmented observation."""
        if not self.state_file.exists():
            # Fallback: return zeros with adjacency
            if self.step_count <= 3:
                print(f"[WARNING] State file not found at {self.state_file}")
            obs_size = self.observation_space.shape[0]
            obs = np.zeros(obs_size, dtype=np.float32)
            return obs, 0.0, {"max_link_utilization_pct": 0.0}

        with self.state_file.open() as f:
            state_data = json.load(f)

        max_util = state_data.get("max_link_utilization_pct", 0.0)
        link_utils_undirected = state_data.get("link_utilizations", [])

        # ns-3 returns undirected link utilizations (15 for Abilene)
        # but we need directed (30). Duplicate each for both directions.
        link_utils_directed = []
        for util in link_utils_undirected:
            link_utils_directed.append(util)  # Forward
            link_utils_directed.append(util)  # Reverse
        link_utils = np.array(link_utils_directed, dtype=np.float32)

        # Build adjacency matrix for GNN
        adj_matrix = nx.to_numpy_array(self.topo.graph, dtype=np.float32)
        adj_flat = adj_matrix.flatten()

        # Concatenate: [link_utils, adj_flat]
        obs = np.concatenate([link_utils, adj_flat]).astype(np.float32)

        return obs, max_util, state_data

    def close(self) -> None:
        """Close environment."""
        pass

    def render(self) -> None:
        """Not implemented."""
        pass


if __name__ == "__main__":
    # Quick test
    print("Creating GNNRoutingAgentEnv...")
    env = GNNRoutingAgentEnv(topo_name="abilene")

    print("\nResetting...")
    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}, Info: {info}")

    print("\nRunning 2 steps...")
    for step in range(2):
        action = env.action_space.sample()
        print(f"  Action: {action[:5]}... (first 5 flows)")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Reward: {reward:.2f}, Max util: {info['max_util']:.2f}%")
        if terminated:
            break

    env.close()
    print("\n✅ Environment works!")
