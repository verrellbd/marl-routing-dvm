"""Simplified graph environment compatible with SB3.

Returns observations as numpy arrays that include both link utilization
and graph adjacency structure for the GNN to process.
"""
from __future__ import annotations

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from marl_routing.routing_env import RoutingEnv


class SimpleGraphRoutingEnv(gym.Env):
    """Wrapper that includes graph structure in numpy observation.

    Observation: concatenates [link_utilization, adjacency_matrix_flattened]
    This is SB3-compatible while preserving graph structure for GNN.
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env: RoutingEnv = None, topo_name: str = "abilene"):
        """Initialize environment.

        Args:
            base_env: RoutingEnv to wrap (if None, creates one)
            topo_name: Topology name if creating base_env
        """
        if base_env is None:
            self.base_env = RoutingEnv(topo_name=topo_name)
        else:
            self.base_env = base_env

        self.topo = self.base_env.topo
        self.n_nodes = self.base_env.topo.n_nodes
        self.n_links = self.base_env.n_links

        # Build adjacency matrix
        self.adj_matrix = nx.to_numpy_array(self.topo.graph, dtype=np.float32)
        self.adj_flat_size = self.n_nodes * self.n_nodes

        # Observation = [link_utils (30,) + adj_matrix (144,)] = 174 dims
        obs_size = self.n_links + self.adj_flat_size
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(obs_size,), dtype=np.float32
        )

        # Action space: same as base
        self.action_space = base_env.action_space if base_env else spaces.Box(
            low=0.1, high=10.0, shape=(30,), dtype=np.float32
        )

        # Store graph structure for later access
        self.edge_index = self._build_edge_index()

        print(
            f"[SimpleGraphRoutingEnv] {self.n_nodes} nodes, {self.n_links} links, "
            f"obs_dim={obs_size}"
        )

    def _build_edge_index(self) -> np.ndarray:
        """Build edge index from topology."""
        edges = []
        for u, v in self.topo.graph.edges():
            edges.append([u, v])
        return np.array(edges, dtype=np.int32)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset and return observation."""
        obs, info = self.base_env.reset(seed=seed, options=options)
        graph_obs = self._make_graph_obs(obs)
        return graph_obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute step and return graph observation."""
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        graph_obs = self._make_graph_obs(obs)
        return graph_obs, reward, terminated, truncated, info

    def _make_graph_obs(self, flat_obs: np.ndarray) -> np.ndarray:
        """Concatenate link utilization with adjacency matrix.

        Args:
            flat_obs: Link utilization array [n_links]

        Returns:
            Observation: [link_utils + flattened_adjacency]
        """
        # Normalize adjacency to [0, 1]
        adj_normalized = self.adj_matrix / (self.adj_matrix.max() + 1e-8)

        # Concatenate
        obs = np.concatenate([flat_obs, adj_normalized.flatten()])
        return obs.astype(np.float32)

    def close(self):
        """Close base environment."""
        self.base_env.close()

    @property
    def graph_info(self):
        """Return graph structure for use in policies."""
        return {
            "n_nodes": self.n_nodes,
            "edge_index": self.edge_index,
            "adj_matrix": self.adj_matrix,
        }


if __name__ == "__main__":
    print("Testing SimpleGraphRoutingEnv...")
    env = SimpleGraphRoutingEnv(topo_name="abilene")

    print("Resetting...")
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Min: {obs.min():.2f}, Max: {obs.max():.2f}")

    print("Running 1 step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Reward: {reward:.2f}")
    print(f"  Observation shape: {obs.shape}")

    print(f"\n  Graph info keys: {env.graph_info.keys()}")
    print(f"  Edge index shape: {env.graph_info['edge_index'].shape}")

    print("\n✅ SimpleGraphRoutingEnv works!")
