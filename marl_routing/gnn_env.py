"""Graph-based gym environment wrapper for GNN policies.

Converts flat link utilization state to graph representation suitable for GNN.
"""
from __future__ import annotations

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
from gymnasium import spaces
from torch_geometric.data import Data

from marl_routing.routing_env import RoutingEnv


class GraphRoutingEnv(gym.Env):
    """Wrapper around RoutingEnv that provides graph-structured observations.

    Instead of flat link utilization vector, outputs PyTorch Geometric graph:
    - Nodes: Network nodes
    - Edges: Links with utilization as edge attributes
    - Node features: Ones (or could be degree, etc.)
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env: RoutingEnv):
        """Wrap a RoutingEnv to provide graph observations.

        Args:
            base_env: RoutingEnv instance to wrap
        """
        self.base_env = base_env
        self.topo = base_env.topo
        self.n_nodes = base_env.topo.n_nodes
        self.n_links = base_env.n_links

        # Build edge index (adjacency list) from topology
        # For PyTorch Geometric: edge_index is [2, num_edges] tensor
        # Each column is [source, target]
        edge_list = []
        self.edge_to_link_idx = {}  # Map (src, dst) → link index in flat state

        link_idx = 0
        for u, v in self.topo.graph.edges():
            edge_list.append([u, v])
            self.edge_to_link_idx[(u, v)] = link_idx
            link_idx += 1

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.num_edges = self.edge_index.shape[1]

        # Action space: same as base env
        self.action_space = base_env.action_space

        # Observation space: Data objects (can't be directly represented in spaces)
        # For compatibility with SB3, we use None
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(self.n_nodes * 2,), dtype=np.float32
        )

        print(
            f"[GraphRoutingEnv] {self.n_nodes} nodes, {self.num_edges} edges, "
            f"action space: {self.action_space}"
        )

    def reset(self, seed=None, options=None) -> tuple[Data, dict]:
        """Reset environment and return graph observation."""
        obs, info = self.base_env.reset(seed=seed, options=options)
        graph_obs = self._flat_to_graph(obs)
        return graph_obs, info

    def step(self, action: np.ndarray) -> tuple[Data, float, bool, bool, dict]:
        """Execute one step and return graph observation."""
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        graph_obs = self._flat_to_graph(obs)
        return graph_obs, reward, terminated, truncated, info

    def _flat_to_graph(self, flat_obs: np.ndarray) -> Data:
        """Convert flat link utilization vector to graph representation.

        Args:
            flat_obs: Link utilization array (one entry per directed link)

        Returns:
            PyTorch Geometric Data object with:
            - x: Node features [n_nodes, 1] (all ones)
            - edge_index: [2, num_edges] adjacency
            - edge_attr: [num_edges, 1] link utilization
        """
        # Node features: ones for all nodes
        x = torch.ones((self.n_nodes, 1), dtype=torch.float32)

        # Edge attributes: link utilization
        # flat_obs has 30 entries for Abilene (15 undirected = 30 directed)
        # Map each (u, v) edge to corresponding utilization
        edge_attr_list = []
        for i, (u, v) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
            u, v = int(u), int(v)
            # Use utilization from flat state
            if i < len(flat_obs):
                util = flat_obs[i]
            else:
                util = 0.0
            edge_attr_list.append([util])

        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        # Create Data object
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            num_nodes=self.n_nodes,
        )

        return data

    def render(self):
        """Not implemented."""
        pass

    def close(self):
        """Close base environment."""
        self.base_env.close()


if __name__ == "__main__":
    # Quick test
    print("Creating GraphRoutingEnv...")
    base_env = RoutingEnv(topo_name="abilene")
    env = GraphRoutingEnv(base_env)

    print("Resetting...")
    obs, info = env.reset()
    print(f"  Observation type: {type(obs)}")
    print(f"  Observation: {obs}")
    print(f"  x shape: {obs.x.shape}, edge_index shape: {obs.edge_index.shape}")
    print(f"  edge_attr shape: {obs.edge_attr.shape}")

    print("\nRunning 1 step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Reward: {reward:.2f}")
    print(f"  Obs edge_attr: {obs.edge_attr.shape}")

    print("\n✅ GraphRoutingEnv test passed!")
