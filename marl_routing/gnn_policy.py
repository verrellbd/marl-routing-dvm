"""GNN-based policy for network routing via Stable-Baselines3.

Uses PyTorch Geometric for graph neural network backbone.
"""
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNFeatureExtractor(BaseFeaturesExtractor):
    """GNN-based feature extractor for graph observations.

    Takes PyTorch Geometric Data objects (network graphs) and extracts features.
    """

    def __init__(
        self,
        observation_space,
        n_nodes: int = 12,
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        output_dim: int = 64,
    ):
        """Initialize GNN feature extractor.

        Args:
            observation_space: Gym observation space (unused, for compatibility)
            n_nodes: Number of nodes in the graph
            hidden_dim: Hidden dimension for GNN layers
            num_gnn_layers: Number of GCN layers
            output_dim: Output feature dimension
        """
        super().__init__(observation_space, features_dim=output_dim)

        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(1, hidden_dim))  # Input: node features (1 dim)
        for _ in range(num_gnn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Readout: mean pooling + MLP
        # After GCN: [num_nodes, hidden_dim] → pool → [hidden_dim]
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        print(
            f"[GNNFeatureExtractor] {num_gnn_layers} GCN layers, "
            f"hidden={hidden_dim}, output={output_dim}"
        )

    def forward(self, observations: Any) -> torch.Tensor:
        """Extract features from graph observation.

        Args:
            observations: Batch of graph Data objects or stacked tensors

        Returns:
            Feature vectors [batch_size, output_dim]
        """
        # Handle both single graph and batch of graphs
        if isinstance(observations, Data):
            # Single graph
            x = observations.x
            edge_index = observations.edge_index
            batch = None
        elif isinstance(observations, list):
            # Batch of graphs
            from torch_geometric.data import Batch

            graphs = observations
            batch_data = Batch.from_data_list(graphs)
            x = batch_data.x
            edge_index = batch_data.edge_index
            batch = batch_data.batch
        elif isinstance(observations, torch.Tensor):
            # Fallback: convert tensor to dummy graph
            # This shouldn't happen in normal operation
            batch_size = observations.shape[0]
            x = torch.ones((batch_size * self.n_nodes, 1), dtype=torch.float32)
            # Create complete graph for simplicity
            nodes = torch.arange(self.n_nodes)
            edge_index = torch.cartesian_prod(nodes, nodes).t().contiguous()
            batch = torch.repeat_interleave(
                torch.arange(batch_size), self.n_nodes
            )
        else:
            raise TypeError(f"Unsupported observation type: {type(observations)}")

        # Ensure tensors are on correct device
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)

        # Forward through GCN layers
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = torch.relu(x)

        # Global mean pooling
        if batch is not None:
            batch = batch.to(device)
            x = global_mean_pool(x, batch)
        else:
            # Single graph: average all nodes
            x = x.mean(dim=0, keepdim=True)

        # Readout MLP
        features = self.readout_mlp(x)

        return features


class GNNActorCriticPolicy(ActorCriticPolicy):
    """Actor-Critic policy with GNN feature extractor."""

    def __init__(self, *args, n_nodes: int = 12, gnn_hidden: int = 64, **kwargs):
        self.n_nodes = n_nodes
        self.gnn_hidden = gnn_hidden
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        """Build feature extractor (overrides default)."""
        self.features_extractor = GNNFeatureExtractor(
            self.observation_space,
            n_nodes=self.n_nodes,
            hidden_dim=self.gnn_hidden,
            output_dim=64,
        )
        self.features_dim = self.features_extractor.output_dim


if __name__ == "__main__":
    # Quick test
    import sys

    sys.path.insert(0, "/home/uceedv1/thesis")

    from marl_routing.gnn_env import GraphRoutingEnv
    from marl_routing.routing_env import RoutingEnv

    print("Creating environments...")
    base_env = RoutingEnv(topo_name="abilene")
    env = GraphRoutingEnv(base_env)

    print("Testing GNN feature extractor...")
    obs, _ = env.reset()

    extractor = GNNFeatureExtractor(
        env.observation_space, n_nodes=12, hidden_dim=64, output_dim=64
    )
    features = extractor(obs)
    print(f"  Input: x={obs.x.shape}, edge_attr={obs.edge_attr.shape}")
    print(f"  Output: {features.shape}")
    print(f"✅ GNN works!")
