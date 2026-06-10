"""GNN feature extractor for SB3 policies.

Processes adjacency-augmented observations through a simple GNN architecture.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SimpleGNNExtractor(BaseFeaturesExtractor):
    """Simple GNN feature extractor for graph-augmented observations.

    Input: [link_utils (30,) + adj_flat (144,)]
    - Reshape adj to (12, 12) adjacency matrix
    - Apply graph convolution
    - Return features
    """

    def __init__(self, observation_space, n_nodes: int = 12, hidden_dim: int = 64):
        """Initialize GNN extractor.

        Args:
            observation_space: Gym observation space
            n_nodes: Number of nodes in graph
            hidden_dim: Hidden dimension
        """
        super().__init__(observation_space, features_dim=hidden_dim)

        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        # Infer n_links from observation space: obs = [link_utils (n_links,) | adj_flat (n_nodes²,)]
        # So: n_links = obs_size - n_nodes²
        obs_size = observation_space.shape[0]
        self.n_links = obs_size - (n_nodes * n_nodes)

        # Extract graph info from observation
        # Obs format: [link_utils(n_links) | adj_flat(n_nodes²)]
        self.adj_start_idx = self.n_links

        # Simple graph processing: use adjacency to weight information
        # Instead of full GCN, use: features = W * (A @ node_features)
        # where A is adjacency and node features are derived from incident links

        # Node feature aggregation: map 30 links to 12 node features
        # Heuristic: sum utilization of incident edges
        self.link_to_nodes = self._build_link_to_nodes_mapping()

        # Graph-aware MLP
        # Input: 12 node features (aggregated from links) + adjacency info
        self.gnn_layers = nn.Sequential(
            nn.Linear(self.n_nodes + self.n_nodes * self.n_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

        print(
            f"[SimpleGNNExtractor] {n_nodes} nodes, {self.n_links} links, "
            f"output_dim={hidden_dim}"
        )

    def _build_link_to_nodes_mapping(self):
        """Build mapping from links to nodes for feature aggregation.

        Abilene has 15 undirected links (30 directed). Each node can have
        multiple incident edges. We aggregate utilization across incident edges.
        """
        # For Abilene topology, we need the actual structure
        # Simplified: use a fixed mapping based on typical abilene structure
        # In production, this would come from the environment's topology

        # For now, create a simple default: each node gets a portion of links
        mapping = {}
        for node in range(self.n_nodes):
            # Simplified: assign links to nodes in round-robin
            mapping[node] = [i for i in range(self.n_links) if i % self.n_nodes == node]
        return mapping

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from graph-augmented observation.

        Args:
            observations: [batch_size, 174] where 174 = 30 links + 144 adj

        Returns:
            Features [batch_size, hidden_dim]
        """
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)

        batch_size = observations.shape[0]

        # Split observation into links and adjacency
        link_utils = observations[:, :self.n_links]  # [batch, 30]
        adj_flat = observations[:, self.adj_start_idx :]  # [batch, 144]

        # Reshape adjacency to matrix
        adj = adj_flat.view(batch_size, self.n_nodes, self.n_nodes)

        # Aggregate link utilization to node features
        # Simple approach: create node features by gathering incident link utils
        node_features_list = []
        for node in range(self.n_nodes):
            incident_links = self.link_to_nodes.get(node, [0])
            if incident_links:
                node_feat = link_utils[:, incident_links].mean(dim=1)
            else:
                node_feat = torch.zeros(batch_size, device=observations.device)
            node_features_list.append(node_feat)

        node_features = torch.stack(node_features_list, dim=1)  # [batch, n_nodes]

        # Simple graph convolution: aggregate neighbor features using adjacency
        # neighbor_features = A @ node_features
        adj_normalized = adj / (adj.sum(dim=2, keepdim=True) + 1e-8)
        aggregated = torch.bmm(adj_normalized, node_features.unsqueeze(2)).squeeze(
            2
        )  # [batch, n_nodes]

        # Combine node and aggregated features
        graph_input = torch.cat([aggregated, adj_flat], dim=1)  # [batch, 12 + 144]

        # Process through GNN layers
        features = self.gnn_layers(graph_input)

        return features


class SeqGNNExtractor(BaseFeaturesExtractor):
    """GNN extractor for the SequentialRoutingEnv observation.

    Obs layout: [link_utils(n_arcs) | adj_flat(n_nodes^2) | cand_feats(k*F) | rate(1)]

    The GNN processes the current network state (per-arc utilization aggregated to
    nodes, then mixed via the adjacency) into a graph embedding; this is combined
    with the current flow's candidate features + demand so the Discrete(k) head can
    score each path with full-network context (enabling non-myopic decisions).
    """

    def __init__(self, observation_space, n_nodes: int, n_arcs: int,
                 arc_list, hidden_dim: int = 64):
        super().__init__(observation_space, features_dim=hidden_dim)
        self.n_nodes = n_nodes
        self.n_arcs = n_arcs
        self.adj_start = n_arcs
        self.adj_end = n_arcs + n_nodes * n_nodes
        self.extra_dim = observation_space.shape[0] - self.adj_end  # cand_feats + rate

        # Exact arc -> tail-node incidence (each directed arc contributes to its src)
        inc = torch.zeros(n_nodes, n_arcs)
        for ai, (u, v) in enumerate(arc_list):
            inc[u, ai] = 1.0
            inc[v, ai] = 1.0  # also count at head node
        # row-normalize for mean aggregation
        inc = inc / inc.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer("incidence", inc)  # [n_nodes, n_arcs]

        self.graph_layers = nn.Sequential(
            nn.Linear(n_nodes + n_nodes * n_nodes, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.extra_dim, hidden_dim), nn.ReLU(),
        )
        print(f"[SeqGNNExtractor] {n_nodes} nodes, {n_arcs} arcs, "
              f"extra={self.extra_dim}, out={hidden_dim}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        b = obs.shape[0]
        util = obs[:, : self.n_arcs]                       # [b, n_arcs]
        adj_flat = obs[:, self.adj_start : self.adj_end]   # [b, n_nodes^2]
        extra = obs[:, self.adj_end :]                     # [b, cand+rate]

        # aggregate arc utilization to nodes via fixed incidence
        node_feat = util @ self.incidence.t()              # [b, n_nodes]
        adj = adj_flat.view(b, self.n_nodes, self.n_nodes)
        adj_n = adj / (adj.sum(dim=2, keepdim=True) + 1e-8)
        mixed = torch.bmm(adj_n, node_feat.unsqueeze(2)).squeeze(2)  # [b, n_nodes]

        graph_emb = self.graph_layers(torch.cat([mixed, adj_flat], dim=1))
        return self.head(torch.cat([graph_emb, extra], dim=1))


if __name__ == "__main__":
    import gymnasium as gym

    # Test
    print("Testing SimpleGNNExtractor...")
    obs_space = gym.spaces.Box(low=0, high=100, shape=(174,), dtype="float32")
    extractor = SimpleGNNExtractor(obs_space, n_nodes=12, hidden_dim=64)

    # Create dummy batch
    batch = torch.randn(4, 174)
    features = extractor(batch)
    print(f"✅ Input shape: {batch.shape}")
    print(f"✅ Output shape: {features.shape}")
    assert features.shape == (4, 64), f"Expected (4, 64), got {features.shape}"
    print("✅ SimpleGNNExtractor works!")
