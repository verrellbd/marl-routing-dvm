"""Topology-agnostic GNN feature extractor.

Unlike `SeqGNNExtractor` (which feeds a flat n_nodes^2 adjacency into an MLP and is
therefore tied to one topology), every weight here is SHARED across arcs and nodes:

  1. shared per-arc encoder            : [util, capacity] -> h_e
  2. R rounds of arc <-> node message passing, using the arc_src/arc_dst index
     vectors carried in the observation (so the graph can be any size)
  3. per-candidate-path embedding by aggregating the embeddings of the arcs that
     lie on that path  (the "path state depends on its links" principle)
  4. fixed-size output: [global arc pooling | k path embeddings | candidate feats | rate]

Because k (candidate paths) is fixed while the number of nodes/arcs is not, the
output dimension is constant for any topology -> a single policy transfers across
networks. Padding is masked throughout.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from marl_routing.graph_routing_env import A_MAX, N_MAX, LMAX, CAND_FEATS


class TopoAgnosticGNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, k_paths: int = 3, hidden_dim: int = 64,
                 rounds: int = 2):
        super().__init__(observation_space, features_dim=hidden_dim)
        self.k = k_paths
        self.h = hidden_dim
        self.rounds = rounds

        # observation slice offsets
        self.o_util = 0
        self.o_cap = A_MAX
        self.o_mask = 2 * A_MAX
        self.o_src = 3 * A_MAX
        self.o_dst = 4 * A_MAX
        self.o_cand = 5 * A_MAX
        self.o_feat = self.o_cand + k_paths * LMAX
        self.o_rate = self.o_feat + k_paths * CAND_FEATS

        self.arc_enc = nn.Sequential(          # shared over arcs
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # one shared update block per round: [h_e, node_src, node_dst] -> h_e
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(rounds)
        ])
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * (1 + k_paths) + k_paths * CAND_FEATS + 1, 128),
            nn.ReLU(), nn.Linear(128, hidden_dim), nn.ReLU(),
        )
        print(f"[TopoAgnosticGNNExtractor] k={k_paths}, hidden={hidden_dim}, "
              f"rounds={rounds}, out={hidden_dim} (topology-agnostic)")

    def _scatter_nodes(self, h, idx2, mask2, B):
        """Mean-aggregate arc embeddings into node slots.

        idx2/mask2 carry BOTH endpoints concatenated ([B, 2A]), so one fused
        index_add replaces the two separate tail/head scatters — each node's
        feature is the mean over all arcs incident to it (either direction).
        """
        H = h.shape[-1]
        flat = (torch.arange(B, device=h.device).unsqueeze(1) * N_MAX + idx2).reshape(-1)
        src = (h * mask2.unsqueeze(-1)).reshape(-1, H)
        acc = torch.zeros(B * N_MAX, H, device=h.device, dtype=h.dtype)
        acc.index_add_(0, flat, src)
        cnt = torch.zeros(B * N_MAX, 1, device=h.device, dtype=h.dtype)
        cnt.index_add_(0, flat, mask2.reshape(-1, 1))
        return (acc / cnt.clamp(min=1.0)).view(B, N_MAX, H)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]

        util = obs[:, self.o_util:self.o_cap]                     # [B,A]
        cap = obs[:, self.o_cap:self.o_mask]
        mask = obs[:, self.o_mask:self.o_src]                     # [B,A] 1=real
        src = obs[:, self.o_src:self.o_dst].long().clamp(0, N_MAX - 1)
        dst = obs[:, self.o_dst:self.o_cand].long().clamp(0, N_MAX - 1)
        cand = obs[:, self.o_cand:self.o_feat].long()             # [B,k*LMAX] arc+1
        feats = obs[:, self.o_feat:self.o_rate]                   # [B,k*CF]
        rate = obs[:, self.o_rate:self.o_rate + 1]                # [B,1]

        # 1. shared per-arc encoding
        h = self.arc_enc(torch.stack([util, cap], dim=-1))        # [B,A,H]
        h = h * mask.unsqueeze(-1)

        # 2. message passing rounds (arc -> node -> arc), shared weights.
        # Endpoint indices/masks concatenated once outside the loop so each round
        # needs a single fused scatter instead of two.
        idx2 = torch.cat([src, dst], dim=1)                       # [B,2A]
        mask2 = torch.cat([mask, mask], dim=1)
        exp_s = src.unsqueeze(-1).expand(-1, -1, self.h)
        exp_d = dst.unsqueeze(-1).expand(-1, -1, self.h)
        for blk in self.upd:
            node = self._scatter_nodes(torch.cat([h, h], dim=1), idx2, mask2, B)
            g_s = torch.gather(node, 1, exp_s)
            g_d = torch.gather(node, 1, exp_d)
            h = blk(torch.cat([h, g_s, g_d], dim=-1)) * mask.unsqueeze(-1)

        # 3. per-candidate path embedding: mean of the arcs on that path
        cm = (cand > 0).float()                                   # [B,k*LMAX]
        ci = (cand - 1).clamp(min=0)
        gathered = torch.gather(h, 1, ci.unsqueeze(-1).expand(-1, -1, self.h))
        gathered = gathered * cm.unsqueeze(-1)
        gathered = gathered.view(B, self.k, LMAX, self.h).sum(dim=2)
        denom = cm.view(B, self.k, LMAX).sum(dim=2, keepdim=True).clamp(min=1.0)
        z = (gathered / denom).reshape(B, self.k * self.h)        # [B,k*H]

        # 4. global (masked-mean) arc pooling + heads
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.out(torch.cat([pooled, z, feats, rate], dim=1))
