#!/usr/bin/env python3
"""
Analytical routing environment (fast training surrogate for ns-3).

The agent selects, per flow, one of its k candidate paths. Given a static
traffic matrix and link capacities, the resulting per-link load — and hence the
max link utilization — is a deterministic function of the routing choice. We
compute it directly (the standard traffic-engineering / LP link-utilization
model) instead of running a packet-level ns-3 simulation. This is validated to
match ns-3's routing-driven utilization (modulo ns-3's UdpEcho 2x echo factor),
while being ~10,000x faster, which makes PPO training actually converge.

Interface mirrors GNNRoutingAgentEnv so the same SimpleGNNExtractor + PPO code
works unchanged. ns-3 remains the high-fidelity *evaluation* backend.

State:  [directed-link utilizations (n_directed_links) | adjacency_flat (n_nodes^2)]
Action: per-flow path index in [0, k-1]
Reward: -max_link_utilization_pct   (same objective as the ns-3 env / OSPF baseline)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from marl_routing.topology import load as load_topology
from marl_routing.gnn_routing_agent import compute_ksp


def route_max_util(
    topo,
    flows: List[dict],
    arc_index: Dict[Tuple[int, int], int],
    chosen_paths: List[List[int]],
) -> Tuple[np.ndarray, float]:
    """Compute per-arc utilization and max utilization for a set of chosen paths.

    Args:
        topo: Topology (DiGraph with 'capacity' in Mbps on each arc)
        flows: list of flow dicts with 'rate_mbps'
        arc_index: map (u, v) -> index into the link vector
        chosen_paths: for each flow, the node list of its selected path

    Returns:
        (utils array [n_arcs] in percent, max_util percent)
    """
    n_arcs = len(arc_index)
    load = np.zeros(n_arcs, dtype=np.float64)  # Mbps per directed arc

    for flow, path in zip(flows, chosen_paths):
        rate = flow["rate_mbps"]
        for i in range(len(path) - 1):
            arc = (path[i], path[i + 1])
            idx = arc_index.get(arc)
            if idx is not None:
                load[idx] += rate

    utils = np.zeros(n_arcs, dtype=np.float32)
    for (u, v), idx in arc_index.items():
        cap = topo.graph[u][v]["capacity"]  # Mbps
        utils[idx] = 100.0 * load[idx] / cap if cap > 0 else 0.0

    return utils, float(utils.max()) if n_arcs else 0.0


class AnalyticalRoutingEnv(gym.Env):
    """Fast analytical routing environment for GNN-PPO training.

    Each step the agent selects a path for every flow; the environment computes
    the resulting link utilizations analytically and returns -max_util as reward.
    Multi-step episodes let the GNN observe current congestion and re-route to
    reduce it (iterative improvement on a static demand).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topo_name: str = "abilene",
        traffic_file: Path | str = None,
        k_paths: int = 3,
        episode_len: int = 20,
        load_scale: float = 1.0,
        reward_mode: str = "neg_max",
        pnorm: float = 6.0,
    ):
        self.topo_name = topo_name
        self.topo = load_topology(topo_name)
        self.k_paths = k_paths
        self.episode_len = episode_len
        self.load_scale = load_scale
        # reward_mode: "neg_max" = -max_util (sparse) | "shaped" = -p-norm of utils
        # (dense: every congested link contributes gradient, still pushes the max down)
        self.reward_mode = reward_mode
        self.pnorm = pnorm
        self.step_count = 0
        self.episode_count = 0

        # Load traffic (optionally scale all flow rates to reach a congested regime)
        if traffic_file is None:
            traffic_file = (
                Path(__file__).resolve().parent.parent
                / "results"
                / f"traffic_{topo_name}_α1.0_min50.json"
            )
        self.traffic_file = Path(traffic_file)
        with self.traffic_file.open() as f:
            raw_flows = json.load(f)["flows"]
        self.flows = [
            dict(fl, rate_mbps=fl["rate_mbps"] * load_scale) for fl in raw_flows
        ]
        self.n_flows = len(self.flows)

        # Fixed arc ordering for the link-utilization vector
        self.arcs = list(self.topo.graph.edges())
        self.arc_index = {arc: i for i, arc in enumerate(self.arcs)}
        n_arcs = len(self.arcs)

        # Precompute k candidate paths per flow
        self.flow_paths: Dict[int, List[List[int]]] = {}
        for i, flow in enumerate(self.flows):
            self.flow_paths[i] = compute_ksp(
                self.topo.graph, flow["src"], flow["dst"], k=k_paths
            )

        # Adjacency (constant) for the GNN observation
        self._adj_flat = nx.to_numpy_array(
            self.topo.graph, dtype=np.float32
        ).flatten()

        # Spaces (match GNN extractor: obs = [link_utils | adj_flat])
        self.action_space = spaces.MultiDiscrete([k_paths] * self.n_flows)
        obs_size = n_arcs + self.topo.n_nodes * self.topo.n_nodes
        self.observation_space = spaces.Box(
            low=0.0, high=1e6, shape=(obs_size,), dtype=np.float32
        )

        # OSPF (min-hop) baseline, for reference / reward shaping context
        self.ospf_max_util = self.compute_ospf_max_util()

        print(
            f"[AnalyticalRoutingEnv] {self.n_flows} flows, {self.topo.n_nodes} nodes, "
            f"{n_arcs} arcs, k={k_paths}, load_scale={load_scale}, "
            f"OSPF max_util={self.ospf_max_util:.2f}%"
        )

    # ------------------------------------------------------------------ helpers
    def _paths_for_action(self, action: np.ndarray) -> List[List[int]]:
        paths = []
        for fid in range(self.n_flows):
            cands = self.flow_paths[fid]
            paths.append(cands[int(action[fid]) % len(cands)])
        return paths

    def _obs(self, utils: np.ndarray) -> np.ndarray:
        return np.concatenate([utils, self._adj_flat]).astype(np.float32)

    def compute_ospf_max_util(self) -> float:
        """Max link utilization under min-hop shortest-path routing (OSPF baseline)."""
        paths = [
            nx.shortest_path(self.topo.graph, f["src"], f["dst"])
            for f in self.flows
        ]
        _, mx = route_max_util(self.topo, self.flows, self.arc_index, paths)
        return mx

    # -------------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.episode_count += 1
        self.step_count = 0
        self.best_max_util = float("inf")
        utils = np.zeros(len(self.arcs), dtype=np.float32)
        info = {"episode": self.episode_count, "step": 0, "max_util": 0.0,
                "ospf_max_util": self.ospf_max_util}
        return self._obs(utils), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        chosen = self._paths_for_action(action)
        utils, max_util = route_max_util(
            self.topo, self.flows, self.arc_index, chosen
        )

        if self.reward_mode == "shaped":
            # p-norm (in %) — a smooth surrogate for the max that gives gradient to
            # every congested link, not just the single bottleneck.
            reward = float(-np.linalg.norm(utils.astype(np.float64), ord=self.pnorm))
        else:
            reward = float(-max_util)
        self.best_max_util = min(self.best_max_util, max_util)
        terminated = self.step_count >= self.episode_len

        info = {
            "step": self.step_count,
            "max_util": max_util,
            "best_max_util": self.best_max_util,
            "ospf_max_util": self.ospf_max_util,
            "beats_ospf": max_util < self.ospf_max_util,
        }
        return self._obs(utils), reward, terminated, False, info

    def close(self) -> None:
        pass


if __name__ == "__main__":
    import numpy as np

    for tf in ["traffic_abilene_α1.0_min50.json", "traffic_abilene_α1.5_min50.json"]:
        env = AnalyticalRoutingEnv(
            traffic_file=Path(__file__).resolve().parent.parent / "results" / tf,
            k_paths=3,
        )
        obs, info = env.reset()
        # Random-policy best over 200 tries, as a sanity reference
        best = float("inf")
        for _ in range(200):
            a = env.action_space.sample()
            _, _, _, _, i = env.step(a)
            best = min(best, i["max_util"])
            if i["step"] >= env.episode_len:
                env.reset()
        print(f"  OSPF={env.ospf_max_util:.2f}%  random-best-of-200={best:.2f}%\n")
