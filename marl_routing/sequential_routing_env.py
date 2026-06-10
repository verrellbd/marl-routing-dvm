#!/usr/bin/env python3
"""
Sequential per-flow routing environment (better credit assignment than the
one-shot joint-action AnalyticalRoutingEnv).

Flows are routed ONE per step, largest-demand first. At each step the agent sees
the current link utilizations + the current flow's k candidate paths (each
summarized by the bottleneck it would create) and chooses one path. The link
loads accumulate within the episode, so the agent observes the consequences of
its earlier decisions — the natural "GNN routes the network" formulation.

Action:  Discrete(k)  — which candidate path for the current flow
Reward:  -(increase in global max-util caused by this placement)  [potential-based;
         telescopes to -final_max_util, the same objective as OSPF/greedy]
Obs:     [link_utils(n_arcs) | adj_flat(n_nodes^2) | per-candidate feats(k*F) | flow_rate]
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

CAND_FEATS = 3  # per candidate: [resulting_bottleneck%, path_len_norm, min_headroom%]


class SequentialRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        topo_name: str = "abilene",
        traffic_file: Path | str = None,
        k_paths: int = 3,
        load_scale: float = 1.0,
    ):
        self.topo_name = topo_name
        self.topo = load_topology(topo_name)
        self.k_paths = k_paths
        self.load_scale = load_scale

        if traffic_file is None:
            traffic_file = (
                Path(__file__).resolve().parent.parent
                / "results" / f"traffic_{topo_name}_α1.5_min30.json"
            )
        self.traffic_file = Path(traffic_file)
        with self.traffic_file.open() as f:
            raw = json.load(f)["flows"]
        flows = [dict(fl, rate_mbps=fl["rate_mbps"] * load_scale) for fl in raw]
        # Largest-demand first (route the biggest, least-flexible flows when empty)
        self.flows = sorted(flows, key=lambda f: -f["rate_mbps"])
        self.n_flows = len(self.flows)

        self.arcs = list(self.topo.graph.edges())
        self.arc_index = {a: i for i, a in enumerate(self.arcs)}
        self.n_arcs = len(self.arcs)
        self.cap = np.array(
            [self.topo.graph[u][v]["capacity"] for (u, v) in self.arcs],
            dtype=np.float64,
        )

        # Candidate paths (as arc-index lists) per flow
        self.flow_arc_paths: List[List[List[int]]] = []
        for fl in self.flows:
            paths = compute_ksp(self.topo.graph, fl["src"], fl["dst"], k=k_paths)
            arc_paths = []
            for p in paths:
                arc_paths.append(
                    [self.arc_index[(p[i], p[i + 1])] for i in range(len(p) - 1)]
                )
            # pad to k by repeating last path (so action space is uniform)
            while len(arc_paths) < k_paths:
                arc_paths.append(arc_paths[-1])
            self.flow_arc_paths.append(arc_paths)

        self._adj_flat = nx.to_numpy_array(self.topo.graph, dtype=np.float32).flatten()

        self.action_space = spaces.Discrete(k_paths)
        obs_size = self.n_arcs + self.topo.n_nodes ** 2 + k_paths * CAND_FEATS + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1e6, shape=(obs_size,), dtype=np.float32
        )

        self.ospf_max_util = self._ospf_max_util()
        print(
            f"[SequentialRoutingEnv] {self.n_flows} flows, {self.topo.n_nodes} nodes, "
            f"{self.n_arcs} arcs, k={k_paths}, load_scale={load_scale}, "
            f"OSPF={self.ospf_max_util:.2f}%"
        )

    # ---------------------------------------------------------------- internals
    def _ospf_max_util(self) -> float:
        load = np.zeros(self.n_arcs)
        for fl in self.flows:
            p = nx.shortest_path(self.topo.graph, fl["src"], fl["dst"])
            for i in range(len(p) - 1):
                load[self.arc_index[(p[i], p[i + 1])]] += fl["rate_mbps"]
        return float((100.0 * load / self.cap).max())

    def _candidate_feats(self) -> Tuple[np.ndarray, float]:
        """Features for each candidate path of the current flow."""
        rate = self.flows[self.cur]["rate_mbps"]
        util = 100.0 * self.load / self.cap
        cur_max = float(util.max()) if self.cur > 0 else 0.0
        feats = np.zeros(self.k_paths * CAND_FEATS, dtype=np.float32)
        for ci, arc_path in enumerate(self.flow_arc_paths[self.cur]):
            after = util.copy()
            after[arc_path] = 100.0 * (self.load[arc_path] + rate) / self.cap[arc_path]
            resulting_bottleneck = max(cur_max, float(after[arc_path].max()))
            path_len_norm = len(arc_path) / self.topo.n_nodes
            min_headroom = float((100.0 - util[arc_path]).min())
            base = ci * CAND_FEATS
            feats[base + 0] = resulting_bottleneck
            feats[base + 1] = path_len_norm
            feats[base + 2] = min_headroom
        return feats, rate

    def _obs(self) -> np.ndarray:
        util = (100.0 * self.load / self.cap).astype(np.float32)
        feats, rate = self._candidate_feats()
        rate_norm = np.array([rate / 1000.0], dtype=np.float32)
        return np.concatenate([util, self._adj_flat, feats, rate_norm]).astype(np.float32)

    # ----------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.load = np.zeros(self.n_arcs, dtype=np.float64)
        self.cur = 0
        self.cur_max = 0.0
        return self._obs(), {"ospf_max_util": self.ospf_max_util}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        arc_path = self.flow_arc_paths[self.cur][int(action) % self.k_paths]
        rate = self.flows[self.cur]["rate_mbps"]
        self.load[arc_path] += rate

        new_max = float((100.0 * self.load / self.cap).max())
        reward = float(-(new_max - self.cur_max))  # marginal bottleneck increase
        self.cur_max = new_max

        self.cur += 1
        terminated = self.cur >= self.n_flows
        info = {"ospf_max_util": self.ospf_max_util}
        if terminated:
            info["max_util"] = self.cur_max
            info["beats_ospf"] = self.cur_max < self.ospf_max_util
            obs = self._obs_terminal()
        else:
            obs = self._obs()
        return obs, reward, terminated, False, info

    def _obs_terminal(self) -> np.ndarray:
        # all flows placed; candidate feats undefined -> zeros for that block
        util = (100.0 * self.load / self.cap).astype(np.float32)
        feats = np.zeros(self.k_paths * CAND_FEATS, dtype=np.float32)
        return np.concatenate(
            [util, self._adj_flat, feats, np.zeros(1, dtype=np.float32)]
        ).astype(np.float32)

    def close(self):
        pass


class MultiTrafficSequentialEnv(gym.Env):
    """Sequential per-flow routing over a DISTRIBUTION of traffic matrices.

    Same mechanics as SequentialRoutingEnv, but each reset() samples a different
    demand matrix from a provided set (all over the same node-pairs, so candidate
    paths are precomputed once). Used to train a single GNN policy that generalizes
    to unseen traffic — the experiment that justifies a learned GNN over a per-matrix
    heuristic, and answers the 'dynamic traffic' research question.

    matrices: list of rate-vectors aligned with `pairs` (Mbps per ordered node-pair).
    """

    metadata = {"render_modes": []}

    def __init__(self, topo_name: str, pairs, matrices, k_paths: int = 3, seed: int = 0):
        self.topo_name = topo_name
        self.topo = load_topology(topo_name)
        self.k_paths = k_paths
        self.pairs = list(pairs)
        self.matrices = [np.asarray(m, dtype=np.float64) for m in matrices]
        self._rng = np.random.RandomState(seed)

        self.arcs = list(self.topo.graph.edges())
        self.arc_index = {a: i for i, a in enumerate(self.arcs)}
        self.n_arcs = len(self.arcs)
        self.cap = np.array(
            [self.topo.graph[u][v]["capacity"] for (u, v) in self.arcs], dtype=np.float64
        )

        # candidate arc-paths per node-pair (shared across all matrices)
        self.pair_arc_paths: List[List[List[int]]] = []
        for (s, d) in self.pairs:
            paths = compute_ksp(self.topo.graph, s, d, k=k_paths)
            ap = [[self.arc_index[(p[i], p[i + 1])] for i in range(len(p) - 1)] for p in paths]
            while len(ap) < k_paths:
                ap.append(ap[-1])
            self.pair_arc_paths.append(ap)

        self._adj_flat = nx.to_numpy_array(self.topo.graph, dtype=np.float32).flatten()
        self.action_space = spaces.Discrete(k_paths)
        obs_size = self.n_arcs + self.topo.n_nodes ** 2 + k_paths * CAND_FEATS + 1
        self.observation_space = spaces.Box(0.0, 1e6, (obs_size,), dtype=np.float32)
        print(f"[MultiTrafficSequentialEnv] {len(self.matrices)} matrices, "
              f"{len(self.pairs)} pairs, k={k_paths}")

    def set_matrix(self, rates):
        """Force a specific matrix (used for deterministic evaluation)."""
        self._forced = np.asarray(rates, dtype=np.float64)

    def _start_episode(self, rates):
        # build flow order: largest demand first, skip zero-rate pairs
        idx = [i for i in range(len(self.pairs)) if rates[i] > 0]
        idx.sort(key=lambda i: -rates[i])
        self.order = idx
        self.rates = rates
        self.load = np.zeros(self.n_arcs, dtype=np.float64)
        self.pos = 0
        self.cur_max = 0.0

    def ospf_max_util(self, rates) -> float:
        load = np.zeros(self.n_arcs)
        for i, (s, d) in enumerate(self.pairs):
            if rates[i] <= 0:
                continue
            p = nx.shortest_path(self.topo.graph, s, d)
            for j in range(len(p) - 1):
                load[self.arc_index[(p[j], p[j + 1])]] += rates[i]
        return float((100.0 * load / self.cap).max())

    def _cand_feats(self):
        pair_i = self.order[self.pos]
        rate = self.rates[pair_i]
        util = 100.0 * self.load / self.cap
        cur_max = float(util.max()) if self.pos > 0 else 0.0
        feats = np.zeros(self.k_paths * CAND_FEATS, dtype=np.float32)
        for ci, arc_path in enumerate(self.pair_arc_paths[pair_i]):
            after_arc = 100.0 * (self.load[arc_path] + rate) / self.cap[arc_path]
            feats[ci * CAND_FEATS + 0] = max(cur_max, float(after_arc.max()))
            feats[ci * CAND_FEATS + 1] = len(arc_path) / self.topo.n_nodes
            feats[ci * CAND_FEATS + 2] = float((100.0 - util[arc_path]).min())
        return feats, rate

    def _obs(self):
        util = (100.0 * self.load / self.cap).astype(np.float32)
        feats, rate = self._cand_feats()
        return np.concatenate(
            [util, self._adj_flat, feats, np.array([rate / 1000.0], np.float32)]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if getattr(self, "_forced", None) is not None:
            rates = self._forced; self._forced = None
        else:
            rates = self.matrices[self._rng.randint(len(self.matrices))]
        self._start_episode(rates)
        return self._obs(), {}

    def step(self, action):
        pair_i = self.order[self.pos]
        arc_path = self.pair_arc_paths[pair_i][int(action) % self.k_paths]
        self.load[arc_path] += self.rates[pair_i]
        new_max = float((100.0 * self.load / self.cap).max())
        reward = float(-(new_max - self.cur_max))
        self.cur_max = new_max
        self.pos += 1
        terminated = self.pos >= len(self.order)
        info = {}
        if terminated:
            info["max_util"] = self.cur_max
            util = (100.0 * self.load / self.cap).astype(np.float32)
            obs = np.concatenate([util, self._adj_flat,
                                  np.zeros(self.k_paths * CAND_FEATS + 1, np.float32)]).astype(np.float32)
        else:
            obs = self._obs()
        return obs, reward, terminated, False, info

    def myopic_max_util(self, rates) -> float:
        self.set_matrix(rates); obs, _ = self.reset(); done = False
        base = self.n_arcs + self.topo.n_nodes ** 2
        while not done:
            feats = obs[base: base + self.k_paths * CAND_FEATS]
            obs, _, done, _, info = self.step(int(np.argmin(feats[0::CAND_FEATS])))
        return info["max_util"]

    def close(self):
        pass


if __name__ == "__main__":
    for sc in [1.0, 2.0, 3.0]:
        env = SequentialRoutingEnv(
            traffic_file=Path(__file__).resolve().parent.parent
            / "results" / "traffic_abilene_α1.5_min30.json",
            k_paths=3, load_scale=sc,
        )
        # myopic policy: pick candidate with lowest resulting bottleneck (feat 0)
        obs, _ = env.reset()
        done = False
        while not done:
            feats = obs[env.n_arcs + env.topo.n_nodes ** 2 :
                        env.n_arcs + env.topo.n_nodes ** 2 + env.k_paths * CAND_FEATS]
            bott = feats[0::CAND_FEATS]
            obs, _, done, _, info = env.step(int(np.argmin(bott)))
        print(f"  myopic-argmin max_util = {info['max_util']:.2f}%  (OSPF {env.ospf_max_util:.2f}%)\n")
