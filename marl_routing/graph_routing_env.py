"""Topology-agnostic sequential routing environment.

Same MDP and reward as `MultiTrafficSequentialEnv` (per-flow k-shortest-path
selection; reward = -(marginal bottleneck increase) - lambda * extra hops), but the
observation is emitted in a STRUCTURED, PADDED form so that a single policy with
shared weights can act on ANY topology:

  arc_util [A_MAX] | arc_cap [A_MAX] | arc_mask [A_MAX] | arc_src [A_MAX] |
  arc_dst [A_MAX]  | cand_arcs [K*LMAX] | cand_feats [K*CF] | rate [1]

Nothing in the layout is indexed by node identity, so the extractor's weights are
shared across nodes/arcs and transfer between networks (unlike the flat n_nodes^2
adjacency used by the original env, which fixes the input to one topology).

The env can hold SEVERAL topologies at once and samples one per episode, which is
what makes cross-topology training (and zero-shot transfer) possible.
"""
from typing import List, Sequence, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from marl_routing.gnn_routing_agent import compute_ksp
from marl_routing.topology import load as load_topology

# ---- padding bounds (cover Abilene 12/30, GEANT 22/72, Germany50 50/176) ----
# Kept tight: every padded slot costs compute in the extractor on EVERY step.
N_MAX = 56
A_MAX = 176        # exactly Germany50's arc count (the largest network)
LMAX = 10          # max arcs on a candidate path (observed max = 9)
CAND_FEATS = 3     # [resulting_bottleneck%, path_len_norm, min_headroom%]
CAP_SCALE = 10000.0  # Mbps normaliser for capacity feature

def obs_size(k_paths: int) -> int:
    return 5 * A_MAX + k_paths * LMAX + k_paths * CAND_FEATS + 1


class _TopoBundle:
    """Precomputed per-topology structure (built once)."""

    def __init__(self, topo_name: str, pairs, matrices, k_paths: int):
        self.name = topo_name
        self.topo = load_topology(topo_name)
        self.pairs = list(pairs)
        self.matrices = [np.asarray(m, dtype=np.float64) for m in matrices]
        self.arcs = list(self.topo.graph.edges())
        self.arc_index = {a: i for i, a in enumerate(self.arcs)}
        self.n_arcs = len(self.arcs)
        self.n_nodes = self.topo.n_nodes
        if self.n_arcs > A_MAX or self.n_nodes > N_MAX:
            raise ValueError(f"{topo_name}: {self.n_nodes} nodes / {self.n_arcs} arcs "
                             f"exceeds padding bounds ({N_MAX}/{A_MAX})")
        self.cap = np.array([self.topo.graph[u][v]["capacity"] for (u, v) in self.arcs],
                            dtype=np.float64)
        # candidate arc-paths per node-pair
        self.pair_arc_paths: List[List[List[int]]] = []
        for (s, d) in self.pairs:
            paths = compute_ksp(self.topo.graph, s, d, k=k_paths)
            ap = [[self.arc_index[(p[i], p[i + 1])] for i in range(len(p) - 1)]
                  for p in paths]
            while len(ap) < k_paths:
                ap.append(ap[-1])
            for p in ap:
                if len(p) > LMAX:
                    raise ValueError(f"{topo_name}: candidate path of {len(p)} arcs > LMAX={LMAX}")
            self.pair_arc_paths.append(ap)
        # static arc structure vectors (padded)
        self.arc_src = np.zeros(A_MAX, dtype=np.float32)
        self.arc_dst = np.zeros(A_MAX, dtype=np.float32)
        self.arc_mask = np.zeros(A_MAX, dtype=np.float32)
        self.arc_cap_f = np.zeros(A_MAX, dtype=np.float32)
        for i, (u, v) in enumerate(self.arcs):
            self.arc_src[i] = u
            self.arc_dst[i] = v
            self.arc_mask[i] = 1.0
            self.arc_cap_f[i] = self.cap[i] / CAP_SCALE
        # OSPF shortest-path arcs per pair, precomputed once (topology is static).
        # Used for the baseline AND for the per-topology reward normaliser, so it
        # must not re-run Dijkstra every episode.
        self.ospf_arc_paths: List[List[int]] = []
        for (s, d) in self.pairs:
            p = nx.shortest_path(self.topo.graph, s, d)
            self.ospf_arc_paths.append(
                [self.arc_index[(p[j], p[j + 1])] for j in range(len(p) - 1)])

    def ospf_max_util(self, rates) -> float:
        load = np.zeros(self.n_arcs)
        for i, arc_path in enumerate(self.ospf_arc_paths):
            if rates[i] > 0:
                load[arc_path] += rates[i]
        return float((100.0 * load / self.cap).max())


class GraphSeqRoutingEnv(gym.Env):
    """Sequential per-flow routing over one or more topologies (structured obs)."""

    metadata = {"render_modes": []}

    def __init__(self, topo_specs: Sequence[Tuple[str, object, object]],
                 k_paths: int = 3, seed: int = 0, delay_penalty: float = 0.5,
                 normalize_reward: bool = True):
        """topo_specs: list of (topo_name, pairs, matrices).

        normalize_reward: express the congestion term as a PERCENTAGE OF THIS
        EPISODE'S OSPF BOTTLENECK rather than in raw utilisation points. Networks
        sit at very different absolute utilisations (Abilene ~90%, Germany50
        ~160%), so without this a single shared policy over-weights the
        high-utilisation networks and mis-serves the others. Set False to recover
        the original reward exactly (used for parity tests).
        """
        self.k_paths = k_paths
        self.delay_penalty = delay_penalty
        self.normalize_reward = normalize_reward
        self._rng = np.random.RandomState(seed)
        self.bundles = [_TopoBundle(n, p, m, k_paths) for (n, p, m) in topo_specs]
        self.action_space = spaces.Discrete(k_paths)
        self.observation_space = spaces.Box(-1e6, 1e6, (obs_size(k_paths),),
                                            dtype=np.float32)
        self._forced = None
        self._forced_topo = None
        names = ", ".join(f"{b.name}({b.n_nodes}n/{b.n_arcs}a)" for b in self.bundles)
        print(f"[GraphSeqRoutingEnv] {len(self.bundles)} topologies: {names}; k={k_paths}")

    # ---- evaluation helpers -------------------------------------------------
    def set_matrix(self, rates, topo_idx: int = 0):
        self._forced = np.asarray(rates, dtype=np.float64)
        self._forced_topo = topo_idx

    def ospf_max_util(self, rates, topo_idx: int = 0) -> float:
        return self.bundles[topo_idx].ospf_max_util(rates)

    # ---- core ---------------------------------------------------------------
    def _start_episode(self, b: _TopoBundle, rates):
        idx = [i for i in range(len(b.pairs)) if rates[i] > 0]
        idx.sort(key=lambda i: -rates[i])
        self.b = b
        self.order = idx
        self.rates = rates
        self.load = np.zeros(b.n_arcs, dtype=np.float64)
        self.pos = 0
        self.cur_max = 0.0
        # per-episode congestion scale: this matrix's OSPF bottleneck (>=1 to be safe)
        self.ospf_ref = max(b.ospf_max_util(rates), 1.0) if self.normalize_reward else 100.0

    def _cand_block(self):
        """(cand_arcs [k*LMAX], cand_feats [k*CF], rate)."""
        b = self.b
        pair_i = self.order[self.pos]
        rate = self.rates[pair_i]
        util = 100.0 * self.load / b.cap
        cur_max = float(util.max()) if self.pos > 0 else 0.0
        arcs_blk = np.zeros(self.k_paths * LMAX, dtype=np.float32)
        feats = np.zeros(self.k_paths * CAND_FEATS, dtype=np.float32)
        for ci, arc_path in enumerate(b.pair_arc_paths[pair_i]):
            for j, ai in enumerate(arc_path):
                arcs_blk[ci * LMAX + j] = ai + 1        # 0 = padding
            after = 100.0 * (self.load[arc_path] + rate) / b.cap[arc_path]
            feats[ci * CAND_FEATS + 0] = max(cur_max, float(after.max()))
            feats[ci * CAND_FEATS + 1] = len(arc_path) / b.n_nodes
            feats[ci * CAND_FEATS + 2] = float((100.0 - util[arc_path]).min())
        return arcs_blk, feats, rate

    def _obs(self, terminal: bool = False) -> np.ndarray:
        b = self.b
        arc_util = np.zeros(A_MAX, dtype=np.float32)
        arc_util[: b.n_arcs] = (100.0 * self.load / b.cap).astype(np.float32)
        if terminal:
            arcs_blk = np.zeros(self.k_paths * LMAX, dtype=np.float32)
            feats = np.zeros(self.k_paths * CAND_FEATS, dtype=np.float32)
            rate = 0.0
        else:
            arcs_blk, feats, rate = self._cand_block()
        return np.concatenate([
            arc_util / 100.0, b.arc_cap_f, b.arc_mask, b.arc_src, b.arc_dst,
            arcs_blk, feats, np.array([rate / 1000.0], np.float32),
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._forced is not None:
            b = self.bundles[self._forced_topo]
            rates = self._forced
            self._forced = None
        else:
            b = self.bundles[self._rng.randint(len(self.bundles))]
            rates = b.matrices[self._rng.randint(len(b.matrices))]
        self._start_episode(b, rates)
        return self._obs(), {"topo": b.name}

    def step(self, action):
        b = self.b
        pair_i = self.order[self.pos]
        k = int(action) % self.k_paths
        arc_path = b.pair_arc_paths[pair_i][k]
        self.load[arc_path] += self.rates[pair_i]
        new_max = float((100.0 * self.load / b.cap).max())
        extra_hops = len(arc_path) - len(b.pair_arc_paths[pair_i][0])
        # congestion term in "% of this episode's OSPF bottleneck" so the
        # congestion/delay trade-off is comparable across networks
        congestion = (new_max - self.cur_max) * (100.0 / self.ospf_ref)
        reward = float(-congestion - self.delay_penalty * extra_hops)
        self.cur_max = new_max
        self.pos += 1
        terminated = self.pos >= len(self.order)
        info = {"max_util": self.cur_max} if terminated else {}
        return self._obs(terminal=terminated), reward, terminated, False, info

    def myopic_max_util(self, rates, topo_idx: int = 0) -> float:
        self.set_matrix(rates, topo_idx)
        obs, _ = self.reset()
        base = 5 * A_MAX + self.k_paths * LMAX
        done = False
        info = {}
        while not done:
            feats = obs[base: base + self.k_paths * CAND_FEATS]
            obs, _, done, _, info = self.step(int(np.argmin(feats[0::CAND_FEATS])))
        return info["max_util"]

    def close(self):
        pass
