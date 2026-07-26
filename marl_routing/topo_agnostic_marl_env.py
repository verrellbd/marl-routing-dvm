#!/usr/bin/env python3
"""Topology-agnostic multi-agent routing environment.

Same per-node, hop-by-hop MARL as MultiAgentRoutingEnv, but the observation is
made TOPOLOGY-INVARIANT so a single shared-weight MAPPO policy can route any
network (and transfer zero-shot to an unseen one):

  * The original env fed each agent a one-hot of (current node, destination),
    which is meaningless across topologies (node 5 in Abilene != node 5 in GEANT).
    We DROP the one-hots. The destination still enters — through each neighbour's
    normalised distance-to-destination — so the policy stays destination-aware
    without any node-identity input.
  * Per-neighbour features (util, resulting bottleneck, headroom, dist-to-dst) are
    already structural; we pad them to a fixed MAX_DEG.
  * The centralized critic sees padded global arc utilisations (to A_MAX) + the
    current bottleneck + rate — no node identities.

Fixed dims across every topology  => one MAPPO network trains on a mix of networks.
The trainer (mappo.MAPPO) is UNCHANGED; it only reads obs_dim/gstate_dim/act_dim.

Optional link-failure domain randomisation (fail_links>0) drops random links per
episode and recomputes reachability — this is the RESILIENCE training signal.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import networkx as nx
import numpy as np

from marl_routing.topology import load as load_topology

MAX_DEG = 12          # >= max out-degree over all topologies (GEANT=8); headroom for more
A_MAX = 176           # >= max directed-arc count (germany50); matches graph_routing_env
NF = 4                # per-neighbour: [arc_util, resulting_bottleneck, headroom, dist_norm]
N_MAX = 56            # >= max node count (germany50=50); matches graph_routing_env
NODE_F = 6            # per-node graph features (see _graph_obs) — for the GNN-actor variant


class _MARLBundle:
    """Per-topology precomputed structure for the intact graph."""

    def __init__(self, topo_name, pairs, matrices):
        self.name = topo_name
        self.topo = load_topology(topo_name)
        self.G = self.topo.graph
        self.n_nodes = self.topo.n_nodes
        self.pairs = list(pairs)
        self.matrices = [np.asarray(m, dtype=np.float64) for m in matrices]
        self.arcs = list(self.G.edges())
        self.arc_index = {a: i for i, a in enumerate(self.arcs)}
        self.n_arcs = len(self.arcs)
        if self.n_arcs > A_MAX:
            raise ValueError(f"{topo_name}: {self.n_arcs} arcs > A_MAX={A_MAX}")
        self.cap = np.array([self.G[u][v]["capacity"] for (u, v) in self.arcs], float)
        self.nbrs = {n: sorted(self.G.successors(n)) for n in range(self.n_nodes)}
        if max(len(v) for v in self.nbrs.values()) > MAX_DEG:
            raise ValueError(f"{topo_name}: degree > MAX_DEG={MAX_DEG}")
        self.nbr_arc = {n: [self.arc_index[(n, m)] for m in self.nbrs[n]]
                        for n in range(self.n_nodes)}
        self.dist_to = {d: nx.single_source_shortest_path_length(self.G.reverse(copy=False), d)
                        for d in range(self.n_nodes)}

    def ospf_max_util(self, rates) -> float:
        load = np.zeros(self.n_arcs)
        for i, (s, d) in enumerate(self.pairs):
            if rates[i] > 0:
                p = nx.shortest_path(self.G, s, d)
                for j in range(len(p) - 1):
                    load[self.arc_index[(p[j], p[j + 1])]] += rates[i]
        return float((100.0 * load / self.cap).max())


class TopoAgnosticMARLEnv:
    obs_dim = MAX_DEG * NF + 1        # per-neighbour feats + rate (NO one-hots)
    gstate_dim = A_MAX + 2           # padded arc utils + cur_max + rate
    act_dim = MAX_DEG
    # --- extra dims for the GNN-actor variant (Option B: multi-hop message passing) ---
    n_max = N_MAX
    node_f_dim = NODE_F
    max_deg = MAX_DEG
    nf_dim = NF

    def __init__(self, topo_specs, seed: int = 0, delay_penalty: float = 0.5,
                 stretch: int = 2, max_stretch=4, fail_links: int = 0,
                 normalize_reward: bool = True):
        self.bundles = [_MARLBundle(n, p, m) for (n, p, m) in topo_specs]
        self.delay_penalty = delay_penalty
        self.stretch = stretch
        self.max_stretch = max_stretch
        self.fail_links = fail_links        # 0 = intact; >0 = drop k random links/episode
        # congestion reward as % of THIS episode's OSPF bottleneck -> comparable across
        # topologies (Abilene ~100%, GEANT ~160%); the fix that made single-agent work.
        self.normalize_reward = normalize_reward
        self._rng = np.random.RandomState(seed)
        self._forced = None
        self._forced_topo = 0
        names = ", ".join(f"{b.name}({b.n_nodes}n)" for b in self.bundles)
        print(f"[TopoAgnosticMARLEnv] {len(self.bundles)} topos: {names}; "
              f"stretch={stretch} max_stretch={max_stretch} fail_links={fail_links}")

    # ---- evaluation hooks ----
    def set_matrix(self, rates, topo_idx: int = 0):
        self._forced = np.asarray(rates, dtype=np.float64)
        self._forced_topo = topo_idx

    def ospf_max_util(self, rates, topo_idx: int = 0) -> float:
        return self.bundles[topo_idx].ospf_max_util(rates)

    # ---- per-episode topology view (supports link failure) ----
    def _build_view(self, b: _MARLBundle):
        """Return (nbrs, nbr_arc, dist_to, live_arc_mask) for this episode.
        With fail_links>0, drop random links and recompute reachability."""
        if self.fail_links <= 0:
            self._live = np.ones(b.n_arcs, dtype=bool)
            return b.nbrs, b.nbr_arc, b.dist_to
        # drop k random undirected links, keep graph connected (retry a few times)
        und = list({tuple(sorted(e)) for e in b.arcs})
        for _ in range(10):
            drop = self._rng.choice(len(und), size=min(self.fail_links, len(und)), replace=False)
            dead = set()
            for di in drop:
                u, v = und[di]
                dead.add((u, v)); dead.add((v, u))
            H = b.G.copy()
            H.remove_edges_from([e for e in dead if H.has_edge(*e)])
            if nx.is_strongly_connected(H):
                break
        live = np.array([b.arcs[i] not in dead for i in range(b.n_arcs)], dtype=bool)
        nbrs = {n: [m for m in b.nbrs[n] if (n, m) not in dead] for n in range(b.n_nodes)}
        nbr_arc = {n: [b.arc_index[(n, m)] for m in nbrs[n]] for n in range(b.n_nodes)}
        dist_to = {d: nx.single_source_shortest_path_length(H.reverse(copy=False), d)
                   for d in range(b.n_nodes)}
        self._live = live
        return nbrs, nbr_arc, dist_to

    # ---- episode ----
    def reset(self, rates=None):
        if rates is not None:
            self.set_matrix(rates, self._forced_topo)
        if self._forced is not None:
            self.b = self.bundles[self._forced_topo]
            r = self._forced; self._forced = None
        else:
            self.b = self.bundles[self._rng.randint(len(self.bundles))]
            r = self.b.matrices[self._rng.randint(len(self.b.matrices))]
        self.nbrs, self.nbr_arc, self.dist_to = self._build_view(self.b)
        self._build_adj()
        self.rates = r
        self.ospf_ref = max(self.b.ospf_max_util(r), 1.0) if self.normalize_reward else 100.0
        self.flows = sorted([(s, d, r[i], i) for i, (s, d) in enumerate(self.b.pairs)
                             if r[i] > 0 and self.dist_to[d].get(s, 10**9) < 10**9],
                            key=lambda x: -x[2])
        self.load = np.zeros(self.b.n_arcs, dtype=np.float64)
        self.cur_max = 0.0
        self.flow_i = 0
        self.cur_path: Dict[int, List[int]] = {}
        self._begin_flow()
        return self._obs(), self._mask(), self._gstate()

    def _begin_flow(self):
        s, d, rate, pi = self.flows[self.flow_i]
        self.cur_node = s; self.cur_dst = d; self.cur_rate = rate; self.cur_pi = pi
        self.visited = {s}
        self.cur_path[pi] = [s]
        self.cur_shortest = self.dist_to[d].get(s, 10 ** 9)

    def _valid(self) -> List[bool]:
        n, d = self.cur_node, self.cur_dst
        v = [False] * MAX_DEG
        if n == d:
            return v
        dn = self.dist_to[d].get(n, 10 ** 9)
        hops = len(self.cur_path[self.cur_pi]) - 1
        for i, m in enumerate(self.nbrs[n]):
            dm = self.dist_to[d].get(m, 10 ** 9)
            if m not in self.visited and dm < dn + self.stretch and dm < 10 ** 9:
                if (self.max_stretch is not None
                        and hops + 1 + dm > self.cur_shortest + self.max_stretch):
                    continue
                v[i] = True
        if not any(v):
            best, bi = 10 ** 9, None
            for i, m in enumerate(self.nbrs[n]):
                dm = self.dist_to[d].get(m, 10 ** 9)
                if m not in self.visited and dm < best:
                    best, bi = dm, i
            if bi is not None:
                v[bi] = True
        return v

    def _mask(self):
        return np.array(self._valid(), dtype=np.float32)

    def _obs(self):
        b = self.b
        util = 100.0 * self.load / b.cap
        o = np.zeros(self.obs_dim, dtype=np.float32)
        dmax = max(1, max(self.dist_to[self.cur_dst].values()))
        nb = self.nbrs[self.cur_node]
        na = self.nbr_arc[self.cur_node]
        for i in range(MAX_DEG):
            if i < len(nb):
                m, ai = nb[i], na[i]
                after = 100.0 * (self.load[ai] + self.cur_rate) / b.cap[ai]
                o[i * NF + 0] = util[ai]
                o[i * NF + 1] = max(self.cur_max, after)
                o[i * NF + 2] = max(0.0, 100.0 - util[ai])
                o[i * NF + 3] = self.dist_to[self.cur_dst].get(m, dmax) / dmax
        o[MAX_DEG * NF] = self.cur_rate / 1000.0
        return o

    def _gstate(self):
        b = self.b
        g = np.zeros(self.gstate_dim, dtype=np.float32)
        g[:b.n_arcs] = (100.0 * self.load / b.cap).astype(np.float32)
        g[A_MAX] = self.cur_max
        g[A_MAX + 1] = self.cur_rate / 1000.0
        return g

    # ---- graph observation for the GNN-actor (Option B) ----
    # Mean-aggregation adjacency over the CURRENT episode's live neighbours (respects
    # link failures), with self-loops, row-normalised. Structure only — no node identity.
    def _build_adj(self):
        b = self.b
        A = np.zeros((N_MAX, N_MAX), dtype=np.float32)
        for n in range(b.n_nodes):
            A[n, n] = 1.0
            for m in self.nbrs[n]:
                A[n, m] = 1.0
        deg = A.sum(1, keepdims=True); deg[deg == 0] = 1.0
        self._adj_norm = A / deg
        mask = np.zeros(N_MAX, dtype=np.float32); mask[:b.n_nodes] = 1.0
        self._node_mask = mask

    def _graph_obs(self):
        """Per-node structural features for L-round message passing (topology-invariant).
        Returns (node_feat[N_MAX,NODE_F], adj_norm[N_MAX,N_MAX], node_mask[N_MAX],
        cur_idx, nbr_idx[MAX_DEG]).  is_current/is_dst let message passing propagate the
        routing task multi-hop (like the MANET paper's per-node embedding exchange)."""
        b = self.b
        util = 100.0 * self.load / b.cap
        nf = np.zeros((N_MAX, NODE_F), dtype=np.float32)
        dmap = self.dist_to[self.cur_dst]
        dmax = max(1, max(dmap.values()))
        for n in range(b.n_nodes):
            oa = self.nbr_arc[n]
            if oa:
                u = util[oa]
                maxu = float(u.max()); minhead = float(max(0.0, 100.0 - float(u.min())))
            else:
                maxu = 0.0; minhead = 0.0
            nf[n, 0] = 1.0 if n == self.cur_node else 0.0
            nf[n, 1] = 1.0 if n == self.cur_dst else 0.0
            nf[n, 2] = dmap.get(n, dmax) / dmax
            nf[n, 3] = maxu / 100.0
            nf[n, 4] = minhead / 100.0
            nf[n, 5] = len(self.nbrs[n]) / MAX_DEG
        nbr_idx = np.full(MAX_DEG, -1, dtype=np.int64)
        for i, m in enumerate(self.nbrs[self.cur_node]):
            nbr_idx[i] = m
        return nf, self._adj_norm, self._node_mask, int(self.cur_node), nbr_idx

    def step(self, action_idx: int):
        b = self.b
        valid = self._valid()
        a = int(action_idx)
        if a >= MAX_DEG or a >= len(self.nbrs[self.cur_node]) or not valid[a]:
            cands = [i for i, ok in enumerate(valid) if ok]
            a = cands[0] if cands else int(np.argmin(
                [self.dist_to[self.cur_dst].get(m, 10 ** 9) for m in self.nbrs[self.cur_node]]))
        nxt = self.nbrs[self.cur_node][a]
        ai = self.nbr_arc[self.cur_node][a]
        d = self.cur_dst
        is_detour = self.dist_to[d].get(nxt, 10 ** 9) >= self.dist_to[d].get(self.cur_node, 0)
        self.load[ai] += self.cur_rate
        new_max = float((100.0 * self.load / b.cap).max())
        congestion = (new_max - self.cur_max) * (100.0 / self.ospf_ref)
        reward = -congestion - self.delay_penalty * (1.0 if is_detour else 0.0)
        self.cur_max = new_max
        self.visited.add(nxt)
        self.cur_path[self.cur_pi].append(nxt)
        self.cur_node = nxt
        done = False
        info = {}
        if self.cur_node == self.cur_dst:
            self.flow_i += 1
            if self.flow_i >= len(self.flows):
                done = True
                info["max_util"] = self.cur_max
                info["paths"] = self.cur_path
            else:
                self._begin_flow()
        return self._obs(), self._mask(), self._gstate(), reward, done, info

    def rollout_paths(self, act_fn, rates, topo_idx: int = 0):
        self.set_matrix(rates, topo_idx)
        obs, mask, _ = self.reset()
        done = False
        info = {}
        while not done:
            a = act_fn(obs, mask)
            obs, mask, _, _, done, info = self.step(a)
        self.last_max = info["max_util"]
        return info["paths"]
