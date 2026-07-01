#!/usr/bin/env python3
"""
Multi-agent per-node routing environment (the MARL core).

Each NODE is an agent. A flow's path is traced hop-by-hop: at the node a packet
currently sits, the local agent observes only LOCAL state (its incident link
utilisations + the candidate next hops toward the destination + the flow rate +
which node it is and where the packet is going) and chooses the next hop. Link
loads accumulate as hops are committed, so later decisions see the congestion that
earlier ones created — genuinely decentralized routing.

Action semantics: a fixed per-node neighbour list, so action i ALWAYS means "the
i-th neighbour of the current node". A dynamic mask enables only the valid next
hops. Validity = the neighbour makes progress toward the destination within a
bounded `stretch` (it may detour up to `stretch-1` extra hops beyond the shortest
distance) AND has not already been visited on this flow's trace. The visited-set +
finite nodes guarantee loop-free, terminating paths; `stretch` controls how much
detour freedom the agents have to relieve congestion (stretch=1 => shortest-path
DAG only; stretch=2 => may step one hop "sideways/away", ~k-shortest freedom).

Interfaces (driven by a custom MAPPO trainer; this is NOT a gym.Env):
  reset(rates=None) -> obs, mask, gstate
  step(action_idx)  -> obs, mask, gstate, reward, done, info
  rollout_paths(act_fn, rates) -> {pair_index: node_path}   # deterministic, for ns-3

The actor consumes (obs, mask); the centralized critic consumes gstate.
The team reward telescopes to -(final max link utilisation) (minus a small per-hop
delay term), the same objective the OSPF baseline and single-agent GNN are scored on.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np


from marl_routing.topology import load as load_topology


class MultiAgentRoutingEnv:
    def __init__(self, topo_name: str, pairs, matrices, seed: int = 0,
                 delay_penalty: float = 0.0, stretch: int = 1, max_stretch=None):
        self.topo_name = topo_name
        self.topo = load_topology(topo_name)
        self.G = self.topo.graph
        self.n_nodes = self.topo.n_nodes
        self.pairs = list(pairs)
        self.matrices = [np.asarray(m, dtype=np.float64) for m in matrices]
        self.delay_penalty = delay_penalty
        self.stretch = stretch
        # hard hop cap: final path length may not exceed shortest-path hops + max_stretch.
        # Constrains the AGENT's forwarding choice only; the topology is unchanged.
        self.max_stretch = max_stretch
        self._rng = np.random.RandomState(seed)
        self._forced = None

        # ---- arcs / capacities ----
        self.arcs = list(self.G.edges())
        self.arc_index = {a: i for i, a in enumerate(self.arcs)}
        self.n_arcs = len(self.arcs)
        self.cap = np.array([self.G[u][v]["capacity"] for (u, v) in self.arcs],
                            dtype=np.float64)

        # ---- shortest-path distance to every destination (reverse view) ----
        self.dist_to: Dict[int, Dict[int, int]] = {
            d: nx.single_source_shortest_path_length(self.G.reverse(copy=False), d)
            for d in range(self.n_nodes)
        }

        # ---- fixed per-node neighbour list -> stable action semantics ----
        self.nbrs: Dict[int, List[int]] = {
            n: sorted(self.G.successors(n)) for n in range(self.n_nodes)
        }
        self.nbr_arc: Dict[int, List[int]] = {
            n: [self.arc_index[(n, m)] for m in self.nbrs[n]] for n in range(self.n_nodes)
        }
        self.max_deg = max(len(v) for v in self.nbrs.values())

        # observation / state sizes
        self.NF = 4  # per-neighbour feats: [arc_util, resulting_bottleneck, headroom, neigh_dist_norm]
        self.obs_dim = (
            2 * self.n_nodes            # current-node one-hot + destination one-hot
            + self.max_deg * self.NF    # per-neighbour features (masked)
            + 1                         # flow rate (norm)
        )
        self.gstate_dim = self.n_arcs + 2 * self.n_nodes + 2  # utils + node + dst + rate + cur_max
        self.act_dim = self.max_deg

    # ------------------------------------------------------------------ helpers
    def ospf_max_util(self, rates) -> float:
        load = np.zeros(self.n_arcs)
        for i, (s, d) in enumerate(self.pairs):
            if rates[i] <= 0:
                continue
            p = nx.shortest_path(self.G, s, d)
            for j in range(len(p) - 1):
                load[self.arc_index[(p[j], p[j + 1])]] += rates[i]
        return float((100.0 * load / self.cap).max())

    def set_matrix(self, rates):
        self._forced = np.asarray(rates, dtype=np.float64)

    # ------------------------------------------------------------------- episode
    def reset(self, rates=None):
        if rates is not None:
            self._forced = np.asarray(rates, dtype=np.float64)
        if self._forced is not None:
            r = self._forced; self._forced = None
        else:
            r = self.matrices[self._rng.randint(len(self.matrices))]
        self.rates = r
        self.flows = sorted(
            [(s, d, r[i], i) for i, (s, d) in enumerate(self.pairs) if r[i] > 0],
            key=lambda x: -x[2])
        self.load = np.zeros(self.n_arcs, dtype=np.float64)
        self.cur_max = 0.0
        self.flow_i = 0
        self.cur_path: Dict[int, List[int]] = {}
        self._begin_flow()
        return self._obs(), self._mask(), self._gstate()

    def _begin_flow(self):
        s, d, rate, pi = self.flows[self.flow_i]
        self.cur_node = s
        self.cur_dst = d
        self.cur_rate = rate
        self.cur_pi = pi
        self.visited = {s}
        self.cur_path[pi] = [s]
        self.cur_shortest = self.dist_to[d].get(s, 10 ** 9)  # OSPF hop-count for this flow

    def _valid(self) -> List[bool]:
        """Per-neighbour validity at the current (node, dst, visited)."""
        n, d = self.cur_node, self.cur_dst
        if n == d:
            return [False] * self.max_deg
        dn = self.dist_to[d].get(n, 10 ** 9)
        hops_so_far = len(self.cur_path[self.cur_pi]) - 1
        v = [False] * self.max_deg
        for i, m in enumerate(self.nbrs[n]):
            dm = self.dist_to[d].get(m, 10 ** 9)
            # progress within stretch, not revisiting
            if m not in self.visited and dm < dn + self.stretch and dm < 10 ** 9:
                # hard hop cap: reject if taking m then going shortest would exceed
                # shortest-path hops + max_stretch (guarantees bounded final path length)
                if (self.max_stretch is not None
                        and hops_so_far + 1 + dm > self.cur_shortest + self.max_stretch):
                    continue
                v[i] = True
        if not any(v):  # dead-end safety: force the shortest-distance unvisited hop
            best, bi = 10 ** 9, None
            for i, m in enumerate(self.nbrs[n]):
                dm = self.dist_to[d].get(m, 10 ** 9)
                if m not in self.visited and dm < best:
                    best, bi = dm, i
            if bi is None:  # everything closer is visited -> any UNVISITED hop toward d
                for i, m in enumerate(self.nbrs[n]):
                    if m not in self.visited and self.dist_to[d].get(m, 10 ** 9) < dn:
                        bi = i; break
            if bi is not None:
                v[bi] = True
        return v

    def _mask(self) -> np.ndarray:
        return np.array(self._valid(), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        util = 100.0 * self.load / self.cap
        o = np.zeros(self.obs_dim, dtype=np.float32)
        p = 0
        o[p + self.cur_node] = 1.0; p += self.n_nodes
        o[p + self.cur_dst] = 1.0;  p += self.n_nodes
        dmax = max(1, max(self.dist_to[self.cur_dst].values()))
        for i in range(self.max_deg):
            base = p + i * self.NF
            if i < len(self.nbrs[self.cur_node]):
                m = self.nbrs[self.cur_node][i]
                ai = self.nbr_arc[self.cur_node][i]
                after = 100.0 * (self.load[ai] + self.cur_rate) / self.cap[ai]
                o[base + 0] = util[ai]
                o[base + 1] = max(self.cur_max, after)
                o[base + 2] = max(0.0, 100.0 - util[ai])
                o[base + 3] = self.dist_to[self.cur_dst].get(m, dmax) / dmax
        p += self.max_deg * self.NF
        o[p] = self.cur_rate / 1000.0
        return o

    def _gstate(self) -> np.ndarray:
        util = (100.0 * self.load / self.cap).astype(np.float32)
        g = np.zeros(self.gstate_dim, dtype=np.float32)
        p = 0
        g[p:p + self.n_arcs] = util; p += self.n_arcs
        g[p + self.cur_node] = 1.0; p += self.n_nodes
        g[p + self.cur_dst] = 1.0;  p += self.n_nodes
        g[p] = self.cur_rate / 1000.0; p += 1
        g[p] = self.cur_max
        return g

    def step(self, action_idx: int):
        valid = self._valid()
        a = int(action_idx)
        if a >= self.max_deg or not valid[a]:   # mask safety -> first valid neighbour
            cands = [i for i, ok in enumerate(valid) if ok]
            if cands:
                a = cands[0]
            else:  # emergency (e.g. corrupted/terminal state): min-distance neighbour
                d = self.cur_dst
                a = int(np.argmin([self.dist_to[d].get(m, 10 ** 9) for m in self.nbrs[self.cur_node]]))
        nxt = self.nbrs[self.cur_node][a]
        ai = self.nbr_arc[self.cur_node][a]
        # QoS delay term: penalise only NON-PROGRESS (sideways) hops — i.e. the extra
        # hops over the shortest path. Strict-progress hops (distance decreases) are
        # free, so the policy keeps OSPF-short paths when uncongested and only "spends"
        # detours to relieve a bottleneck. (dm==dn under stretch=1 ⇒ a detour hop.)
        d = self.cur_dst
        is_detour = self.dist_to[d].get(nxt, 10 ** 9) >= self.dist_to[d].get(self.cur_node, 0)
        self.load[ai] += self.cur_rate
        new_max = float((100.0 * self.load / self.cap).max())
        reward = -(new_max - self.cur_max) - self.delay_penalty * (1.0 if is_detour else 0.0)
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

    # --------------------------------------------------------------- evaluation
    def rollout_paths(self, act_fn: Callable[[np.ndarray, np.ndarray], int],
                      rates) -> Dict[int, List[int]]:
        """Deterministically trace every flow with policy act_fn(obs, mask)->action.
        Returns {pair_index: node_path}. Feeds ns-3 (unchanged eval)."""
        self.set_matrix(rates)
        obs, mask, _ = self.reset()
        done = False
        info = {}
        while not done:
            a = act_fn(obs, mask)
            obs, mask, _, _, done, info = self.step(a)
        return info["paths"]

    def greedy_max_util(self, rates) -> float:
        """Myopic best-response: each hop pick the valid neighbour with the lowest
        resulting bottleneck. Sanity target between OSPF and the trained policy."""
        def act(obs, mask):
            base = 2 * self.n_nodes
            res = obs[base + 1::self.NF][:self.max_deg].copy()
            res[mask[:self.max_deg] == 0] = 1e9
            return int(np.argmin(res))
        self.rollout_paths(act, rates)
        return self.cur_max


if __name__ == "__main__":
    from marl_routing.traffic import generate_matrix
    for topo, load in [("abilene", 3.0), ("geant", 1.5)]:
        t = load_topology(topo); n = t.n_nodes
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        mats = [np.array([generate_matrix(topo, load, seed=s)[a, b] for a, b in pairs])
                for s in range(3)]
        env = MultiAgentRoutingEnv(topo, pairs, mats, delay_penalty=0.0, stretch=1)
        print(f"\n=== {topo} (load {load}) ===")
        print(f"[env] {n} node-agents, {env.n_arcs} arcs, max_deg={env.max_deg}, "
              f"obs_dim={env.obs_dim}, gstate_dim={env.gstate_dim}, act_dim={env.act_dim}")
        rates = mats[0]
        paths = env.rollout_paths(lambda o, m: 0, rates)  # 'first valid' policy
        bad = 0
        for i, (s, d) in enumerate(pairs):
            if rates[i] <= 0:
                continue
            p = paths[i]
            if p[0] != s or p[-1] != d or len(set(p)) != len(p):
                bad += 1
            for h in range(len(p) - 1):
                assert (p[h], p[h + 1]) in env.arc_index
        print(f"[validate] {len(paths)} flows, loop-free & valid: "
              f"{'OK' if bad == 0 else f'{bad} BAD'}")
        ospf = env.ospf_max_util(rates)
        greedy = env.greedy_max_util(rates)
        print(f"[max-util]  OSPF {ospf:.1f}%   greedy(stretch=1) {greedy:.1f}%"
              f"   -> headroom {ospf - greedy:+.1f} pts")
