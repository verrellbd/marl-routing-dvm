# Plan — Adding genuine Multi-Agent RL (Option B)

*Goal: replace the single-agent GNN router with a true multi-agent system, so the
thesis title ("MARL for Network Routing") matches the implementation — without
throwing away the validated pipeline (analytical training + ns-3 judging) or the
existing single-agent result (which becomes a baseline).*

---

## 0. What we keep vs what changes

**Keep unchanged (the whole evaluation half of the pipeline):**
- `marl_routing/topology.py`, `traffic.py` — topologies + gravity traffic.
- `marl_routing/gnn_routing_agent.py::compute_ksp` — candidate paths.
- `evaluate_ns3.py`, `run_ns3_phase2.py`, `abilene-validate.cc` — ns-3 judging.
  *The agents still produce a per-flow path; ns-3 still installs and measures it.*
- The analytical link-utilisation model (fast training surrogate).
- `train_gnn_qos.py` single-agent GNN → **becomes a baseline** (OSPF vs single-agent
  GNN vs MARL).

**Changes (the decision half):**
- New multi-agent environment (per-node agents, hop-by-hop forwarding).
- New MARL trainer (MAPPO: centralized training, decentralized execution).
- Path extraction = roll out the agents' next-hop decisions src→dst.

---

## 1. Formulation (recommended default — per-node router agents)

The canonical MARL-for-routing formulation (Q-routing lineage):

- **Agents = nodes/routers.** Abilene → 12 agents, GÉANT → 23 agents.
- **Decentralized execution:** each agent sees only LOCAL state (its incident link
  utilisations, 1-hop neighbour headroom, the packet's destination) and chooses the
  **next hop** among its neighbours. This partial observability is what makes it
  genuinely multi-agent (vs the current global single brain).
- **Cooperative / team objective:** shared reward = −(max link utilisation)
  − delay_penalty·(path length) — same objective as now, so results are comparable.
- **Loop avoidance (critical):** mask each agent's action set to the
  **destination-oriented DAG** (next hops that lie on the union of k-shortest paths
  toward the destination). Guarantees loop-free, finite paths. Fallback: TTL +
  revisit penalty.
- **Parameter sharing:** one shared GNN policy used by all agents (scales to any node
  count, standard in cooperative MARL). Agents differ only by their local observation.

**Why per-node (not per-flow):** per-node is the textbook "multiple routers each
making local decisions" picture — decentralized execution, partial observability,
non-stationarity — i.e. real MARL. Per-flow path-pickers would be closer to the
current single-agent setup and weaker as a "multi-agent" claim.

## 2. Algorithm — MAPPO (CTDE)

- **Actor (decentralized):** shared GNN; input = local observation; output = masked
  next-hop distribution. Reuse the `SeqGNNExtractor` message-passing core.
- **Critic (centralized, training only):** sees the GLOBAL state (all link utils) to
  give a low-variance value estimate → solves multi-agent credit assignment. Discarded
  at execution time (decentralized).
- **Why MAPPO:** the cooperative-team, parameter-shared, centralized-critic standard;
  stable, well-documented, and on-policy like our current PPO so the jump is small.

**Infra default:** a **custom lightweight MAPPO** over a PettingZoo-style parallel env,
reusing our analytical model + GNN. Rationale: full control, no RLlib new-API-stack
churn (Ray 2.51.2 is installed but its multi-agent API is volatile). RLlib remains a
fallback if the custom loop underperforms.

## 3. Phased work plan

| Phase | Deliverable | Est. |
|-------|-------------|------|
| **M1** | `marl_routing/multiagent_routing_env.py` — PettingZoo ParallelEnv: N node-agents, hop-by-hop forwarding on the analytical model, DAG action masking, team reward, `rollout_paths()` for eval. + unit test (loop-free, reaches dst). | 1–2 d |
| **M2** | `marl_routing/mappo.py` — shared GNN actor + centralized critic, GAE, clipped PPO, action masking. Smoke-train on a toy matrix. | 2–3 d |
| **M3** | `train_marl.py` — mixed-load training on Abilene (congested regime). Sanity: MARL ≥ OSPF analytically; compare to single-agent GNN. | 1 d |
| **M4** | ns-3 validation: extract per-flow paths from the MARL rollout → existing `evaluate_ns3.py`/`run_ns3_phase2.py` UNCHANGED. Multi-seed, stratified. | 0.5 d |
| **M5** | GÉANT repeat + analysis: OSPF vs single-agent GNN vs MARL on both topologies. Figures + honest write-up (incl. decentralization cost). | 1 d |

(~6–8 working days end-to-end.)

## 4. Key risks & mitigations
- **Loops / non-termination** → DAG action masking (primary), TTL + revisit penalty (backup).
- **Non-stationarity** (all agents learn at once) → CTDE centralized critic + parameter
  sharing + on-policy MAPPO.
- **MARL underperforms single-agent GNN** → that is itself an honest, expected finding
  (decentralization has a coordination cost); report it. Centralized critic + shaped
  reward narrow the gap.
- **Scalability to GÉANT (23 agents)** → parameter sharing makes node-count irrelevant
  to the policy size.

## 5. Honest expectation
Decentralized MARL usually **does not beat** a centralized single-agent optimum — it
trades global optimality for locality/scalability/robustness. The likely, defensible
result: **MARL approaches OSPF/single-agent-GNN with only local information**, which is
the real selling point of multi-agent routing (no central controller needed). We will
report the coordination cost honestly rather than claim MARL is strictly best.

## 6. Decisions (LOCKED 2026-06-17)
1. Agent granularity: **per-node router agents** ✅ (each node picks next hop from local obs).
2. Infra: **custom MAPPO** ✅ (shared GNN actor + centralized critic; no RLlib).
3. Single-agent GNN kept as an explicit baseline (OSPF vs single-agent GNN vs MARL) ✅.

## 7. Implementation notes (per-node next-hop, locked design)
- **Decision = (node n, destination d, flow rate r) → choose next hop.** A flow's path
  is traced hop-by-hop: at each node the agent picks a next hop toward d; load
  accumulates per committed arc.
- **Loop-free guarantee:** valid next hops at n for d = neighbours m with
  `dist(m,d) < dist(n,d)` (the shortest-path DAG). Strict distance decrease ⇒ finite,
  loop-free paths ≤ diameter, and multiple choices where the topology is rich (Abilene)
  → that's the agent's load-balancing room. (If MARL underperforms, widen to a
  k-shortest DAG + visited-set in M3.)
- **Actor obs (LOCAL, decentralized):** current-node id, destination id, incident-arc
  utilisations, per-candidate features [arc util, resulting bottleneck, headroom,
  neighbour dist-to-dst], flow rate. Fixed-size action space = max DAG out-degree, with
  an **action mask** zeroing invalid candidates.
- **Critic state (GLOBAL, training only):** all link utilisations + node/dst/rate.
- **Reward (team):** −(marginal global max-util increase) − delay_penalty per hop;
  telescopes to −final max-util (same objective as OSPF/single-agent GNN, comparable).
- **Eval:** `rollout_paths()` traces each flow deterministically → per-flow path →
  feeds the UNCHANGED `evaluate_ns3.py` / ns-3 scenario.
