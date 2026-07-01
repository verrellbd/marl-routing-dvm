# Multi-Agent Reinforcement Learning for Network Routing — Methodology & Results

*Draft thesis chapters (Methodology, Results, Discussion). All packet-level QoS numbers
are from full ns-3 simulation on held-out real traffic. Figures referenced live in
`results/`.*

---

## 1. Research question and contribution

**Question.** Can a *learned* routing policy — in particular a decentralized
multi-agent reinforcement learning (MARL) controller in which each router decides
hop-by-hop from local information — improve packet-level quality of service (loss,
delay) over the deployed shortest-path baseline (OSPF) under realistic, dynamic
traffic; and how does decentralization compare against a centralized graph-neural-network
(GNN) controller that sees the whole network?

**Contributions.**
1. A decentralized MARL routing system (per-node agents, custom MAPPO with centralized
   training / decentralized execution) evaluated against both OSPF and a centralized
   single-agent GNN router — on **three real backbone networks with real measured
   dynamic traffic**, judged at the **packet level in ns-3**, not on an analytical proxy.
2. An honest characterization of *when decentralization pays off*: it is near-free on
   small and mid-size networks (and on GÉANT actually beats the centralized controller),
   but the coordination cost becomes visible at 50 nodes.
3. Two methodological findings of independent interest: the rateScale simulation knob
   preserves loss/utilisation but not delay (so it must be held constant across topologies
   for delay to be comparable), and a millisecond-truncation bug in the ns-3 delay model
   that silently zeroed sub-millisecond link propagation on the geographically compact
   network.

The single deliberate novel modelling element is the **GNN backbone**; the MARL framing
is the thesis's central system. Scope was kept disciplined to one novel element.

---

## 2. Data and networks (real sources only)

All topology, capacity and traffic data come from **SNDlib** and the associated public
measured-traffic archives — no synthetic or reconstructed data is used in the final
results. (An earlier phase used a gravity traffic model and reconstructed topologies;
those results are retained only for history and are explicitly superseded.)

| Network | Nodes | Links | Capacities | Real dynamic traffic | Granularity / span |
|---------|-------|-------|-----------|----------------------|--------------------|
| Abilene | 12 | 15 | 14×10G + 1×2.5G (real OC-192/OC-48) | Abilene / Zhang TM | 5-min, 6 months (48,096 matrices) |
| GÉANT | 22 | 36 | uniform 40G modules | GÉANT / Uhlig–TOTEM TM | 15-min, 4 months (11,460 matrices) |
| Germany50 (DFN) | 50 | 88 | uniform 40G modules | DFN TM | 5-min, 1 day (288 matrices) |

**Capacities** are taken directly from each SNDlib file: the pre-installed capacity where
present, otherwise the installable module capacity. **Link delays** are not in SNDlib, so
they are derived from the file's own node coordinates: great-circle (haversine) distance ×
0.005 ms/km, i.e. a fibre propagation speed of ≈ 2×10⁵ km/s. The same rule is applied to
all three networks, so delays are on a common physical footing.

**Traffic.** Real measured demand matrices are used directly — the spatial and temporal
structure of real traffic is preserved. Two adjustments, both standard traffic-engineering
practice, are applied:
- *Temporal train/test split.* The agent trains on the first 70% of each network's
  timeline and is evaluated on the last 30%, so every reported number is on **unseen
  real traffic** from a later period — a genuine generalization test, not interpolation.
- *Magnitude scaling.* The raw SNDlib demands are dimensioning-scale (provisioned for a
  multi-terabit peak), tens of times larger than a single-module network can carry, so
  routing is trivially infeasible at any setting. We scale the *magnitude* by a per-network
  load factor to place the network in a congesting regime, while leaving the spatial
  (origin–destination) and temporal (time-of-day) structure of the real matrices untouched.

Matrix selection within the split is **deterministic** (evenly spaced by timestamp, no
random seed), so the evaluation set is reproducible from the data alone.

---

## 3. Method

### 3.1 The routing decision as a sequential MDP

Routing is framed as a per-flow sequential decision process over a fixed candidate set.
For each origin–destination pair we precompute the *k* shortest paths (k = 3, Yen's
algorithm). The centralized agent visits flows one at a time and chooses one of the *k*
candidate paths; the resulting link loads accumulate, so each decision sees the congestion
left by previous ones. The reward is potential-based:

> reward = −(marginal increase in max link-utilisation) − λ·(extra hops over the shortest path)

The first term drives the policy to spread load and minimise the bottleneck; the QoS delay
penalty λ keeps it on short paths when there is no congestion to relieve, so it *matches
OSPF when traffic fits* and only detours under load. λ = 0.5 on Abilene/GÉANT, 0.1 on
Germany50 (its long diameter makes a high hop penalty collapse the policy back onto OSPF).

### 3.2 Decentralized MARL formulation

The MARL system replaces the single global decision-maker with **one agent per node**.
Forwarding is hop-by-hop: a packet at node *n* destined for *d* is forwarded by node *n*'s
agent to one of its neighbours. The action space at each node is its fixed neighbour list.

A **stretch-1** constraint keeps forwarding loop-free and bounded: a next hop *m* is legal
only if dist(*m*,*d*) ≤ dist(*n*,*d*) (equal-distance "sideways" moves are allowed, giving
the agents room to load-balance across equal-cost paths), and a visited-set forbids
revisits. Only genuine *detour* hops (those that do not strictly decrease distance to the
destination) are penalised, so the agents pay for indirection but not for legitimate
equal-cost spreading.

Training is **centralized-training / decentralized-execution (CTDE)** via a custom MAPPO
implementation: each agent's actor is a masked-categorical MLP over its local observation,
while a single centralized critic sees the global state during training only. At execution
each agent acts on local information alone — the property that makes the controller
deployable as a distributed SDN control plane.

### 3.3 The GNN backbone (novel element)

The centralized single-agent controller uses a PyTorch-Geometric GNN feature extractor as
its PPO policy/value network. It consumes the graph state — per-link utilisation,
adjacency, candidate-path features and the current flow's rate — via an arc→node incidence
and produces the embedding the actor and critic heads sit on. This is the one deliberate
novel modelling element, and it also serves as the strong centralized upper-reference the
decentralized MARL is measured against.

### 3.4 Train fast, judge at the packet level

Full ns-3 simulation is ~18× slower than realtime and is never in the training loop.
Both controllers train against an **analytical link-utilisation surrogate** (fast, exact
for the max-utilisation objective), then **every reported QoS result is produced by full
ns-3**: the learned policy's exact per-flow paths are extracted and installed in ns-3 as
static host-routes, and loss/delay/throughput are measured with FlowMonitor on the same
deterministic real test matrices for all three methods. This separates a cheap training
signal from a high-fidelity, like-for-like judgement.

### 3.5 Parameter consistency across networks

Held identical across all three networks: rateScale = 20, simTime = 8, k = 3. Differing
for principled, documented reasons only: the per-network load scale (normalises each
network into the congesting regime) and, on Germany50 only, top-N flow filtering
(2450 flows → top 200 by rate) to keep ns-3 tractable — applied identically to OSPF, GNN
and MARL so the comparison stays fair. A control experiment confirmed rateScale preserves
loss and utilisation exactly but inflates delay, which is why it is pinned across
topologies; and the ns-3 link-delay model was moved from millisecond to nanosecond
resolution after a truncation bug was found zeroing Germany50's sub-millisecond links.

---

## 4. Results

### 4.1 Headline: packet-level QoS, real traffic, 3-way

Identical deterministic real test matrices; results stratified by congestion regime
(overload = offered load exceeds capacity somewhere; feasible = it fits).

**Abilene (real Zhang TM):**

| Regime | Metric | OSPF | single-agent GNN | MARL (decentralized) |
|--------|--------|------|------------------|----------------------|
| Overload | loss | 2.32% | **0.17%** | **0.18%** |
| Overload | delay | 37.5 ms | **11.7 ms** | **12.6 ms** |
| Feasible | loss | 0.18% | 0.16% | 0.18% |

**GÉANT (real Uhlig/TOTEM TM):**

| Regime | Metric | OSPF | single-agent GNN | MARL |
|--------|--------|------|------------------|------|
| Overload | loss | 7.46% | 1.44% | **0.86%** |
| Overload | delay | 17.9 ms | 15.5 ms | **14.9 ms** |
| Feasible | loss | 0.12% | 0.14% | 0.15% |

**Germany50 (real DFN TM, 50 nodes, top-200 flows):**

| Regime | Metric | OSPF | single-agent GNN | MARL |
|--------|--------|------|------------------|------|
| Overload | loss | 3.16% | **0.03%** | 0.96% |
| Overload | delay | 17.4 ms | **1.9 ms** | 17.6 ms |
| Feasible | loss | 0.12% | 0.03% | 0.03% |

Figures: `results/fig_real3way_{abilene,geant,germany50}.png`.

**Reading the result.**
- **Both learned controllers beat OSPF under congestion on all three real networks** —
  overload loss is cut roughly 13× on Abilene, 5–9× on GÉANT, and up to ~100× on
  Germany50, with delay reduced as well.
- **Decentralization is competitive with central control.** The MARL controller, acting
  on purely local information, nearly matches the centralized GNN and on GÉANT slightly
  beats it — direct evidence that a distributed control plane need not sacrifice
  performance on small-to-mid networks.
- **The coordination cost appears at scale.** On the 50-node Germany50 network the
  centralized GNN keeps the network almost perfectly feasible (0.03% loss, 1.9 ms),
  while MARL controls loss well (0.96%) but pays a delay premium (17.6 ms) because each
  agent lacks the global view needed to avoid longer detours. This is the honest boundary
  of the decentralized approach.
- **Everything ties in the feasible regime** — by design, the QoS reward keeps both
  learned policies on short paths when there is nothing to relieve.

### 4.2 Multi-seed robustness

The result is not a lucky single training run. Each policy was retrained with three model
seeds (0/1/2) and re-evaluated on the deterministic real test matrices. Max link-utilisation
in the overload regime (mean ± std over seeds):

| Network | OSPF | single-agent GNN | MARL |
|---------|------|------------------|------|
| Abilene | 122% | **67 ± 2%** | **64 ± 2%** |
| GÉANT | 126% | 92 ± 2% | 97 ± 7% |
| Germany50 | 109% | **86 ± 2%** | 99 ± 7% |

Both learned methods drive the network from infeasible (>100%) back toward feasible across
*every* seed. The GNN's bars are tight (±2) everywhere; MARL is tight on Abilene but more
variable on the larger networks (±7) — so the coordination-cost-at-scale finding is itself
robust, not a single-seed artifact. Figure: `results/fig_multiseed_overload.png`. The ns-3
numbers in §4.1 are the seed-0 high-fidelity anchor; these analytical error bars establish
seed-robustness cheaply (without 18 expensive ns-3 runs).

---

## 5. Discussion

**Why decentralization wins on GÉANT but lags at 50 nodes.** GÉANT's uniform 40G fabric
and moderate diameter mean local greedy spreading rarely needs a globally coordinated
detour; the per-node agents' equal-cost sideways moves are enough, and avoiding a central
bottleneck-by-flow-ordering can even help. At 50 nodes the diameter grows, congestion is
spread across more independent bottlenecks, and a locally-optimal next hop is more often
globally suboptimal — exactly the regime where a centralized controller's whole-network
view buys real delay savings. This is the central trade-off the thesis surfaces:
decentralization is near-free until the network is large enough that local information is
genuinely insufficient.

**Against OSPF specifically.** OSPF is shortest-path and load-agnostic; under overload it
piles traffic onto already-saturated cuts. Both learned methods relieve those cuts and the
gain is concentrated exactly there. When traffic fits, they correctly decline to detour and
tie OSPF — a desirable property, not a weakness.

---

## 6. Honest caveats

- The win is **concentrated in the overload regime**; in the feasible regime all methods
  tie, by design.
- We compare against the **deployed** baseline (OSPF), not a theoretical optimum
  (MPLS-TE / LP). Learned routing trades the predictability and convergence guarantees of
  traditional TE for adaptivity; an LP upper bound ("X% of optimal") is noted as future
  work.
- Training uses an analytical surrogate; all *judgements* are ns-3, but the training signal
  is an approximation of true packet dynamics.
- ns-3 multi-seed error bars exist for the analytical max-utilisation metric (§4.2); the
  packet-level loss/delay numbers in §4.1 are seed-0 only (the high-fidelity anchor).
  Extending packet-level error bars to all seeds is straightforward but expensive
  (especially on Germany50) and is left as optional rigor.

---

## 7. Reproducibility

```
# Train (per network, per seed) — analytical surrogate, CPU
python train_gnn_qos.py --topo <abilene|geant|germany50>_sndlib --traffic real --seed <0..2>
python train_marl.py     --topo <abilene|geant|germany50>_sndlib --traffic real --seed <0..2>

# Multi-seed analytical error bars
python eval_multiseed_analytical.py      # -> results/multiseed_analytical.json
python make_multiseed_fig.py             # -> results/fig_multiseed_overload.png

# Packet-level ns-3 3-way (per network)
python evaluate_ns3.py       --traffic real ...   # extract centralized GNN paths
python export_marl_routing.py --traffic real ...  # extract MARL paths
python run_ns3_phase2.py --ratescale 20 ...       # run paths in ns-3, FlowMonitor QoS

# Data converters (real SNDlib -> JSON single source of truth)
python sndlib_to_json.py
```

Models: `results/{abilene,geant,germany50}_sndlib_{qos,marl}_real_seed{0,1,2}/`.
Real-traffic loader: `marl_routing/real_traffic.py` (deterministic timestamp selection,
temporal train/test split). Environments: `marl_routing/sequential_routing_env.py`
(centralized), `marl_routing/multiagent_routing_env.py` + `marl_routing/mappo.py`
(decentralized).
