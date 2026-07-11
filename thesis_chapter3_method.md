# Chapter 3 — Method

## 3.1 Design philosophy and principles

The design of this project is governed by a small number of guiding beliefs, from which the
concrete engineering choices follow.

**Compare against what is deployed, not against a theoretical ceiling.** The research
question is whether a learned — and in particular a *decentralized* — routing policy can
improve on routing as it is actually run in operational networks. The primary baseline is
therefore OSPF shortest-path routing, the dominant deployed intra-domain protocol, together
with a centralized learned controller as an upper reference. A linear-programming optimum is
deliberately excluded: it answers a different question (how far from theoretically optimal?)
than the one posed here (does the method beat current practice?).

**One novel element, disciplined scope.** The single deliberate modelling novelty is the use
of a graph neural network (GNN) over the router-level topology as the policy backbone; the
central system contribution is the decentralized multi-agent formulation. Every other
component is kept deliberately standard so that any observed effect can be attributed to
these two choices rather than to incidental design.

**Decentralization is the objective, not a by-product.** The target is a policy that each
router can execute using only local information — a deployable distributed control plane.
The architecture commits to local execution throughout, even where a global controller would
be simpler, because the reliability and competitiveness of *local* decision-making is
precisely what the work sets out to measure.

**Honesty and reproducibility over headline numbers.** All results rest on real network and
traffic data, are judged at the packet level rather than on an analytical proxy, and are
reported with multi-seed variance even where that variance is unflattering. This is treated
as a design constraint, not an afterthought.

These beliefs translate into the operating principles used throughout:

- **Real-data provenance.** Topologies, link capacities, and traffic demands are taken from
  the SNDlib repository; no synthetic topologies or traffic models are used in the final
  results.
- **Centralized training, decentralized execution (CTDE).** Learning uses a centralized
  critic with access to global state, while each agent acts on local observations only;
  agents share network parameters and pool their experience.
- **Train fast, judge at high fidelity.** Policies are trained against a fast analytical
  link-utilisation surrogate, and every reported result is produced by full ns-3
  packet-level simulation.
- **QoS-aware, potential-based reward.** The reward minimises the network bottleneck while
  penalising only unnecessary detours, so a learned policy reduces to short-path routing
  when the network is uncongested.
- **Temporal generalisation.** Models are trained on earlier real traffic and evaluated on
  later, unseen real traffic, with deterministic (seed-free) selection of matrices.
- **Loop-free by construction.** Path validity is guaranteed by structural constraints
  (bounded stretch and a visited-set), not learned.

## 3.2 System

The system is organised as a pipeline of independent components sharing a single source of
truth for the network and its traffic.

**Topology layer.** Each network is loaded from its SNDlib description into a directed graph
whose nodes are physical routers (points of presence) and whose edges are physical links
carrying a capacity and a propagation delay. Capacities are read directly from the SNDlib
files; propagation delays, absent from SNDlib, are derived from the node coordinates as
great-circle distance under a fixed fibre speed. The same graph is consumed by both the
learning environment and the ns-3 simulator, so the model and the evaluator never disagree
about the network. Three networks of increasing size are used: Abilene (12 nodes), GÉANT
(22 nodes), and Germany50 (50 nodes).

**Traffic layer.** Traffic is drawn from real measured dynamic demand-matrix time series
(Abilene/Zhang, GÉANT/Uhlig–TOTEM, Germany50/DFN). The timeline is split temporally — the
earlier portion for training, the later portion for testing — so that evaluation is always
on unseen, later traffic. The real spatial and temporal structure of the demands is left
untouched; only the overall magnitude is scaled, by a per-network load factor, to place the
network in a congesting regime.

**Environments.** Two Markov decision processes are implemented over this topology and
traffic. The *single-agent* environment frames routing as sequential path selection: flows
are routed one at a time, and at each step a single controller chooses one of the *k*
shortest candidate paths for the current flow, observing the congestion left by earlier
choices. The *multi-agent* environment frames routing as hop-by-hop forwarding: each router
is an agent that forwards a packet to one of its neighbours, subject to a bounded-stretch,
loop-free constraint. The two environments instantiate the centralized and decentralized
ends of the design space respectively.

**Reward.** Both environments share a QoS-aware, potential-based reward: the negative
marginal increase in maximum link utilisation, minus a penalty on detour (non-shortest-path)
hops. The first term drives the policy to relieve the bottleneck; the second keeps it on
short paths when there is no congestion to relieve, so the learned behaviour coincides with
OSPF in the uncongested regime and departs from it only under load.

**Model.** The policy backbone is a GNN feature extractor over the router-level topology. It
aggregates per-link utilisation onto nodes through the link–node incidence structure, mixes
each node's state with that of its immediate neighbours through the adjacency, and passes the
result through fully-connected layers to the actor and critic heads. The extractor is
intentionally shallow — a single neighbourhood-mixing step — which both keeps the parameter
count modest and aligns with the decentralized setting, in which a router naturally reasons
from local and one-hop-neighbour information. Because the graph operations are shared across
all nodes, the same architecture applies unchanged across networks of different size.

**Learning algorithms and evaluation harness.** The single-agent GNN policy is trained with
Proximal Policy Optimisation (PPO); the multi-agent policy is trained with a custom
multi-agent PPO (MAPPO) implementing CTDE. At evaluation, the trained policy's exact per-flow
paths are exported and installed in ns-3 as static host routes, and packet-level loss, delay,
and throughput are measured with the ns-3 FlowMonitor. Training therefore never runs the
expensive simulator, while every reported result does.

## 3.3 Training and evaluation

**Training.** Each policy is trained on the analytical surrogate over a distribution of real
training-split matrices spanning several load levels, so that a single policy generalises
across congestion conditions rather than overfitting one operating point. PPO and MAPPO use
standard hyperparameters (clipped objective, generalised advantage estimation, a small
entropy bonus); the multi-agent critic consumes global state while its actors consume only
local observations. Randomness in training — weight initialisation, exploratory action
sampling, matrix ordering, and minibatch shuffling — is controlled by a single training seed.
Each configuration is trained with three independent seeds; the seed is retained precisely so
that the *variance* of the outcome can be measured, since the reliability of a learned policy
is itself a result. Training runs are logged and checkpointed to disk.

**Evaluation.** All quality-of-service results are produced by full ns-3 simulation on the
held-out test-split matrices, identical across the three methods, so that the comparison is
like-for-like. Reported metrics are packet loss, mean delay, and throughput, together with
maximum link utilisation from the analytical model. Results are **stratified by congestion
regime** — overload (offered load exceeds capacity somewhere) versus feasible — rather than
averaged blindly, because the methods are expected to differ only under congestion and to
coincide when traffic fits. Finally, results are reported as **mean and standard deviation
across the three training seeds**, so that both the central tendency and the reliability of
each method are visible. Simulator parameters (rate scaling, simulation time, candidate-path
count) are held constant across networks; the only quantities that vary per network are the
load scale, which normalises congestion, and, on the largest network, a top-*N* flow filter
applied identically to all methods for tractability.
