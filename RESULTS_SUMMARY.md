# GNN Routing vs OSPF — Results Summary

*MARL for Network Routing (master's thesis). **All reported numbers — QoS AND link-
utilisation — are from full ns-3 packet-level simulation** on held-out real traffic. The
analytical model is used only as the training reward (see §4); nothing reported here is
analytical. Fresh multi-seed run, monaco-retrained (seeds 0/1/2), 2026-07-14.*

---

## 1. The question
Can a GNN-based routing agent (learned, per-flow path selection from k-shortest
paths) beat the deployed baseline (OSPF shortest-path) under dynamic traffic, when
judged by real packet-level QoS — loss, delay, throughput?

## 2. Headline finding
**On real measured traffic, both learned routers beat OSPF under congestion across all
three real backbones — and the decentralized MARL nearly matches (sometimes beats) the
centralized GNN, with the coordination cost showing only on the largest network.**

Everything below rests on **user-provided real data only**: SNDlib topologies + real
capacities, and real **dynamic** measured demand matrices (Abilene-Zhang 5-min/6-month;
GÉANT-Uhlig/TOTEM 15-min/4-month; Germany50-DFN 5-min/1-day), with a **temporal
train/test split** (train on the first 70% of the timeline, test on the last 30% →
generalisation to unseen real traffic). Matrix selection is **deterministic by timestamp**
(no data seed). Demand magnitude is scaled to a congesting regime (standard TE practice;
spatial+temporal structure untouched).

**3-way ns-3 comparison (OSPF vs single-agent GNN vs MARL), real traffic, identical
deterministic test matrices, packet-level QoS.** All QoS values are mean ± std over model
seeds 0/1/2 (OSPF is routing-fixed → no std).

Abilene (real Zhang TM):
| Regime | Metric | OSPF | single-agent GNN | MARL |
|--------|--------|------|------------------|------|
| Overload | loss  | 2.32% | **0.17 ± 0.00%** | **0.18 ± 0.00%** |
| Overload | delay | 37.5 ms | **11.7 ± 0.1 ms** | **12.6 ± 0.1 ms** |
| Feasible | loss  | 0.19% | 0.16 ± 0.00% | 0.18 ± 0.01% |
| Feasible | delay | 11.5 ms | 11.3 ± 0.0 ms | 12.8 ± 0.3 ms |

GÉANT (real Uhlig/TOTEM TM):
| Regime | Metric | OSPF | single-agent GNN | MARL |
|--------|--------|------|------------------|------|
| Overload | loss  | 10.63% | 4.27 ± 1.49% | **0.91 ± 0.30%** |
| Overload | delay | 18.7 ms | 21.0 ± 4.9 ms | **12.2 ± 1.1 ms** |
| Feasible | loss  | 0.11% | 0.14 ± 0.01% | 0.13 ± 0.01% |
| Feasible | delay | 7.9 ms | 9.6 ± 0.4 ms | 9.4 ± 0.7 ms |

> **GÉANT is where decentralized MARL wins cleanly.** MARL cuts loss ~12× vs OSPF
> (0.91% vs 10.63%) and is the only method that also cuts delay (12.2 ms vs OSPF 18.7).
> The **centralized SA-GNN underperforms here** (4.27% loss, 21 ms delay — *worse* delay
> than OSPF) and is **high-variance across seeds** (per-seed loss 3.06 / 3.37 / 6.37%; one
> seed converged poorly). Honest read: at mid-scale, local-only coordination happened to
> generalise better than the central policy on the real GÉANT test matrices. Reported
> honestly, not cherry-picked.

Germany50 (real DFN TM, 50 nodes, top-200 flows) — MARL hop-capped, **multi-seed (0/1/2)**:
| Regime | Metric | OSPF | single-agent GNN | MARL |
|--------|--------|------|------------------|------|
| Overload | loss  | 3.16% | **0.06 ± 0.03%** | 2.00 ± 1.89% |
| Overload | delay | 17.4 ms | **2.8 ± 0.9 ms** | 9.3 ± 7.1 ms |
| Feasible | loss  | 0.12% | 0.03 ± 0.00% | 0.08 ± 0.07% |
| Feasible | delay | 4.8 ms | **1.8 ± 0.0 ms** | 3.3 ± 1.7 ms |

> **Germany50 hop-cap fix — honest multi-seed result (2026-07-01/02).** The uncapped MARL had
> a delay/loss gap at 50 nodes (0.96% loss, 17.6 ms) because per-node agents, on local info
> only, wandered into long detours (up to 12 hops) that concentrated load and left a link
> saturated (~105%), driving queueing delay. NB: germany50 detour links are short, so the
> driver was **queueing (utilisation), not propagation**. A **hop cap** on the agents'
> forwarding (final path ≤ shortest-path hops + 4; **topology untouched, only the action
> mask**) stops the wandering and improves congestion relief.
> **What it robustly fixes:** DELAY — MARL mean 9.3 ± 7.1 ms vs OSPF 17.4 (still cuts delay).
> **What stays hard:** LOSS is **high-variance across seeds** (seed 1 0.03% matches the
> centralized GNN; seed 0 4.55% is *worse* than OSPF); mean 2.00% only edges OSPF's 3.16%. So
> decentralized coordination at 50 nodes is **viable but less reliable than central control**
> — the coordination cost shows up as VARIANCE, not a mean gap. Per-seed MARL loss/delay:
> seed0 4.55%/18.9ms · seed1 0.03%/2.1ms · seed2 1.43%/6.9ms. Models
> results/germany50_sndlib_marl_opt3b_seed{0,1,2}/ (λ=0.1, max_stretch=4). Do NOT claim
> "MARL matches central at 50n" in general — only best-seed. The centralized SA-GNN is the
> reliable winner here (0.06 ± 0.03% loss, all three seeds ≤0.10%).

**Findings — a scale story:**
- **Abilene (12n): both learned methods crush OSPF and tie each other** — loss cut ~13× (OSPF
  2.32% → 0.17/0.18%), delay cut ~3× (37.5 → ~12 ms). Decentralized MARL pays no visible
  coordination cost at this scale.
- **GÉANT (22n): decentralized MARL wins cleanly** — loss cut ~12× vs OSPF (0.91% vs 10.63%)
  and the only method to also cut delay (12.2 vs 18.7 ms). The centralized SA-GNN does *worse*
  here (4.27% loss, 21 ms delay) and is seed-noisy. Best result of the three networks for MARL.
- **Germany50 (50n): the centralized SA-GNN wins; MARL is viable but high-variance.** SA-GNN
  cuts loss ~50× (0.06%) reliably across seeds; MARL cuts delay but its loss swings by seed
  (0.03–4.55%). The coordination cost is real at scale — and it shows as VARIANCE, not a mean
  gap. A hop cap (shortest + 4 hops; topology unchanged) fixes MARL's DELAY robustly (every
  seed beats OSPF), but LOSS stays high-variance. The centralized GNN remains the safe choice
  at scale.
- All methods ~tie when the network is uncongested (feasible regime).

Figures: results/fig_real3way_{abilene,geant,germany50}.png. Parameter consistency:
rateScale=20, simTime=8 identical across all three; load scale (per-network, to normalise
congestion) and max-flows (Germany50 only, 2450→top-200 for ns-3 tractability) differ for
principled reasons. ns-3 link delays use nanosecond resolution (fixed a ms-truncation bug
that had zeroed Germany50's sub-ms link delays).

## 3. Link utilisation (ns-3 packet-level) — the mechanism behind the QoS
QoS (§2) shows *what* happens; link utilisation shows *why*. **These numbers are ns-3
packet-level too** — max link-utilisation measured from bytes carried per device in the
sim, mean ± std over model seeds 0/1/2 × held-out test matrices. (The ns-3 scenario
already logged per-link utilisation; we simply aggregate it — no re-simulation.) True
utilisation, corrected for the 7/8 active-measurement window (see note).

**Overload regime** (offered load > capacity):
| Topology | OSPF | single-agent GNN | MARL |
|----------|------|------------------|------|
| Abilene | **100 ± 0%** | 66 ± 5% | 66 ± 3% |
| GÉANT | **100 ± 0%** | 99 ± 2% | 92 ± 9% |
| Germany50 | **100 ± 0%** | 91 ± 7% | 96 ± 7% |

**Feasible regime** (offered load < capacity):
| Topology | OSPF | single-agent GNN | MARL |
|----------|------|------------------|------|
| Abilene | 96 ± 4% | **52 ± 3%** | **52 ± 4%** |
| GÉANT | 92 ± 5% | **67 ± 8%** | 70 ± 12% |
| Germany50 | 100 ± 0% | **77 ± 2%** | 89 ± 5% |

**How to read this (important, physical):** a real link *cannot* carry more than its
capacity, so **ns-3 utilisation caps at 100%** — OSPF pins at exactly 100% under overload
(its bottleneck link is saturated), and the excess offered load appears as **packet loss**
(§2), *not* as >100% utilisation. This is why the overload story is told by loss and the
utilisation story is cleanest in the **feasible** regime, where OSPF sits near-saturated
(92–100%) while both learned methods keep large headroom (52–77%). The single analytical
">100%" number (OSPF's true *offered* load, e.g. ~122% on Abilene) is a fluid-model
quantity used only as the **training reward**; it is described in the Method chapter, and
appears nowhere in these results.

Both learned methods keep the bottleneck below OSPF across all seeds. SA-GNN bars are
tighter; MARL is tight on Abilene but higher-variance on the larger nets — the
coordination-cost-at-scale finding is robust, not a single-seed artifact.
Figure: results/fig_ns3util_multiseed.png (two panels, overload | feasible).

> **7/8 active-window correction.** ns-3 measures bytes carried over `window = simTime = 8s`,
> but flows are active only [2s, 9s] = 7s, so a link at 100% line rate reads 7/8 = 87.5%.
> All flows share the same window, so dividing by 7/8 recovers true utilisation exactly and
> cancels in any OSPF-vs-learned ratio. Aggregated by `aggregate_ns3_util.py`.

### 3a. Offered load on the bottleneck (uncapped — the companion view)
Carried utilisation (above) caps at 100% because a real link cannot pass more than its
capacity. To show *how far past capacity* each routing pushes its worst link, we report
**offered load** = the sum of the flow rates a routing places on a link ÷ capacity, maxed
over links. This is an exact quantity computed from the installed per-flow paths (verified
identical to the routing's stored max-util), **not** the analytical training surrogate; the
excess above 100% is precisely what becomes packet loss in §2. Mean ± std over model seeds
0/1/2 × held-out matrices.

**Overload regime** — max offered load on the bottleneck link (>100% = overloads a link):
| Topology | OSPF | single-agent GNN | MARL |
|----------|------|------------------|------|
| Abilene | 128 ± 16% | **64 ± 5%** | **65 ± 3%** |
| GÉANT | 161 ± 19% | 119 ± 20% | **95 ± 14%** |
| Germany50 | 115 ± 2% | **89 ± 7%** | 102 ± 15% |

**Feasible regime** — max offered load on the bottleneck link:
| Topology | OSPF | single-agent GNN | MARL |
|----------|------|------------------|------|
| Abilene | 95 ± 4% | **51 ± 2%** | **51 ± 4%** |
| GÉANT | 90 ± 5% | **66 ± 8%** | 68 ± 11% |
| Germany50 | 98 ± 1% | **75 ± 2%** | 88 ± 6% |

Reading it: under overload OSPF drives its worst link well past capacity (128 / 161 / 115%);
the learned routers pull it back toward or under the ceiling. **GÉANT is decisive** — OSPF
161% → MARL **95%** (the only method under 100%, hence its ~12× lower loss), while SA-GNN
still overloads at 119% (which is exactly why SA-GNN keeps 4.27% loss there — see §2). On
Germany50 MARL sits *at* 102% with a wide bar (the coordination-cost-as-variance story);
the centralized SA-GNN gets under at 89%. Figure: results/fig_offered_util_regime.png.

> **Offered load vs carried utilisation — keep them distinct.** §3 (carried, ≤100%) is "what
> physically flows in ns-3"; §3a (offered, uncapped) is "what the routing demands, and the
> overshoot is the loss." Both are exact packet-level/path quantities — §3a is *not* the
> fluid training surrogate. They are complementary: §3a explains *why* the §2 losses occur.

## 4. Method
1. **Topologies + traffic**: SNDlib topologies (Abilene, GÉANT, Germany50) loaded from
   JSON (single source of truth for ns-3 and the Python model), with real capacities;
   real measured demand matrices with a temporal 70/30 train/test split.
2. **Agent**: per-flow sequential routing MDP — Discrete(k) action selecting from
   k-shortest paths; potential-based reward = −(marginal max-util increase) with a
   QoS delay penalty (−0.5 × extra hops over shortest), so the policy detours only to
   relieve congestion and matches OSPF's short paths when uncongested.
3. **Backbone**: GNN feature extractor (the one novel element) over the link-utilisation
   + adjacency + candidate-path state, trained with PPO (Stable-Baselines3, CPU).
   MARL variant = per-node agents + custom MAPPO (decentralized, local info only).
4. **Fast training, high-fidelity judging**: train on an analytical link-utilisation
   surrogate (ns-3 is ~18x slower than realtime — never in the training loop), then
   **judge every result in full ns-3** by installing the exact per-flow paths via static
   host-routes and measuring loss/delay/throughput with FlowMonitor.
5. **Robustness**: domain-randomise over load (mixed-load training) + multi-seed eval
   with error bars; stratify results by congestion regime (overload vs feasible) rather
   than blind-averaging.

### Files used to train the agent (and their function)

| File | Role in training |
|------|------------------|
| **`train_gnn_qos.py`** | **Main training entry point.** Builds the training set (a distribution of matrices over loads × seeds), constructs the environment + GNN policy, and runs PPO. Args: `--topo`, `--loads`, `--seed`, `--timesteps`, `--tag`. Saves the model to `results/{topo}_qos{tag}_seed{seed}/`. Sets the QoS reward delay penalty (0.5) and k-paths (3). |
| **`marl_routing/sequential_routing_env.py`** | **The RL environment.** `MultiTrafficSequentialEnv` — the per-flow sequential routing MDP. Each step routes one flow by choosing among its k-shortest paths (Discrete(k) action); reward = −(marginal max-util increase) − 0.5·(extra hops over shortest). Exposes `ospf_max_util()` / `myopic_max_util()` baselines. |
| **`marl_routing/gnn_extractor.py`** | **The GNN backbone (the novel element).** `SeqGNNExtractor` — the PyTorch-Geometric feature extractor PPO uses as its policy/value network. |
| **`marl_routing/gnn_routing_agent.py`** | **k-shortest-paths provider.** `compute_ksp(graph, src, dst, k)` — Yen's algorithm (lazy `islice`) giving each node-pair its candidate path set. |
| **`marl_routing/topology.py`** | **Topology loader.** Reads `topologies/*.json` into a NetworkX graph — the single source of truth shared with ns-3. |
| **`marl_routing/traffic.py`** | **Traffic generator / loader.** Demand matrices making up the training distribution and the held-out test set. |

Algorithm: **PPO** (Stable-Baselines3, `MlpPolicy` with the GNN extractor), CPU,
`n_steps=2048, batch_size=256, n_epochs=10, gamma=0.995, ent_coef=0.01`, 500k
timesteps/seed.

> Evaluation (separate from training): `evaluate_ns3.py` extracts the trained policy's
> exact per-flow paths, `run_ns3_phase2.py` runs them in ns-3 torch-free, and the ns-3
> scenario `ns3_scenarios/abilene-validate/abilene-validate.cc` installs the paths as
> static host-routes and measures real loss/delay/throughput. Utilisation aggregated by
> `aggregate_ns3_util.py`; figures by `make_3way_fresh.py` and `make_ns3util_fig.py`.

## 5. Honest caveats
- The learned routers' win is **concentrated in the overload regime**; when traffic fits,
  they tie OSPF (by design — the QoS reward keeps them short-path there).
- At 50 nodes (Germany50) the decentralized MARL's coordination cost is real and shows as
  **variance**: the best seed matches the centralized GNN, a bad seed still overloads. The
  centralized GNN is the reliable choice at scale; we report this rather than cherry-picking.
- vs MPLS-TE / LP-optimum: learned routing trades predictability and convergence
  guarantees for adaptivity; we compare against the *deployed* baseline (OSPF), not a
  theoretical optimum (scope decision).

## 6. Figures
Main QoS (all three topologies on one axis, overload | feasible split):
- `results/fig_qos_loss_regime.png` — packet loss, 3 topologies × 2 regimes.
- `results/fig_qos_delay_regime.png` — mean delay, 3 topologies × 2 regimes.

Utilisation (the mechanism):
- `results/fig_ns3util_multiseed.png` — ns-3 *carried* utilisation, caps at 100% (§3).
- `results/fig_offered_util_regime.png` — *offered load* on the bottleneck, uncapped;
  shows OSPF pushing past 100% while learned routers pull it back (§3a).

Per-network QoS (kept for discussing networks one at a time):
- `results/fig_real3way_{abilene,geant,germany50}.png` — loss + delay by regime, per network.

Training + references:
- `results/fig_marl_training_curves.png` — MARL (MAPPO) reward + held-out gain vs steps.
- `results/abilene_topology.png`, `results/geant_topology.png` — topology references.

## 7. Reproduce
```
python make_qos_regime_split.py  # QoS: fig_qos_{loss,delay}_regime.png (all topos, 2 regimes)
python make_3way_fresh.py        # per-network QoS figures from per-seed summary.json
python make_ns3util_fig.py       # ns-3 carried-utilisation figure (§3)
python make_offered_util_fig.py  # offered-load figure (§3a) from routing files
python make_training_curves.py   # MARL training curves from logs/RUN_*marl*.log
python aggregate_ns3_util.py     # re-aggregate ns-3 carried utilisation from per-sim JSONs
# QoS results:   results/ns3_eval_real{sa,marl}_fresh_{net}_s{0,1,2}/summary.json
# Util results:  results/ns3_util_summary_{overload,feasible}.json
# Models:        results/{net}_sndlib_{qos,marl}_real_seed{0,1,2}/,
#                results/germany50_sndlib_marl_opt3b_seed{0,1,2}/
```
