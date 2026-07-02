# GNN Routing vs OSPF — Results Summary

*MARL for Network Routing (master's thesis). All QoS numbers are from full ns-3
packet-level simulation on held-out traffic, not the analytical model.*

---

## 1. The question
Can a GNN-based routing agent (learned, per-flow path selection from k-shortest
paths) beat the deployed baseline (OSPF shortest-path) under dynamic traffic, when
judged by real packet-level QoS — loss, delay, throughput?

## 2. Headline finding (REAL DATA — current/final result)
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
deterministic test matrices, packet-level QoS:**

Abilene (real Zhang TM):
| Regime | Metric | OSPF | single-agent GNN | MARL (decentralized) |
|--------|--------|------|------------------|----------------------|
| Overload | loss  | 2.32% | **0.17%** | **0.18%** |
| Overload | delay | 37.5 ms | **11.7 ms** | **12.6 ms** |
| Feasible | loss  | 0.18% | 0.16% | 0.18% |

GÉANT (real Uhlig/TOTEM TM):
| Regime | Metric | OSPF | single-agent GNN | MARL |
|--------|--------|------|------------------|------|
| Overload | loss  | 7.46% | 1.44% | **0.86%** |
| Overload | delay | 17.9 ms | 15.5 ms | **14.9 ms** |
| Feasible | loss  | 0.12% | 0.14% | 0.15% |

Germany50 (real DFN TM, 50 nodes, top-200 flows) — MARL hop-capped, **multi-seed (0/1/2)**:
| Regime | Metric | OSPF | single-agent GNN | MARL (mean ± std) | MARL best seed |
|--------|--------|------|------------------|-------------------|----------------|
| Overload | loss  | 3.16% | **0.03%** | 2.78 ± 2.56% | **0.03%** (seed 0) |
| Overload | delay | 17.4 ms | **1.9 ms** | **8.2 ± 5.4 ms** | 2.2 ms (seed 0) |
| Feasible | loss  | 0.12% | 0.03% | ~0.03% | 0.03% |

> **Germany50 hop-cap fix — honest multi-seed result (2026-07-01/02).** The uncapped MARL had
> a delay/loss gap at 50 nodes (0.96% loss, 17.6 ms) because per-node agents, on local info
> only, wandered into long detours (up to 12 hops) that concentrated load and left a link
> saturated (~105%), driving queueing delay. NB: germany50 detour links are short, so the
> driver was **queueing (utilisation), not propagation**. A **hop cap** on the agents'
> forwarding (final path ≤ shortest-path hops + 4; **topology untouched, only the action
> mask**) stops the wandering and improves congestion relief.
> **What it robustly fixes:** DELAY — every seed beats OSPF (mean 8.2 ± 5.4 ms vs OSPF 17.4,
> vs uncapped 17.6). **What stays hard:** LOSS is **high-variance across seeds** (seed 0
> 0.03% matches the centralized GNN; seed 2 6.2% is *worse* than OSPF); mean 2.78% only edges
> OSPF's 3.16%. So decentralized coordination at 50 nodes is **viable but less reliable than
> central control** — the coordination cost shows up as VARIANCE, not a mean gap (consistent
> with the analytical ±7 bars). Per-seed loss/delay: seed0 0.03%/2.2ms · seed1 2.10%/7.1ms ·
> seed2 6.20%/15.2ms. Models results/germany50_sndlib_marl_opt3b_seed{0,1,2}/ (λ=0.1,
> max_stretch=4). Do NOT claim "MARL matches central at 50n" in general — only best-seed.

**Findings (real data):**
- **Both learned methods beat OSPF under congestion on all three real networks** — loss
  cut ~13x (Abilene), ~5–9x (GÉANT); on Germany50 the centralized GNN cuts loss ~100x while
  the decentralized MARL cuts delay reliably but is loss-variance-limited (see below).
- **Decentralized MARL matches the centralized GNN on the small/mid networks** (Abilene,
  GÉANT — on GÉANT it slightly beats it) — local-only routing is competitive with a central
  controller up to ~22 nodes.
- **At 50 nodes the coordination cost is real and shows as VARIANCE.** A hop cap (shortest +
  4 hops; topology unchanged) fixes MARL's DELAY robustly (every seed beats OSPF) and stops
  the load-concentrating wandering, but LOSS stays high-variance across seeds — the best seed
  matches the centralized GNN, a bad seed still overloads. Honest scale finding: decentralized
  coordination is viable but less *reliable* than central control at 50 nodes. The centralized
  GNN remains the safe choice at scale.
- All methods ~tie when the network is uncongested (feasible regime).

Figures: results/fig_real3way_{abilene,geant,germany50}.png. Parameter consistency:
rateScale=20, simTime=8 identical across all three; load scale (per-network, to normalise
congestion) and max-flows (Germany50 only, 2450→top-200 for ns-3 tractability) differ for
principled reasons. ns-3 link delays use nanosecond resolution (fixed a ms-truncation bug
that had zeroed Germany50's sub-ms link delays).

## 2a. Multi-seed robustness (model seeds 0/1/2)
The result is not a lucky single run. Re-training each policy with 3 model seeds and
evaluating on the deterministic real test matrices, max link-utilisation in the overload
regime (mean ± std over seeds):
| Topology | OSPF | single-agent GNN | MARL |
|----------|------|------------------|------|
| Abilene | 122% | **67 ± 2%** | **64 ± 2%** |
| GÉANT | 126% | 92 ± 2% | 97 ± 7% |
| Germany50 | 109% | **86 ± 2%** | 99 ± 7% |
Both methods beat OSPF across all seeds. SA-GNN bars are tight (±2) everywhere; MARL is
tight on Abilene but higher-variance on the larger nets (±7) — the coordination-cost-at-
scale finding is robust, not a single-seed artifact. Figure: fig_multiseed_overload.png.
(ns-3 packet-level numbers in §2 are the seed-0 high-fidelity anchor; these analytical
error bars establish seed-robustness cheaply.)

> NOTE: sections 2b/3 below are the EARLIER synthetic-gravity + reconstructed-topology
> runs. They are SUPERSEDED by the real-data result above and kept only for history.

## 2b. (SUPERSEDED — synthetic gravity / reconstructed topologies)
The earlier 3-way used a synthetic gravity traffic model and reconstructed/approximate
topologies. Headline then: MARL beat OSPF and nearly matched the centralized GNN on
Abilene; GÉANT looked "capacity-limited" (an artifact of the reconstructed mixed-2.5G/10G
topology — the REAL uniform-40G GÉANT is winnable); Germany50 MARL looked high-variance
(an artifact of adversarial synthetic hot-spots — real traffic is better-behaved). Numbers
retained in git history / older checkpoints.

## 3. The numbers (ns-3, held-out traffic, stratified by congestion regime)

### Abilene — multi-seed (3 seeds, mean ± std)
| Regime | Metric | OSPF | GNN | Result |
|--------|--------|------|-----|--------|
| Overload (>100% offered) | loss  | 7.41% | **0.74% ± 0.28** | **~10x lower** |
| Overload | delay | 98.9 ms | **33.5 ± 7.1 ms** | **~3x lower** |
| Feasible (<100%) | loss  | 0.29% | 0.30% ± 0.01 | tie |
| Feasible | delay | 20.0 ms | 21.2 ± 0.6 ms | ≈ OSPF |

### GÉANT — multi-seed (3 seeds, mean ± std)
| Regime | Metric | OSPF | GNN | Result |
|--------|--------|------|-----|--------|
| Overload (>100% offered) | loss  | 5.95% | 5.54% ± 0.09 | ~tie (small edge) |
| Overload | delay | 90.9 ms | 87.8 ± 3.9 ms | ~tie (small edge) |
| Feasible / stressed (~90%) | loss  | 0.30% | 0.30% | tie |
| Feasible | delay | 20.5 ms | 21.1 ± 0.5 ms | tie |

Best single GÉANT case (model seed 0, heavily overloaded matrix @145%): loss
10.5 → 9.4%, delay 140 → 124 ms. But with model-seed variance (one seed regresses on
a borderline matrix), so honestly: **a tie with a small, noisy edge.**

## 4. Why GÉANT ties (the capacity-limited proof)
A near-optimal **greedy best-response** path selector — given up to **k=8** candidate
paths per flow — *still* cannot beat OSPF on the overloaded GÉANT matrices:

| matrix | OSPF | greedy k=3 | greedy k=5 | greedy k=8 |
|--------|------|-----------|-----------|-----------|
| 1005 | 156% | 156% | 156% | 156% |
| 1008 | 112% | 107% | 107% | 107% |
| 1013 | 102% | 102% | 102% | 102% |

(analytical max link-utilisation, load 1.5). More candidate paths open **zero**
headroom. A load sweep confirms the optimal reroute beats OSPF by only **3–5 points of
max-utilisation at any load** — GÉANT's shortest paths are already near-optimal, and
the overloads sit on a saturated cut no rerouting can fix. Decisive corroboration: even
at 90% analytical utilisation, ns-3 packet loss is near-zero (0.19–0.39%) for *both*
OSPF and GNN — the traffic still fits, so there is nothing to win.

## 5. What we did (method)
1. **Topologies**: Abilene & GÉANT loaded from JSON (single source of truth for ns-3
   and the Python model); gravity-model traffic with a load-factor sweep.
2. **Agent**: per-flow sequential routing MDP — Discrete(k) action selecting from
   k-shortest paths; potential-based reward = −(marginal max-util increase) with a
   QoS delay penalty (−0.5 × extra hops over shortest), so the policy detours only to
   relieve congestion and matches OSPF's short paths when uncongested.
3. **Backbone**: GNN feature extractor (the one novel element) over the link-utilisation
   + adjacency + candidate-path state, trained with PPO (Stable-Baselines3, CPU).
4. **Fast training, high-fidelity judging**: train on an analytical link-utilisation
   surrogate (ns-3 is ~18x slower than realtime — never in the training loop), then
   **judge every result in full ns-3** by installing the exact per-flow paths via static
   host-routes and measuring loss/delay/throughput with FlowMonitor.
5. **Robustness**: domain-randomise over load (mixed-load training) + multi-seed eval
   with error bars; stratify results by congestion regime (overload vs feasible) rather
   than blind-averaging.
6. **GÉANT**: retrained on the *flexible* load band (1.0/1.2/1.4) — the old 1.0/1.5/2.0
   band was half capacity-limited noise — then ran the headroom diagnostic that proved
   the tie is structural.

## 5b. Files used to train the agent (and their function)

Training pipeline — what each file does, in the order data flows through it:

| File | Role in training |
|------|------------------|
| **`train_gnn_qos.py`** | **Main training entry point.** Builds the training set (a distribution of gravity matrices over loads × seeds), constructs the environment + GNN policy, and runs PPO. Args: `--topo` (abilene/geant), `--loads` (mixed-load band, e.g. `1.0,1.2,1.4`), `--seed`, `--timesteps`, `--tag`. Saves the model to `results/{topo}_qos{tag}_seed{seed}/gnn_generalist_qos.zip`. Sets the QoS reward delay penalty (0.5) and k-paths (3). |
| **`marl_routing/sequential_routing_env.py`** | **The RL environment.** `MultiTrafficSequentialEnv` — the per-flow sequential routing MDP the agent trains in. Each step routes one flow by choosing among its k-shortest paths (Discrete(k) action); reward = −(marginal max-util increase) − 0.5·(extra hops over shortest). Precomputes candidate paths once, randomises which traffic matrix each episode uses, and exposes `ospf_max_util()` / `myopic_max_util()` baselines. |
| **`marl_routing/gnn_extractor.py`** | **The GNN backbone (the novel element).** `SeqGNNExtractor` — the PyTorch-Geometric feature extractor PPO uses as its policy/value network. Consumes the graph state (per-link utilisation, adjacency, candidate-path features, flow rate) via an arc→node incidence and produces the embedding the actor/critic heads sit on. |
| **`marl_routing/gnn_routing_agent.py`** | **k-shortest-paths provider.** `compute_ksp(graph, src, dst, k)` — Yen's algorithm (lazy `islice`) giving each node-pair its candidate path set, the action space the agent selects from. |
| **`marl_routing/topology.py`** | **Topology loader.** Reads `topologies/{abilene,geant}.json` into a NetworkX graph (capacities, delays, strong-connectivity check) — the single source of truth shared with ns-3. |
| **`marl_routing/traffic.py`** | **Traffic generator.** `generate_matrix(topo, load_factor, seed)` — gravity-model N×N demand matrices (different seeds = different hot spots) that make up the training distribution and the held-out test set. |

Algorithm: **PPO** (Stable-Baselines3, `MlpPolicy` with the GNN extractor), CPU,
`n_steps=2048, batch_size=256, n_epochs=10, gamma=0.995, ent_coef=0.01`, 500k
timesteps/seed (~5 min/seed after the KSP fix).

Example invocation (the GÉANT flex-band run):
```
python train_gnn_qos.py --topo geant --loads 1.0,1.2,1.4 --seed 0 --timesteps 500000 --tag _flex
```

> Evaluation (separate from training): `evaluate_ns3.py` extracts the trained policy's
> exact per-flow paths, `run_ns3_phase2.py` runs them in ns-3 torch-free, and the ns-3
> scenario `ns3_scenarios/abilene-validate/abilene-validate.cc` installs the paths as
> static host-routes and measures real loss/delay/throughput.

## 6. Honest caveats
- The GNN's win is **concentrated in the overload regime**; when traffic fits, it ties
  OSPF (by design — the QoS reward keeps it short-path there).
- On GÉANT the result is a **tie**, with model-seed variance that can regress on
  borderline matrices. We report this rather than cherry-picking the best seed.
- vs MPLS-TE / LP-optimum: learned routing trades predictability and convergence
  guarantees for adaptivity; we compare against the *deployed* baseline (OSPF), not a
  theoretical optimum (scope decision).

## 7. Figures
- `results/fig_ospf_vs_gnn.png` — main result: loss + delay by regime, both topologies.
- `results/fig_geant_headroom.png` — GÉANT capacity-limited proof (OSPF vs greedy k=3/5/8).
- `results/abilene_topology.png`, `results/geant_topology.png` — topology references.

## 8. Reproduce
```
python make_figures.py                     # regenerate both figures from the JSONs
python geant_headroom.py                   # GÉANT capacity-limited diagnostic
# Abilene results:  results/ns3_eval_multiseed_robust.json
# GÉANT results:    results/ns3_eval_geantStress_seed{0,1,2}/summary.json
# Models:           results/{abilene,geant}_qos*_seed{0,1,2}/
```
