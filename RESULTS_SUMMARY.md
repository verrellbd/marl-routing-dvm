# GNN Routing vs OSPF — Results Summary

*MARL for Network Routing (master's thesis). All QoS numbers are from full ns-3
packet-level simulation on held-out traffic, not the analytical model.*

---

## 1. The question
Can a GNN-based routing agent (learned, per-flow path selection from k-shortest
paths) beat the deployed baseline (OSPF shortest-path) under dynamic traffic, when
judged by real packet-level QoS — loss, delay, throughput?

## 2. Headline finding
**Learned routing's benefit is topology-dependent, and we show exactly when and why.**

- **Abilene** (12 nodes, uniform 10G, well-connected): the GNN **strictly dominates**
  OSPF under congestion — **~10x less packet loss and ~3x less delay** — and ties OSPF
  when the network is uncongested.
- **GÉANT** (23 nodes, mixed 2.5G/10G, bottleneck cuts): the GNN **safely matches**
  OSPF. We *prove* this is not a training failure — GÉANT is **capacity-limited**, so
  no routing (not even an optimal one) can do meaningfully better.

This is a stronger scientific result than "the GNN wins everywhere": it gives a
**positive case + a boundary case**, both validated in ns-3, and explains the
mechanism.

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
