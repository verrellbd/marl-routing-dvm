# What is in `results/`

Everything here is the reported measurement batch. Superseded runs have been removed, so
there is nothing in this directory that the tables do not use.

## Aggregated grids

| File | What it is |
|------|------------|
| `final_ns3_grid.json` | **The packet-level table.** Loss, delay and utilization by topology, regime and method, over 558 ns-3 simulations with 0 failures. |
| `offered_grid.json` | The analytical table: maximum offered load on the same six topology/regime cells, so the two are directly comparable. |
| `reward_decomp.json` | Episode return split into its congestion and delay terms, with detour counts and the final bottleneck as a fraction of OSPF's. |
| `surrogate_validation.json` | Rank agreement between the analytical training objective and the ns-3 measurement. Produced by `validate_surrogate.py`; re-simulates nothing. |
| `greedy_k3_grid.json` | Greedy best-response reference. Shares the learned action space exactly, but needs global link state after every hop, so it is an upper reference rather than a deployable method. |
| `width_selection.json` | Hidden-width choice for the decentralized policy, scored on the **training** topologies with held-out matrices — never on the evaluation backbones. |

## Per-simulation output

`ns3f_<arm>_<topo>_s<seed>/` — 120 directories.

- **Arms**: `ecmp`, `marlh32` (the decentralized policy), `singleRM` (the centralized baseline).
- **Topologies**: `abilene`, `geant`, `germany50`, plus `g50feas` — Germany50's feasible
  regime, measured at lower demand scales because its normal scales are entirely overload.
- **Seeds**: `s0` … `s9`.

Each directory holds `routing_seed<i>.json` (the per-flow paths installed in ns-3, with the
matching OSPF paths and that matrix's regime label) and `ns3_<tag>_<i>.json` (the simulator
output: loss, delay, throughput, per-link utilization).

OSPF is deterministic given a matrix, so it was simulated once rather than per seed. Its 18
runs live inside the `ns3f_marlh32_*_s0`, `_s1` and `_s2` directories as `ns3_ospf_<i>.json`.

That gives 3 arms × 10 seeds × 18 matrices = 540, plus 18 OSPF = **558 simulations**.

## Policies

| Directory | Arm |
|---|---|
| `marlgnn_tier2m15cm_seed0` … `seed9` | Decentralized MAPPO + GNN, hidden width 32 (`policy.pt`) |
| `single_singleH64gRM_seed0` … `seed9` | Centralized PPO baseline, hidden width 64 (`policy.zip`) |

Each also carries `summary.json`, holding the zero-shot evaluation recorded at the end of
training — on the analytical objective, before any ns-3 run.

Intermediate training checkpoints are not kept; only the final policy of each run is here.
