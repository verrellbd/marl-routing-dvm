# Checkpoint — 2026-06-17

## TL;DR
Decided **Option B: build genuine multi-agent RL** (the prior system was single-agent
PPO+GNN, which didn't match the "MARL" thesis title). Cleaned 9 dead files, wrote the
plan (`MARL_PLAN.md`), and **built + validated M1–M3**: a per-node MAPPO routing system
that, on Abilene, beats OSPF AND a myopic greedy on every held-out congested matrix
using only LOCAL observations. Also made today's explainer figures (bottleneck,
traffic heatmap, pipeline) and the results summary.

## Decisions (locked)
- **Option B — genuine MARL.** Single-agent GNN (train_gnn_qos.py) is now a BASELINE.
- **Per-node router agents** (each node picks next hop from local obs; Q-routing lineage).
- **Custom MAPPO** (CTDE: decentralized actor + centralized critic), NOT RLlib.
- ns-3 evaluation half is REUSED unchanged (agents emit per-flow paths via rollout).

## Built today (MARL — M1..M3 done)
- `marl_routing/multiagent_routing_env.py` (M1): per-node hop-by-hop forwarding on the
  analytical link-util model. Key lever **stretch=1** (next hop must be equal-or-closer
  to dst). stretch=0 → no headroom (=OSPF); stretch=2 → over-detours + revisit bug.
  stretch=1 = loop-free (visited-set) + real load-balancing room. Interfaces:
  reset/step → (obs, mask, gstate); rollout_paths(act_fn, rates) → per-flow paths for ns-3.
- `marl_routing/mappo.py` (M2): ActorCritic (MLP actor over LOCAL obs + masked
  Categorical; MLP critic over GLOBAL gstate) + MAPPO (collect→GAE→clipped PPO).
  Parameter sharing across node-agents. Smoke test learns + beats OSPF.
- `train_marl.py` (M3): mixed-load, multi-seed trainer mirroring train_gnn_qos.py.
  Saves actor-critic to results/{topo}_marl{tag}_seed{seed}/mappo_actor_critic.pt.

## Key result so far (Abilene, analytical, held-out load 4.0) — results/abilene_marl_v1_seed0/
| seed | OSPF | greedy | MARL | |
|------|------|--------|------|---|
| 1000 | 106.5% | 106.7% | **84.1%** | WIN (→feasible) |
| 1005 | 126.7% | 128.2% | **113.5%** | WIN |
| 1009 | 134.7% | 133.2% | **99.8%** | WIN (→feasible) |
| 1013 | 148.0% | 145.1% | **132.3%** | WIN |
| 1018 | 162.0% | 163.6% | **149.1%** | WIN |
MARL beats OSPF AND the myopic greedy on all 5 → learned non-myopic coordination with
only LOCAL info. (600k steps, ~6 min on CPU.)

## Bugs fixed today
- `compute_ksp` 3000x slowdown (islice) — from yesterday, now in.
- stretch design lever (0/1/2 swept; 1 chosen).
- MAPPO eval corrupted training by reusing the trainer's env → eval now uses a SEPARATE
  env. Also hardened env.step with an emergency min-distance hop (never crashes).

## Cleanup
Removed 9 dead files via git rm (recoverable): weight-control cluster (routing_env,
gnn_env, gnn_env_simple, gnn_policy) + superseded (ksp, analytical_routing_env,
export_routing, train_gnn_sequential, train_gnn_generalization). Live pipeline verified.

## Explainer artifacts made today (for write-up)
- results/fig_ospf_vs_gnn.png — single-agent GNN vs OSPF, both topologies, by regime.
- results/fig_geant_headroom.png — GEANT capacity-limited proof (OSPF vs greedy k3/5/8).
- results/fig_geant_bottleneck.png — node 22 (RU) single 2.5G uplink, 3893→156%.
- results/fig_traffic_heatmap.png — gravity traffic hot spots (Abilene + GEANT).
- results/fig_pipeline.png — demand → packets → loss/delay (train-fast/judge-in-ns3).
- RESULTS_SUMMARY.md — full written summary incl. §5b files-used-to-train.

## Next session (continue MARL)
1. **M4 — ns-3 validation of MARL**: extract MARL per-flow paths (rollout_paths) →
   write routing JSON (gnn_path slot = MARL path) → run_ns3_phase2.py / abilene-validate.cc
   UNCHANGED. Multi-seed, stratified loss/delay. (May need a small adapter so
   evaluate_ns3.py can load a MAPPO actor instead of an SB3 model.)
2. Train Abilene seeds 1,2 (multi-seed error bars) + add a delay-penalty sanity (ties
   OSPF when feasible, like the single-agent QoS story).
3. **M5 — GEANT**: repeat; expect the capacity limit to bind MARL too (honest).
4. Final 3-way comparison + figures: OSPF vs single-agent GNN vs MARL on both topologies.

## To resume
  cd ~/thesis
  python -m marl_routing.mappo            # smoke test (sanity)
  # model ready: results/abilene_marl_v1_seed0/mappo_actor_critic.pt
  # build M4 adapter: load MAPPO actor, rollout_paths -> routing JSON -> ns-3
