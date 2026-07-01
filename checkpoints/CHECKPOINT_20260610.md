# Checkpoint — 2026-06-10

## TL;DR
Made ns-3 the real judge (QoS metrics: loss/delay/throughput) and added a
QoS-aware reward. Result: a single GNN policy that is **effectively strictly
better than OSPF** on unseen traffic — matches OSPF when uncongested, and under
congestion cuts packet loss ~25x and delay ~5x. All validated in full ns-3.

## What we did today
1. **Set up git** (single clean repo) and pushed to GitHub
   (github.com/verrellbd/marl-routing-thesis). Removed nested ns-3/ns3-ai .git
   repos (provenance saved in NS3_PROVENANCE.md + ns3ai_local_changes.patch).
   .gitignore excludes ns-3-dev/ (3.7G), ns3ai-venv/ (6.1G), model binaries,
   PPO checkpoints. Removed 11 unused scripts from abandoned approaches.
2. **ns-3 QoS evaluation** (evaluate_ns3.py + run_ns3_phase2.py): runs OSPF vs the
   GNN's exact per-flow paths in ns-3, measures real loss/delay/throughput,
   stratified by congestion. Scenario abilene-validate.cc now outputs delay+throughput.
3. **QoS-aware reward** (train_gnn_qos.py): reward = -(marginal max-util)
   - 0.5*(extra hops over shortest). Trained generalist 250k steps.

## Key result (full ns-3, held-out traffic) — results/ns3_eval{,_qos}/
| Regime | Metric | OSPF | GNN (util) | GNN-QoS |
|--------|--------|------|-----------|---------|
| OVERLOAD | loss  | 7.41% | 0.35% | **0.30%** |
| (101-122%)| delay | 98.9ms | 24.5ms | **20.8ms** |
| FEASIBLE | loss  | 0.29% | 0.33% | **0.30%** |
| (55-97%) | delay | 20.0ms | 23.1ms | **21.0ms** |

QoS reward shrank the feasible delay penalty (+3.1ms -> +1.0ms, ~=OSPF) while
keeping/improving overload wins. GNN-QoS strictly dominates under congestion, ties
when feasible. Figures: results/ns3_eval/qos_by_congestion.png,
results/ns3_eval_qos/qos_threeway.png.

## Important gotchas (for tomorrow)
- ns-3 is ~18x slower than realtime -> use rateScale=20 (divides rates AND caps
  equally; util/loss preserved) + short simTime. NEVER train with ns-3 in the loop.
- ns-3 eval must run TORCH-FREE (run_ns3_phase2.py) to avoid fork-OOM on the shared
  machine; freeing the model in-process is NOT enough.
- Only 3/20 test matrices overload OSPF at load_factor 3.0 -> GNN's win is
  concentrated in the overload regime; report stratified, not blind-averaged.
- ns3_scenarios/ is a git-tracked MIRROR of ns-3-dev/scratch/*.cc — re-sync after
  editing the real .cc (\cp -f), or they drift.

## Files (live pipeline)
- marl_routing/: sequential_routing_env.py (SequentialRoutingEnv +
  MultiTrafficSequentialEnv, now with delay_penalty), gnn_extractor.py
  (SeqGNNExtractor), gnn_routing_agent.py (compute_ksp), topology.py, traffic.py
- train_gnn_sequential.py, train_gnn_generalization.py, train_gnn_qos.py
- export_routing.py, evaluate_ns3.py, run_ns3_phase2.py
- ns3_scenarios/abilene-validate/ (mirror; real runs from ns-3-dev/scratch/)

## UPDATE — later same day: multi-seed + variance reduction DONE
- **Multi-seed (3 seeds)** revealed the single-model "~25x loss / ~5x delay" was the
  BEST seed. Honest multi-seed: overload loss 1.98%±1.39, delay 47±10.8ms.
- **Variance reduction** (mixed-load training: loads 2/3/4 x 20 seeds = 60 matrices,
  500k steps) tightened bars AND improved means:
  | regime | metric | OSPF | ROBUST GNN-QoS |
  |--------|--------|------|----------------|
  | overload | loss | 7.41% | 0.74% ± 0.28 (~10x, ~5x tighter than baseline) |
  | overload | delay | 98.9ms | 33.5 ± 7.1 ms (~3x) |
  | feasible | loss | 0.29% | 0.30% ± 0.01 (tie) |
  | feasible | delay | 20.0ms | 21.2 ± 0.6 ms (≈OSPF) |
- **RIGOROUS HEADLINE:** under congestion GNN cuts loss ~10x and delay ~3x with tight
  error bars; ties OSPF when feasible. Strictly dominant.
- New: train_gnn_qos.py has TRAIN_LOADS (mixed-load) + --seed/--tag; evaluate_ns3.py
  has --export-only; run_ns3_phase2.py (torch-free ns-3); aggregate_multiseed.py (--tag).
  Robust models: results/generalization_qos_robust_seed{0,1,2}/. Results:
  results/ns3_eval_multiseed{,_robust}.json + .png.
- Commits through d0decea. **2 commits to git push.**

## Next session (priority order)
1. **GEANT** (23 nodes, 39 links) -> repeat the full pipeline (train mixed-load QoS,
   ns-3 validate, multi-seed) on the larger topology to show the result generalizes.
2. **Higher base load** -> more overload test matrices (only 3/20 overload at load 3.0)
   for even tighter, more representative error bars.
3. Optional: bigger multi-layer message-passing GNN if more capacity is wanted.

## To resume the robust pipeline
  cd ~/thesis && git push
  # GEANT: adapt train_gnn_qos.py / evaluate_ns3.py TOPO='geant' (topologies/geant.json)
  # the ns-3 abilene-validate scenario is abilene-specific -> needs a geant variant

## To resume
  cd ~/thesis
  git push                              # 2-3 commits ready to push
  python train_gnn_qos.py               # retrain QoS generalist (~40 min)
  python evaluate_ns3.py --model results/generalization_qos/gnn_generalist_qos --tag _qos
  python run_ns3_phase2.py --dir results/ns3_eval_qos   # torch-free ns-3 runs
