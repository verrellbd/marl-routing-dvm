# Checkpoint — 2026-06-09

## TL;DR
Pivoted training from ns-3-in-the-loop to a fast **analytical** environment, then
reformulated to **sequential per-flow routing**. Result: **GNN beats OSPF by 8–21
pts, consistently, with the margin growing under congestion** — and stays feasible
(83.6%) where OSPF overloads (104.4%). Strong core thesis result.

## Why we changed the approach (key findings)
1. ns-3 in the training loop was fatal: ~18x slower than real-time AND the env's
   `timeout 30` truncated each 60s sim to ~2-5s → agent learned from **garbage
   (non-steady-state) rewards**. ~66h for two models. GPU doesn't help (GNN is tiny;
   bottleneck is single-core packet sim).
2. Built `marl_routing/analytical_routing_env.py` — routing+demand+capacity → max
   link util (standard TE/LP computation). Validated vs ns-3 (analytical 21.95% ≈
   ns-3 43.85%/2, the UdpEcho echo factor). ~10,000x faster, correct rewards.
3. ns-3 is retained for **final evaluation only** (per-flow path validation, pending).

## Results (Abilene, α1.5_min30 = 106 flows, max link util %, lower better)

### Sequential GNN (the strong result) — results/sequential_gnn/
| load | OSPF | GNN | myopic | random | GNN vs OSPF |
|------|------|-----|--------|--------|-------------|
| ×1 | 34.8 | 26.7 | 25.2 | 31.3 | +8.2 |
| ×2 | 69.6 | 54.6 | 50.4 | 68.3 | +15.1 |
| ×3 | 104.4 (overload) | 83.6 (feasible) | 75.5 | 101.7 | +20.8 |
Figure: results/sequential_gnn/load_sweep.png

### Earlier one-shot GNN (weaker; kept for comparison)
- v1 sparse reward: results/analytical_gnn/  (+1.4/+8.2/+8.0)
- v2 shaped reward: results/analytical_gnn_v2/ (+1.4/+2.8/+8.0)

## Honest caveats (address in writeup / next session)
- **GNN ≈ myopic heuristic** (slightly worse). PPO learned ~greedy behavior. Need to
  justify GNN over a simple heuristic → see "generalization" below.
- **Single-seed** runs → several pts of variance. Need multi-seed mean±std for the
  final thesis numbers.

## Files created today
- marl_routing/analytical_routing_env.py  (one-shot analytical env, load_scale + reward_mode)
- marl_routing/sequential_routing_env.py   (sequential per-flow env — the good one)
- marl_routing/gnn_extractor.py            (added SeqGNNExtractor)
- train_gnn_analytical.py / _v2.py / train_gnn_sequential.py

## Next session (priority order)
1. **Generalization experiment** (makes the GNN *necessary*, not just sufficient):
   train one GNN policy on a DISTRIBUTION of gravity-model traffic matrices, test on
   UNSEEN matrices. GNN (single policy, no recompute) vs OSPF vs per-matrix greedy.
   This is the "dynamic traffic" research-question payoff.
2. **Multi-seed** (5 seeds) for mean±std on the headline sweep.
3. **ns-3 validation (step 2):** modify abilene-gym scenario to install the GNN's
   exact per-FLOW paths (static/source routing) — current scenario routes
   per-DESTINATION via OSPF weights, so it can't represent arbitrary per-flow choices.
   Then report ns-3-backed GNN vs OSPF (max-util + packet loss at ×3 overload).
4. Repeat on GEANT once Abilene story is locked.

## To resume: re-run any sweep with
  cd ~/thesis && python train_gnn_sequential.py      # ~30 min, CPU
