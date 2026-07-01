# Checkpoint — 2026-06-23

## TL;DR
Closed most of the real-data ns-3 loop. Removed the data-side "seed" entirely (real
matrix selection is now DETERMINISTIC by timestamp). Ran the seed-free real 3-way for
Abilene + GÉANT (both: learned methods beat OSPF on real packet-level QoS, MARL strongest
under congestion). Germany50 ns-3 is pathologically slow — diagnosed + parameterised the
fix; one open methodology question (rateScale consistency) to settle before finishing it.

## What we did today
1. **Deleted the data-side seed** (`marl_routing/real_traffic.py`): `real_matrices()` now
   picks real matrices by EVENLY-SPACED timestamps within the temporal split (train=first
   70%, test=last 30%) — no RNG, reproducible from data alone. Verified two calls identical.
   Removed `seed=` from all 4 callers (train_marl, train_gnn_qos, evaluate_ns3,
   export_marl_routing). Kept the MODEL `--seed` (PPO/MAPPO init) for multi-seed error bars.
   NB: the "seed" still seen in eval OUTPUT (routing_seedN.json, stratify tuples) is now
   just a MATRIX INDEX label, not randomness — cosmetic; could rename to timestamp.
2. **Wired `--traffic real`** into evaluate_ns3.py + export_marl_routing.py (use real
   test-split matrices; same matrices for both -> fair 3-way).
3. **Deleted 25 superseded model folders** (old gravity + reconstruction runs):
   results/ 61M -> 38M. Kept the 6 real-data models (`*_sndlib_*_real_seed0`). Figures
   they produced are saved as PNGs, so nothing lost.
4. **Ran seed-free real 3-way**: Abilene + GÉANT done; Germany50 stalled (see below).

## Real-data ns-3 results (deterministic test matrices) — DONE
### Abilene (load 12) — results/ns3_eval_real{sa,marl}_abilene_sndlib/
| Regime | OSPF | SA-GNN | MARL |
|--------|------|--------|------|
| overload loss | 2.30% | 0.15% | 0.16% |
| overload delay | 36.1 ms | 10.3 ms | 11.1 ms |
| feasible loss | 0.16% | 0.14% | 0.16% |
Both cut overload loss ~15x and delay ~3.5x. Figure: fig_real3way_abilene.png (from the
earlier seeded run — regenerate from deterministic dirs).

### GÉANT (load 5) — results/ns3_eval_real{sa,marl}_geant_sndlib/
| Regime | OSPF | SA-GNN | MARL |
|--------|------|--------|------|
| overload loss | 4.24% | 3.17% | 0.83% |
| overload delay | 16.8 ms | 22.9 ms | 14.4 ms |
| feasible loss | 0.12% | 0.12% | 0.12% |
MARL strongest (loss 4.24->0.83%, ~5x). Confirms real GÉANT (22n, uniform 40G) is
WINNABLE — old "capacity-limited" was a reconstruction artifact (mixed 2.5G/10G).

## Germany50 — BLOCKED on ns-3 speed (not finished)
- ns-3 ~6–10 min PER run (50 nodes, 300 flows, load 35). At ×1000-scaled 40G caps the
  absolute packet count explodes. Job ran 3h+ and was killed.
- Also had STALE routing files mixed into the eval dir (from an earlier killed attempt) ->
  it was processing 8 matrices, not the current ones. Cleaned the dirs.
- FIX IN PLACE: parameterised `--ratescale` in run_ns3_phase2.py (was hardcoded 20). Higher
  rateScale divides rates AND caps equally -> far fewer packets, util/loss preserved ->
  much faster. ns-3 timeout already raised 400->900s.

## RESOLVED: rateScale control test (important methodology finding)
Ran the control (Germany50 seed0, OSPF, same matrix):
| rateScale | loss | delay | maxutil | wall |
|-----------|------|-------|---------|------|
| 20  | 3.791% | 15.38 ms | 87.5% | 279 s |
| 100 | 3.792% | 36.66 ms | 87.5% |  55 s |
**rateScale preserves LOSS and UTILISATION exactly, but NOT DELAY** (higher rateScale =
proportionally slower sim links = inflated serialization+queueing delay). => For delay to
be comparable across topologies we MUST use the SAME rateScale everywhere. DECISION:
**keep rateScale=20 for all three.** Germany50 is tractable at rs=20 anyway (~4.6 min/run;
the earlier 3h stall was stale routing files = 8 matrices + 300 flows). Running now at
rs=20, max-flows=200, 5 matrices -> ~90 min.
Params that legitimately differ per topology: load scale (normalises congestion) and
max-flows (germany50 2450 flows -> filtered, applied equally to all 3 methods). rateScale,
simTime now identical across all three.

## ns-3 DELAY BUG found + fixed (important)
Germany50 SA-GNN delay came out 0.04 ms — sub-physical (below the 0.13 ms min link delay).
Root cause: abilene-validate.cc line 82 used `MilliSeconds(uint64_t(link.delayMs))`, which
TRUNCATES sub-millisecond link delays to 0. Germany50 links are 0.13-1.26 ms (small
country) → most became 0 ms propagation; with the GNN avoiding queueing, total delay ≈ 0.
Abilene/GÉANT also lost fractional ms (less severe — their links are larger). FIX:
`NanoSeconds(uint64_t(link.delayMs * 1e6))` (ms→ns, no truncation). Rebuilt ns-3, synced
the ns3_scenarios mirror. Verified: germany50 seed0 GNN delay 0.04 → 1.86 ms (sensible).
LOSS is unaffected by this (depends on queue overflow, not delay precision). RE-RUNNING
the ns-3 PHASE on all 6 existing routing dirs (paths unchanged) at rateScale 20 to get
correct, consistent delays everywhere (~2-2.5h). Loss numbers below stand; delay numbers
will be refreshed.

## Tools touched today
real_traffic.py (deterministic selection) · evaluate_ns3.py + export_marl_routing.py
(--traffic real) · run_ns3_phase2.py (--ratescale param, timeout 900s).

## EOD UPDATE (2026-06-24): real-data 3-way COMPLETE
All 6 dirs re-run through the delay-fixed ns-3. FINAL real-data, seed-free,
delay-corrected 3-way (overload regime, loss / delay):
- Abilene:  OSPF 2.32% 37.5ms | SA-GNN 0.17% 11.7ms | MARL 0.18% 12.6ms
- GÉANT:    OSPF 7.46% 17.9ms | SA-GNN 1.44% 15.5ms | MARL 0.86% 14.9ms  (MARL best)
- Germany50:OSPF 3.16% 17.4ms | SA-GNN 0.03% 1.9ms  | MARL 0.96% 17.6ms  (SA-GNN best)
Feasible: all ~tie. Both learned methods beat OSPF on all three. MARL ≈/beats SA on
small+mid nets; SA clearly best at 50 nodes (decentralization cost shows). Figures
results/fig_real3way_{abilene,geant,germany50}.png. RESULTS_SUMMARY.md §2 updated to real
numbers (old §2b/§3 marked SUPERSEDED). Nothing running; state clean.

## Next session (priority order, discussed with user)
1. **Multi-seed error bars** — currently model seed 0 only. Train seeds 1,2 on real data
   (train_gnn_qos.py + train_marl.py --traffic real, per-topo loads abilene 8/12/16,
   geant 3/5/7, germany50 35/50/65), re-eval, report mean±std. Most important rigor step.
2. **Write-up** — result is complete + defensible. Methodology gold: real SNDlib data,
   temporal split, the two bugs caught (rateScale delay-dependence, ns-3 ms-truncation).
3. (optional) Upper-bound baseline (LP-opt / iterated best-response) → "X% of optimal".
4. (optional) Explain GÉANT-vs-Germany50 flip (MARL beats SA on GÉANT, loses at 50n) —
   the "when does decentralization pay off" analysis.
5. (optional, research) inter-agent comms to close MARL's Germany50 delay gap.
User leaning: lock down (multi-seed) then write. Decide tomorrow.

## OLD next-session notes (superseded by EOD update above)
1. Settle the rateScale question (run the control test), then finish Germany50 real 3-way.
2. Regenerate all three real figures from the deterministic dirs (fig_real3way_*).
3. Update RESULTS_SUMMARY.md to the real-data, seed-free numbers (replace the old
   synthetic/reconstructed §2b/§3).
4. (optional) rename data-side "seed" label -> matrix index/timestamp in eval outputs.
5. (optional) multi-seed (model seeds 1,2) for error bars on real data.
