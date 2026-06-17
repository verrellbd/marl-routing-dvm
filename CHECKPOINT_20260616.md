# Checkpoint — 2026-06-16

## TL;DR
GEANT is **done and resolved**: the GNN **ties OSPF** on GEANT (it does NOT
dominate like on Abilene), and we **proved why** — GEANT is *capacity-limited*,
not undertrained. This is the honest boundary case that completes the thesis:
learned routing strictly dominates on a well-connected uniform fabric (Abilene,
~10x loss cut under congestion) and safely matches OSPF on a bottleneck-limited
backbone (GEANT), never catastrophically regressing. Positive case + boundary
case, both ns-3-validated. Also fixed a 3000x KSP performance bug.

## What we did today
1. **Diagnosed GEANT's weak result** (was: GNN ≈ OSPF, suspected undertraining).
   - Headroom diagnostic (geant_headroom.py): even a near-optimal greedy
     best-response at **k=8** can't beat OSPF on the overload matrices.
     Load 1.5 analytical max-util %: seed 1005 OSPF 156 / g@k3 156 / k5 156 / k8 156
     (zero headroom — saturated cut); 1008 OSPF 112 / greedy 107; 1013 102 / 102.
     **More candidate paths open NO headroom.**
   - Load sweep: optimal rerouting beats OSPF by only **3–5 pts of max-util at any
     load**; the overloads sit on a hard bottleneck cut. => GEANT is structurally
     capacity-limited, NOT a training/k problem.
2. **Retrained GEANT on the flexible band** (loads 1.0/1.2/1.4 instead of the old
   1.0/1.5/2.0 — half of which were capacity-limited noise). 3 seeds, 500k steps,
   ~5 min each. Models: results/geant_qos_flex_seed{0,1,2}/.
3. **ns-3 QoS eval at load 1.4** (multi-seed, stressed-feasible stratification).
4. **Fixed a 3000x perf bug** + two eval bugs (see below).

## Key result (full ns-3, GEANT, load 1.4, 3 model seeds) — results/ns3_eval_geantStress_seed{0,1,2}/
| Regime | Metric | OSPF | GNN | Verdict |
|--------|--------|------|-----|---------|
| OVERLOAD (1005@145%, 1008@104%) | loss  | 5.95% | ~5.54% (5.42/5.56/5.64) | ~tie, slight edge |
| OVERLOAD | delay | 90.9ms | ~87.8ms (83.0/92.7/87.7) | ~tie, slight edge |
| STRESSED-FEASIBLE (~90% util) | loss | 0.30% | 0.30% | tie |
| STRESSED-FEASIBLE | delay | 20.5ms | 20.7ms | tie |

- Best single case (model seed 0 on heavily overloaded 1005): loss 10.5→9.4%,
  delay 140→124ms. But seed variance: model seed 1 REGRESSES on borderline 1008
  (1.4→2.0% loss). Honest summary: **tie with a small, noisy edge.**
- Decisive insight: even at 90% analytical util, ns-3 loss is near-zero (0.19–0.39%)
  for BOTH OSPF and GNN — the traffic still fits, so there's nothing to win.

## Bugs fixed today
1. **KSP 3000x slowdown** (marl_routing/gnn_routing_agent.py): `compute_ksp` did
   `list(shortest_simple_paths(...))[:k]` — materializing EVERY simple path before
   taking k. ~0.8s/pair on dense GEANT, ~400s per 506-pair precompute. Fixed with
   lazy `itertools.islice` -> **400s -> 0.13s**. (This is why the original
   diagnostic ran 49 min and produced nothing.) Silently taxed all env builds.
2. **evaluate_ns3.py IndexError**: 4 GEANT node-pairs have <3 simple paths; the env
   pads (repeat last) but `gnn_paths_for` used unpadded compute_ksp -> index overrun.
   Fixed by clamping the chosen index to len(pair_paths[pi])-1.
3. **evaluate_ns3.py stratification**: feasible band took the LIGHTEST seeds (28–41%
   util) — too trivial to show any routing effect. Changed to the MOST-STRESSED
   feasible seeds (~90% util), where rerouting could matter.
4. **train_gnn_qos.py NameError**: post-train check referenced undefined LOAD_FACTOR;
   now uses TRAIN_LOADS[-1] (heaviest trained load).

## Thesis status — BOTH topologies done
- **Abilene** (uniform 10G, well-connected): GNN STRICTLY DOMINATES under congestion
  (~10x loss cut, ~3x delay cut, tight error bars), ties when feasible. [prior work]
- **GEANT** (mixed 2.5G/10G, bottleneck cuts): GNN TIES OSPF — routing headroom is
  structurally ≤5pt even optimally; capacity-limited overloads are unfixable by any
  routing. GNN never catastrophically regresses.
- **Headline:** learned routing's benefit is topology-dependent; we show WHEN/WHY it
  helps with a positive case + a boundary case, both validated in ns-3.

## Git state (NOT yet committed — for tomorrow)
Modified: evaluate_ns3.py, marl_routing/gnn_routing_agent.py, run_ns3_phase2.py,
train_gnn_qos.py, ns3_scenarios/abilene-validate/abilene-validate.cc
New: geant_headroom.py, CHECKPOINT_20260616.md, results/ns3_eval_geantStress_seed*/
(NOTE: large results/ dirs — check .gitignore before committing; only commit code +
small JSON summaries + this checkpoint, not raw ns-3 state dumps.)

## Next session (to discuss)
1. **Decide commit scope** then commit + push today's fixes and the GEANT result.
2. **Write-up: Abilene-vs-GEANT contrast** — this is the core thesis narrative now.
3. **Comparison figures**: side-by-side GNN-vs-OSPF on both topologies by regime
   (loss + delay), plus the GEANT headroom plot (OSPF vs greedy@k3/5/8) that proves
   capacity-limited.
4. Optional: is a ~tie on GEANT worth one more angle (e.g. lower load where a clean
   win exists), or report the honest boundary case as-is? (Recommend: report as-is —
   it's the stronger scientific story.)
