# MARL for Network Routing — Master's Thesis

## Project
MARL-based routing optimization on an SDN platform, evaluated against OSPF and
LP-optimum under dynamic traffic conditions, with QoS-aware reward design.
Novel angle: GNN as the agent backbone (one novel element — keep scope disciplined).

## Environment (VERIFIED WORKING — do not reconfigure without reason)
- PRIMARY MACHINE (as of 2026-07-27): **geneva.ee.ucl.ac.uk** — 128 threads
  (AMD EPYC 7543, Zen 3), 512GB RAM, CPU-ONLY. Typically at load ~15-30/128, i.e. mostly
  FREE, and much faster per-core than monaco. ns-3 + the venv are verified working here
  (binaries built on monaco run fine — no AVX-512 on either, so no illegal-instruction
  problem). Measured on geneva: abilene/geant ns-3 sim ~1 min, germany50 ~17 min;
  single-agent 1.5M-step training ~3h; MARL 366 updates ~40 min.
- MOVED OFF monaco.ee.ucl.ac.uk (4x Xeon E5-4620 v4 @2.1GHz, 80 threads): old/slow cores,
  and it is now SHARED and often SATURATED (load ~81/80). A single 1.5M training run took
  6h+ there vs ~3h on geneva. Do not use for big batches.
- Other good hosts if geneva is busy, best first: berlin (EPYC 9455, 96c), malmo
  (Xeon Gold 6426Y), turin, london. ALWAYS `uptime` + `nproc` first: an idle slow box beats
  a saturated fast one. Rule of thumb: load < 0.3 x nproc = plenty free.
- Etiquette everywhere: `nice -n 19`, and cap threads (see Working conventions).
- HOME is NFS-shared (128.40.41.191:/vol_home_2/uceedv1) across ALL ee.ucl machines,
  so ~/thesis, the venv, ~/.claude (transcripts + memory) are identical everywhere.
  Switching machines keeps all context; `claude --continue` from ~/thesis resumes sessions.
- OLD machine: malmo.ee.ucl.ac.uk (2x Xeon Gold 6426Y, 512GB, 4x H100 96GB, SHARED).
  Faster per-core but shared/busy (hit thread-creation limits) + GPU-etiquette. Avoid for
  big CPU batches.
- Python 3.9 venv at ~/thesis/ns3ai-venv (auto-activates via .bashrc); verified imports OK
  on monaco. Torch 2.6.0+cu124 runs CPU-only fine (cuda unavailable -> graceful).
- PyTorch 2.6.0+cu124, PyG 2.6.1, Ray/RLlib 2.51.2, Stable-Baselines3 2.7.1
- ns-3.42 at ~/thesis/ns-3-dev. NOTE: ns-3 binaries built on malmo MAY fail on monaco's
  older cores (illegal instruction) — rebuild on monaco if needed:
  cd ns-3-dev && ~/thesis/configure_ns3.sh && ./ns3 build -j8. Also monaco is slow for
  ns-3: raise per-sim timeout via NS3_TIMEOUT env (run_ns3_phase2.py reads it; default 900
  is TOO LOW for germany50 200-flow sims — they take ~25min idle / ~45-60min under
  contention. Use NS3_TIMEOUT=3600 for germany50; 900 is fine for abilene/geant).
- ns3-ai at commit b8c9858 (main branch, NOT the v1.2.0 tag)
- Protobuf sourced from Anaconda (/opt/anaconda3), not system

## Critical environment quirks
- HOME quota is 50GB hard cap. Heavy outputs may need /scratch/uceedv1 later.
- ns-3 must be configured via ~/thesis/configure_ns3.sh (custom -D flags for
  pybind11, protobuf, venv python). A bare ./ns3 configure WILL fail.
- PYTHONPATH must include the gym-interface/py dir (set in .bashrc).
- Disabled ns3-ai examples (API drift with ns-3.42): rate-control, multi-bss.
  Do not re-enable them.
- Build with -j8 (not nproc) on shared machines. On monaco (dedicated) more is fine.
- Training/eval are CPU-ONLY (device="cpu" hardcoded in train_gnn_qos.py + mappo.py).
  No GPU selection needed. For parallel jobs cap threads (OMP_NUM_THREADS=2) so many run
  at once; see train_all_monaco.sh + eval_ns3_all_monaco.sh (18-model / 18-dir launchers).

## Sanity check (bridge works)
cd ~/thesis/ns-3-dev/contrib/ai/examples/a-plus-b/use-gym && python apb.py
# Expect "set: X,Y; get: Z;" repeating then "Experiment destroyed"

## Status

### Completed
- ✅ Environment setup, ns-3 + ns3-ai bridge verified end-to-end.
- ✅ Topology layer:
  * topologies/abilene.json: 12 nodes, 15 links, all 10 Gbps (canonical Abilene)
  * topologies/geant.json: 23 nodes, 39 links, mixed 10G/2.5G (European backbone)
  * marl_routing/topology.py: NetworkX loader, validates strong-connectivity, diameter
  * marl_routing/visualize.py: renders topology with lat/lon positioning
  * results/abilene_topology.png, geant_topology.png: visual reference
- ✅ Gravity-model traffic generator (marl_routing/traffic.py):
  * Generate N×N demand matrix from log-normal node weights (realistic hot spots)
  * Load factor sweep: α=0.3 (light), 1.0 (moderate), 1.5 (heavy)
  * Filtering by min-flow threshold to keep ns-3 tractable
    - 62 flows ≥50 Mbps (83% of demand)
    - 102 flows ≥30 Mbps (96% of demand)
    - 132 flows unfiltered (100% demand, ns-3 overhead too high)
  * Outputs JSON flows for ns-3 OnOff apps
- ✅ ns-3 baseline scenario (scratch/geant-ospf/geant-ospf.cc):
  * Reads topology JSON + traffic JSON (single source of truth)
  * Instantiates full flow set as UDP OnOff applications
  * OSPF baseline: GlobalRouting with refBW/linkBW metric
  * ECMP option: --ecmp=true toggles RandomEcmpRouting
  * Per-link bidirectional utilization + flow stats output
  * Tested: 10 flows run cleanly with zero loss, 2.49% max util

### Completed (Continued)
- ✅ Comprehensive baseline comparison (OSPF vs ECMP load sweep):
  * Tested three flow thresholds: min 100 Mbps (20 flows), min 50 Mbps (62 flows), min 30 Mbps (102 flows)
  * Load factors: α=0.3 (light), α=1.0 (moderate), α=1.5 (heavy)
  * Results: OSPF and ECMP perform identically across all scenarios
    - α=0.3: ~5.28% max link util (38-102 flows)
    - α=1.0: ~10.47% max link util (102 flows)
    - α=1.5: ~10.90% max link util (106 flows)
  * All scenarios: zero packet loss, stable delivery
  * Key insight: Abilene topology is so well-connected that shortest-path routing
    naturally load-balances as efficiently as ECMP
  * This is a strong baseline — any MARL improvement is meaningful

- ✅ Phase 2A: Gym environment with interactive ns-3 bridge
  * Python gym env ↔ action.json ↔ ns-3 sim ↔ state.json
  * Interactive loop: each gym step = 1 second ns-3 simulation
  * Tested: 5 steps run successfully, rewards computed from real utilization
  * α=0.3 (light traffic, 38 flows) baseline: ~13.6% max link util

> NOTE: everything above this line is EARLY-PHASE HISTORY (gravity traffic, per-topology
> models, α load factors). It is kept for provenance. The CURRENT experiment is described
> below and supersedes it. Do not resume "Phase 3 PPO" — that line of work is finished and
> was superseded twice (per-topology GNN -> MARL -> topology-agnostic MARL).

## CURRENT STATE (2026-07-28)

Branch `sndlib-tmgen-matched` holds the main result. Branch `ns3-ospf-baseline` is an
in-flight side quest (see below).

**The experiment.** Train ONE topology-agnostic policy on 17 SNDlib topologies with TMgen
modulated-gravity traffic; test ZERO-SHOT on abilene/geant/germany50 with real measured
traffic. Five arms, all scored on identical matrices:
OSPF | ECMP | single-agent GNN | MARL h32 | MARL h64 (hidden=64 = capacity-matched).
Matched budget = 1.5M env steps each, 3 seeds each.

**!! THIS BLOCK IS SUPERSEDED — see "FINAL PACKET-LEVEL GRID" below. !!**
The numbers that used to sit here (abilene 7.19->0.62%, geant 14.13->3.65%, germany50
18.94->11.60%, "germany50 headroom at NO QoS cost", "ECMP is WORSE than OSPF on germany50")
came from `results/matched_ns3_grid.json`, which was measured with CAPACITY-BLIND learned
paths and equal-HOP ECMP. Three of those four claims did not survive re-measurement.
**Use `results/final_ns3_grid.json` for everything. Do not quote matched_ns3_grid.json.**
What still holds from the old block: MARL is the STABLE method and the single agent is
seed-fragile (germany50 spread +/-33.9 vs h32's +/-3.5) — and that now holds under
IDENTICAL PPO hyperparameters, so the fragility is architectural, not a settings artifact.

**!! RESULT REVERSAL 2026-07-28 — ABILENE IS A LOSS. Do not report the old abilene win. !!**
Measured with a correctly configured OSPF (cost = refBW/linkBW), on all 18 abilene test
matrices: **weighted OSPF 57.25%** < MARL h64 70.64% < MARL h32 72.61% < single 88.99% <
ECMP 100.06% < hop-count OSPF 102.61%. Real OSPF BEATS every learned method by ~13pt, and
abilene has NO overload regime at all under it (0/18 matrices >=100%; the old "abilene
overload" was entirely an artifact of hop-count routing through the single 2.48G link).
WHY (this is the insight, not just a bug): abilene_sndlib is 14x9.92G + 1x2.48G. Weighted
OSPF gives that link cost 4 and avoids it. Everything else in the table is CAPACITY-BLIND in
path construction — hop-count OSPF walks into it, ECMP sprays onto it, and our learned
policies choose among HOP-COUNT k-shortest paths with hop-based detour limits, so the slow
link sits in their candidate set. Honest framing: on heterogeneous-capacity topologies a
capacity-aware baseline beats capacity-blind optimisation, however good the optimiser.
WHAT SURVIVES: geant + germany50 have UNIFORM 40G links, so weighted == hop-count there
(verified identical, 145.5%). ALL geant/germany50 results — including the whole packet-level
grid and the germany50 headroom finding — STAND UNCHANGED. The core claim holds on the two
larger topologies and fails on the 12-node toy.
PACKET-LEVEL CONFIRMATION (2026-07-28, 126 sims, 0 fails, `results/ns3m_*_abilenew_s*`,
loads 16/22/28, `--metric weighted` on all exporters). Abilene with a CORRECT baseline:
  loss %  OVERLOAD  OSPF 14.35 | ECMP 14.30 | MARLh64 14.37 | MARLh32 16.53 | single 18.29
  loss %  FEASIBLE  OSPF  0.17 | ECMP  0.18 | MARLh64  1.61 | MARLh32  1.67 | single  2.75
  delay   FEASIBLE  OSPF 11.45ms vs learned 36.7-42.1ms  (3-4x WORSE)
So on abilene the learned methods are at BEST tied (MARLh64 in overload) and clearly worse
everywhere else — the detours cost delay without buying congestion relief. Report abilene as
a NEGATIVE RESULT. `_abilenew_` dirs are the correct ones; the old `ns3m_*_abilene_s*` dirs
use the hop-count straw man and must NOT be reported.
MECHANISM PROVEN (2026-07-28): the slow link is arc (1,5)/(5,1) @2480 Mbps. Mean utilisation
OF THAT LINK across the 18 abilene matrices: hop-count OSPF **83.6%**, weighted OSPF
**0.0%** (it avoids it entirely), MARL h64 **58.0%**. Fast links sit at only ~21-24% in all
three, so there is ample spare capacity — weighted OSPF simply moves everything off the slow
link. MARL learned to PARTIALLY relieve it (83.6 -> 58.0) but cannot eliminate it, because
its k=3 candidates are HOP-shortest (for some pairs all 3 traverse the slow link) and its
hop-based stretch limit forbids the longer capacity-avoiding route.
RE-MATCH AT HIGHER LOAD DOES NOT SAVE IT: at loads 16/22/28 (weighted OSPF genuinely
congested, 129.3% on its overload subset) the learned methods are still far worse —
MARL h64 148.9% (-19.6pt), h32 155.4%, single 195.9%. The failure is STRUCTURAL
(capacity-blind candidate paths), not a load-distribution artifact.
NOTE ECMP IS ALSO WRONG ON ABILENE: real ECMP splits among equal-cost paths under OSPF's
METRIC; ours used equal-HOP-COUNT paths. Re-baselining abilene must fix both OSPF and ECMP.
**CAPACITY-AWARE FIX — IMPLEMENTED 2026-07-28, RETRAIN IN FLIGHT.** The deeper fix was
taken. `--metric weighted` now threads through the ENVIRONMENTS, not just the exporters:
- `marl_routing/ospf_metric.py`: added `dist_to_all()` (Dijkstra on OSPF cost) + `edge_cost()`.
- `compute_ksp(..., weight="w")`: candidate k-shortest paths are now COST-shortest.
- `graph_routing_env._TopoBundle(metric=)`: weighted KSP, weighted `ospf_arc_paths`,
  equal-COST next-hops in `ecmp_max_util` (float costs -> 1e-9 tolerance, not `== dist-1`).
- `topo_agnostic_marl_env._MARLBundle(metric=)`: `dist_to` is COST distance, so the agent's
  progress test and the reward's `is_detour` are capacity-aware. `_valid()` now accumulates
  `self.cur_cost` in cost units instead of counting hops — the hop-based stretch limit was
  the thing forbidding the capacity-avoiding route.
- `--metric` flag on both trainers; exporters pass theirs into the env too.
DEFAULT REMAINS "hop" everywhere, so every committed result keeps its old meaning.
PRE-LAUNCH VERIFICATION (do not re-derive):
  abilene   slow link in k=3 candidates 112/396 -> 50/396; random policy max util 80.4 -> 40.2
  geant     identical under both metrics (uniform 40G) — confirms nothing else can regress
  germany50 max util identical 69.9; OSPF ref 28.6 -> 28.2 (equal-cost tie-break only)
RETRAIN: `train_capaware.sh`, 9 runs (single | MARL h32 | MARL h64) x 3 seeds, matched
1.5M steps, tags `_singleH64gcap` / `_tier2m15cap` / `_tier2m15h64cap`. Hyperparameters
were RECOVERED FROM THE SAVED POLICIES, not guessed, so `--metric` is the only variable.
Old capacity-blind dirs stay on disk for the before/after table.
**!! ABILENE NEGATIVE RESULT IS REVERSED (2026-07-28, 126 ns-3 sims, 0 fails). !!**
`results/ns3m_*_abilenecap_s*` — the SAME saved policies, re-exported with capacity-aware
candidates (no retraining; the policies are topology-agnostic and consume whatever the env
gives them). Compare `_abilenew_` (capacity-blind learned paths) -> `_abilenecap_`:
  OVERLOAD loss%   OSPF 14.35 | ECMP 14.30 | single 18.29->14.11 | h32 16.53->15.06 |
                   h64 14.37->**13.62**   (h64 delay 119.9->102.1ms vs OSPF 122.6ms)
  FEASIBLE loss%   OSPF  0.17 | single 2.75->0.17 | h32 1.67->0.20 | h64 1.61->0.20
  FEASIBLE delay   OSPF 11.45ms | single 42.10->11.40 | h32 36.68->12.09 | h64 36.93->12.11
  FEASIBLE maxutil OSPF 96.7 | single 100.1->**91.6** | h32 98.5->94.5 | h64 100.0->96.5
THE OLD CONCLUSION "detours cost delay without buying congestion relief" WAS AN ARTIFACT OF
CAPACITY-BLIND CANDIDATE PATHS. Delete it. The 3-4x delay penalty is gone (11.4-12.1ms vs
OSPF 11.45), and the single agent frees 5.1pt of headroom at identical QoS.
HONEST READING: this is PARITY + a headroom gain, NOT dominance. Feasible loss 0.20 vs OSPF
0.17 is marginally worse (noise); h32 still trails OSPF in overload (15.06 vs 14.35).
Report abilene as "capacity-blind failure diagnosed and fixed", not as a win.
STILL TO DO: re-run geant/germany50 with `--metric weighted` env candidates and rebuild
`results/matched_ns3_grid.json`. Expect NO change there (uniform 40G -> weighted == hop),
but verify rather than assume.

## !! SINGLE-AGENT REWARD FORM PROMOTED (2026-07-30) — READ BEFORE QUOTING ANY SINGLE NUMBER
The reported single agent is now `results/single_singleH64gRM_seed*` (`--reward-form marl`),
NOT `_singleH64gcap`. The old arm is kept in the grid as `single_whole_ablation`.
WHY: the two arms' delay penalties differed in MEANING though not in value (both beta=0.5).
Single charged beta per EXTRA HOP and normalised it with the congestion term; MARL charges a
flat beta per DETOUR HOP unnormalised. Measured consequence: under the old form the
cost-shortest candidate was myopically optimal for 95.2% of geant / 97.2% of germany50
demands (median congestion benefit of deviating = 0.000, median penalty 1.011), and the
policy reproduced OSPF on 90.3% of geant demands.
PACKET-LEVEL EFFECT (54 sims, 0 fails, `results/ns3f_singleRM_*`), old -> new:
  abilene FEASIBLE loss 3.69 -> **0.24** | delay 37.84 -> **13.77**ms | util 92.5 -> **88.1**
  abilene OVERLOAD loss 21.64 -> **13.41** (now BEST in cell: OSPF 14.35, ECMP 14.30, h32 15.41)
  geant   OVERLOAD loss 13.29 -> 11.38 | g50 FEASIBLE util 74.8 -> **61.5** at OSPF's 0.03 loss
  REGRESSION: g50 OVERLOAD loss 16.73 -> 18.17 (the one cell it loses)
CONSEQUENCES FOR THE WRITE-UP (already applied to paper/):
- **ABILENE IS NO LONGER A BLANKET NEGATIVE.** The single agent BEATS OSPF in both abilene
  regimes. MARL still does not. Correct claim: "learned routing wins at 12 nodes, the
  hop-by-hop decentralised formulation does not". Section renamed accordingly.
- **h32 vs single is now an EVEN 3-3 SPLIT** across the 6 cells (was 5-1). MARL takes both
  geant cells + g50 overload; single takes both abilene cells + g50 feasible. Do not claim
  MARL dominates.
- Seed fragility STILL holds and is stronger: single g50 analytical +/-51.4 vs h32 +/-3.5.
  But on abilene the single agent is the TIGHTER arm (+/-0.82 vs +/-2.13) — state both.
- MARL's advantage SURVIVES the identical reward: geant overload +8.5pt, geant feasible
  +27.7pt util, g50 overload +7.6pt. The architecture conclusion does not reduce to reward.
NOT TESTED (stated as a limitation): whether MARL would also improve under the single
agent's whole-reward form. Only the centralized arm was re-run.

## FINAL PACKET-LEVEL GRID (2026-07-28) — `results/final_ns3_grid.json`
378 ns-3 sims, 0 failures, `results/ns3f_*`. Capacity-aware (`--metric weighted` in the ENV)
+ matched hyperparameters. Arms: single `_singleH64gcap`, h32 `_tier2m15cm`, h64
`_tier2m15h64cm`. THIS SUPERSEDES `matched_ns3_grid.json` AND ALL `ns3m_*` DIRS.
Abilene ran at loads 16/22/28 (weighted OSPF has no overload regime below ~16), so its
packet row does NOT correspond to its analytical row (8/12/16). geant/germany50 do.

  geant OVERLOAD    loss%  OSPF 14.13 | ECMP 6.82 | single 13.29 | h32 **2.92** | h64 3.56
  geant FEASIBLE    util%  OSPF 97.3  | ECMP 78.2 | single 95.8  | h32 **66.7** | h64 68.2
                    (h32 loss 0.12 vs OSPF 0.16, delay 8.33 vs 9.46 — headroom is FREE here)
  g50   OVERLOAD    loss%  OSPF 27.29 | ECMP 22.90 | single 16.73 | h32 10.62 | h64 **8.88**
  g50   FEASIBLE    util%  OSPF 95.6  | single 74.8 | h32 70.8 | h64 61.9
                    BUT loss  OSPF 0.03 | single **0.03** | h32 1.02 | h64 2.44
                    -> SINGLE AGENT WINS THIS CELL (OSPF's loss, 20.8pt more headroom).
                       MARL's extra headroom here is BOUGHT WITH LOSS. Report the trade.
  abilene FEASIBLE  loss%  OSPF **0.17** | h32 0.97 | h64 3.30 | single 3.69
                    delay  OSPF **11.45**| h32 14.17| h64 24.80| single 37.84
  abilene OVERLOAD  loss%  OSPF 14.35 | ECMP 14.30 | h32 15.41 | h64 14.57 | single 21.64

**ABILENE IS STILL A NEGATIVE RESULT — the reversal recorded below applies ONLY to the OLD
capacity-blind-trained policies re-exported with capacity-aware candidates (`_abilenecap_`).
The RETRAINED matched arm is WORSE on abilene (h64 feasible loss 0.20 -> 3.30).** The
analytical grid predicted this independently (h64 abilene 61.6 +/-6.5 vs OSPF 57.2), so both
measurements agree. The matched hyperparameters that helped geant/germany50 hurt abilene.
Do not quote `_abilenecap_` as the final abilene result.

**ECMP FINDING FLIPPED.** Real equal-COST ECMP beats OSPF in BOTH overload cells (geant 6.82
vs 14.13; germany50 22.90 vs 27.29). The old "ECMP is WORSE than OSPF on germany50 (23.40 vs
18.94)" was equal-HOP ECMP, a straw man. ECMP is a much stronger baseline than we reported.

MARL h32 > h64 on geant + abilene and has ~1/3 the seed spread; h64 wins only germany50.
The results chapter currently treats h64 as headline — change it to h32.

## Hyperparameter matching (audited 2026-07-28)
ENV/DATA SIDE IS ALREADY FULLY MATCHED between single-agent and MARL — same 17 train topos,
same TMgen call (n_patterns=3, load_scales 0.6-1.5, same seed), same max_flows=500 filter,
same test matrices (real_matrices n_per_scale=6 split=test), delay_penalty 0.5,
normalize_reward True, eval seed A.seed+1. Nothing to fix. Do not re-audit this.
PPO SIDE had four mismatches (lr/gae_lambda/clip/vf_coef/max_grad_norm/ent_coef were already
identical): gamma 0.99 vs 0.995, n_epochs 6 vs 10, minibatch 512 vs 256, buffer 4096 vs 2048.
DIRECTION OF THE FIX MATTERS: the single agent is the BASELINE, so it KEEPS SB3's defaults
untouched (trimming a baseline's optimisation to match ours would read as handicapping it);
OUR MAPPO moves to meet it. `train_marl_gnn_tier2.py` gained `--gamma/--n-epochs/--minibatch`
for this. Launcher: `train_matched_hparams.sh`, tags `_cm`. The earlier `_cap` MARL runs
(n_epochs 6, gamma 0.99) stay on disk as the "MARL at its own defaults" ablation.
NOT MATCHABLE, state as a limitation: one single-agent step routes a WHOLE FLOW, one MARL
step routes ONE HOP, so 1.5M steps = ~6x more flows routed for the single agent. Matched on
env steps, unmatched on work. Report a second budget axis (flows routed) and show the
conclusion is invariant. NOTE this mismatch favours the SINGLE AGENT (it also got 3.3x more
gradient steps under the old settings), so MARL winning anyway is not an under-training
artifact — but SB3's 10 epochs over a 2048 buffer could itself explain the single agent's
seed collapse, which is why the matched re-run is worth having.

**KNOWN DEFECT — being fixed (this is what caused the reversal above).** Our OSPF baseline routes by
HOP COUNT (`nx.shortest_path` with no weight), but real OSPF uses cost = refBW/linkBW.
abilene_sndlib has ONE 2.48G link among fourteen 9.92G links, so hop-count OSPF routes
straight through it and looks ~45pt worse than a correctly configured OSPF
(102.6% -> 57.2% mean offered load). **Every abilene gain is currently measured against a
straw-man baseline and may not survive the fix.** geant/germany50 have uniform 40G links,
so weighting changes nothing there — but germany50 swings 92pt on equal-cost TIE-BREAKING
alone, so its OSPF number is fragile too. Fix = weighted Dijkstra in the exporters +
re-run the OSPF arm (~1h of sims).

## Next steps (ranked, after long deliberation)
1. **Fix the OSPF metric** — DONE for the baselines (exporters), and the deeper
   capacity-aware fix to the ENVIRONMENTS is implemented + retraining (see above).
   Remaining: re-run the ns-3 grid weighted once the 9 runs land.
1b. **Specialist baseline** (decided 2026-07-28): KEEP the 17-topology zero-shot design as
   the headline — it is the contribution the survey identifies as unmet (GDDR/GROM fail to
   beat per-topology retraining). Add per-topology specialists at the SAME 1.5M budget as a
   REFERENCE arm to quantify the generalization gap (the ROAR-style comparison nobody
   reports). Needs a `--topos` flag on `train_marl_gnn_tier2.py` (currently hardcoded
   `TRAIN_TOPOS`). ~9 runs, ~3h parallel on geneva. NOT a replacement for the 17.
   The existing `*_sndlib_marl_real_seed*` dirs are an OLDER env/budget — not comparable.
2. **Finish the ns3-ospf branch** — see below. Citability, not correctness.
3. **Failure robustness**: env already supports `fail_links` (currently 0). Evaluate the
   SAVED policies zero-shot with 1 link failed. No retraining, ~2h. Biggest claim-per-hour.
4. **Within-episode traffic shifts.** The project premise says "dynamic traffic" but each
   ns-3 run holds ONE static matrix; we test generalization ACROSS matrices, not adaptation
   WITHIN an episode. State this as a limitation or close it.
5. Update the results chapter: two written conclusions changed (germany50 MARL variance;
   "no benefit in the feasible regime" was abilene-specific and is FALSE at 50 nodes).
6. Greedy-k3 oracle as a cheap upper reference (no ceiling in the table currently).

## ns-3 evaluation: what is real and what is not
- **Reported loss/delay/throughput/utilisation are ns-3 packet-level.** Training and the
  reward stay on the analytical surrogate (ns-3 is far too slow in a training loop).
- **Analytical ">100%" is OFFERED LOAD, not utilisation.** ns-3 utilisation SATURATES at
  100% (excess becomes loss). So: report utilisation in the FEASIBLE regime, loss/delay in
  OVERLOAD. Never present offered load as measured utilisation.
- **ECMP exists in two flavours, deliberately.** Analytical `ecmp_max_util` splits each
  demand fractionally (fluid). ns-3 has (a) our flow-level hashed ECMP —
  `export_ecmp_routes.py`, one equal-cost path per flow, what real routers do — and
  (b) ns-3 native `RandomEcmpRouting` (per-packet spraying) via `--routing=ecmp`, which we
  ADDED to `abilene-validate.cc`. They are different policies; label them separately.
- **`abilene-validate.cc` installs per-flow static host routes**, which is how a learned
  policy's exact path gets into ns-3. Consequence to state in the thesis: those routes are
  source-routed and would need MPLS/segment routing to deploy — plain destination-based IP
  forwarding cannot express them.
- germany50 at loads 35/50/65 is ALL overload (OSPF 207-220%); its feasible regime needs
  loads <=30 and lives in separate `…_g50feas_…` dirs.

## ns3-ospf side quest (branch `ns3-ospf-baseline`)
WHY: ns-3 ships no OSPF, so "we used a real OSPF implementation" is more citable than
"we installed shortest paths". WHAT IT BUYS: control-plane realism (Hello/LSA/convergence).
WHAT IT DOES NOT BUY: it does NOT fix the metric defect above, and our evaluation is static
(one matrix per sim, no link failures), so in steady state real OSPF converges to exactly
the shortest paths we already install. Treat as nice-to-have, not a blocker.
- Module: github.com/markverick/ns3-ospf, pinned `c56950f` (2026-03-02), in `vendor/ns3-ospf`,
  symlinked to `ns-3-dev/contrib/ospf`. Actively maintained; Application-based; installs
  routes into Ipv4StaticRouting.
- **PORT DONE 2026-07-28 and it WORKS** — far cheaper than feared. Only TWO changes were
  needed; all 39 upstream source files compile unmodified against ns-3.42:
  (1) `ospf/CMakeLists.txt` (upstream ships only a waf wscript);
  (2) `model/ospf-app-sockets.cc` — Hello/LSA raw sockets bound to the OSPF multicast group
      224.0.0.5, and ns-3.42's `Ipv4RawSocketImpl::SendTo` asserts
      `GetInterfaceForAddress(m_src) >= 0`, which fails for a multicast group (ns-3.35 did
      not check). Bind to `Any` instead; `Connect()` still sets the multicast destination
      and receive filtering is by IP protocol 89 + `BindToNetDevice`.
  Verified: `scratch/ospf-smoke` (= upstream ospf-four-nodes) converges, all LSDBs
  synchronised with correct adjacencies at t=99.
- BOTH changes are saved in `ns3_patches/` (patch + CMakeLists + setup README) because
  `ns-3-dev/` is gitignored. Re-apply from there after any ns-3 reinstall.
- NEXT for this branch: write `scratch/ospf-validate` — our topology JSON + OspfAppHelper on
  every node with interface metric = refBW/linkBW, let it converge, then run the same traffic
  and measure. That gives a real-OSPF baseline AND fixes the hop-count metric defect at the
  same time. Compare against the static shortest-path OSPF arm: in steady state they should
  match if the metrics agree — if they do, that VALIDATES the static approach for the thesis.

## REPRODUCIBILITY GAP (fix this)
`ns-3-dev/` is in .gitignore and has NO git of its own. Our `abilene-validate.cc` changes
(the ECMP mode) and the ns3-ospf CMake port are therefore UNVERSIONED. Track the scratch
scenario sources + the CMake port inside the thesis repo, with a setup script that clones
the pinned vendor commit.

## Scope decisions
- LP-optimum baseline SKIPPED. Research question is "MARL vs traditional routing";
  OSPF + ECMP are the deployed baselines. No theoretical ceiling, so we compare to
  what's actually used. Revisit if time permits (~half a day with CVXPY).

## Topology choice
- SUPERSEDED. We no longer train per-topology. Training = 17 SNDlib topologies
  (`*_sndlib.json`); testing = zero-shot on abilene/geant/germany50 SNDlib variants with
  real measured traffic. The old hand-built `abilene.json`/`geant.json` (gravity traffic)
  belong to the early phase. NOTE `abilene_sndlib` is NOT all-10G: 14x9.92G + 1x2.48G.
- Topology Zoo (245 topos, `ingest_topzoo.py`, `vendor`-free, in `topologies/zoo_*.json`)
  was tried and is KEPT ONLY as an honest negative: more topological diversity did not
  help; germany50 degraded out-of-distribution. Not the main story.

## Working conventions
- Research design questions: discuss, don't just implement.
- Be honest where MARL loses to OSPF/MPLS-TE (predictability, convergence).
- One novel element only. Push back on scope creep.
- Always report the honest negative alongside the win (ECMP beating us somewhere, a seed
  collapsing, a regime where learned routing adds loss). Several findings this project
  REVERSED on more data — treat single-seed/short-budget conclusions as provisional.
- Parallelism gotcha: cap `OMP_NUM_THREADS=1` (and MKL/OPENBLAS) when running many python
  jobs at once. 24 uncapped torch procs took geneva to load 275/128 and died with
  MemoryError + segfaults.
- Long jobs: run with `nohup nice -n 19`, write a `.done` marker, and watch for the marker
  rather than polling a process.