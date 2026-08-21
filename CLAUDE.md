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
- Training/eval are CPU-ONLY (device="cpu" hardcoded in train_single_tier2.py,
  marl_routing/mappo.py + marl_routing/marl_gnn.py, and the exporters/aggregators that
  load a policy: export_topoagn_routes.py, fill_offered_grid.py, evaluate_ns3.py).
  No GPU selection needed. For parallel jobs cap threads (OMP_NUM_THREADS=1) so many run
  at once; the CURRENT launchers are train_seeds10.sh + eval_seeds10.sh (2 arms x 10 seeds
  / 558 sims). The old 18-model monaco launchers were deleted on 2026-08-21.

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
  * topology figures were deleted 2026-08-21; regenerate with marl_routing/visualize.py
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

## CURRENT STATE (2026-08-21)

Branch `seeds-10` holds the main result. Branch `ns3-ospf-baseline` is an in-flight side
quest (see below).

**The experiment.** Train ONE topology-agnostic policy on 17 SNDlib topologies with TMgen
modulated-gravity traffic; test ZERO-SHOT on abilene/geant/germany50 with real measured
traffic. FOUR reported arms, all scored on identical matrices:
OSPF | ECMP | single-agent GNN (`_singleH64gRM`) | MARL h32 (`_tier2m15cm`).
Matched budget = 1.5M env steps each, **10 seeds** each.
A MARL h=64 arm existed only as the width-selection sensitivity check. It is not reported
anywhere, and its runs were deleted on 2026-08-21 (commit 3b6067d).
`results/width_selection.json` keeps the summary that chose h=32; the underlying h64 runs
are gone, so `fill_offered_grid.py` now finds 0/10 policies for its third arm and silently
skips it. The reported `offered_grid.json` never contained h64, so no reported number moves.

## Capacity-aware routing (`--metric weighted`)
`--metric weighted` threads OSPF cost (refBW/linkBW) through the ENVIRONMENTS, not just the
exporters. This is live code, not history:
- `marl_routing/ospf_metric.py`: `dist_to_all()` (Dijkstra on OSPF cost) + `edge_cost()`.
- `compute_ksp(..., weight="w")` (`marl_routing/gnn_routing_agent.py`): candidate
  k-shortest paths are COST-shortest, not hop-shortest.
- `graph_routing_env._TopoBundle(metric=)`: weighted KSP, weighted `ospf_arc_paths`,
  equal-COST next-hops in `ecmp_max_util` (float costs -> 1e-9 tolerance, not `== dist-1`).
- `topo_agnostic_marl_env._MARLBundle(metric=)`: `dist_to` is COST distance, so the agent's
  progress test and the reward's `is_detour` are capacity-aware. `_valid()` accumulates
  `self.cur_cost` in cost units instead of counting hops.
- `--metric` flag on both trainers; exporters pass theirs into the env too.
DEFAULT REMAINS "hop" everywhere; every reported run passes `--metric weighted` explicitly.

WHY IT MATTERS — the abilene mechanism, still the explanation for the abilene negative:
abilene_sndlib is 14x9.92G + 1x2.48G. The slow arc is (1,5)/(5,1) @2480 Mbps, OSPF cost 4.
Mean utilisation OF THAT ARC across the 18 abilene matrices: hop-count OSPF **83.6%**,
weighted OSPF **0.0%** — it avoids the link entirely. Fast links sit at only ~21-24%, so
there is ample spare capacity and weighted OSPF simply routes around the slow link. The
learned policies only PARTIALLY relieve it, because for some pairs all k=3 candidates
traverse it. Honest framing: on heterogeneous-capacity topologies a capacity-aware baseline
beats capacity-blind optimisation, however good the optimiser. geant + germany50 are uniform
40G, so weighted == hop-count there (verified: identical path COSTS; the paths that differ
are equal-cost tie-breaks only).

## SINGLE-AGENT REWARD FORM (2026-07-30) — READ BEFORE QUOTING ANY SINGLE NUMBER
The reported single agent is `results/single_singleH64gRM_seed*` (`--reward-form marl`), NOT
the older `_singleH64gcap` whole-reward arm (deleted 2026-08-21; it is also no longer in the
grid as `single_whole_ablation`).
WHY: the two arms' delay penalties differed in MEANING though not in value (both beta=0.5).
Single charged beta per EXTRA HOP and normalised it with the congestion term; MARL charges a
flat beta per DETOUR HOP unnormalised. Measured consequence of the old form: the
cost-shortest candidate was myopically optimal for 95.2% of geant / 97.2% of germany50
demands (median congestion benefit of deviating = 0.000, median penalty 1.011), and the
policy reproduced OSPF on 90.3% of geant demands. Unifying the reward form is what makes the
single agent a fair baseline, and it STRENGTHENED that baseline.

**!! THE 3-SEED CONCLUSIONS THIS BLOCK USED TO DRAW DID NOT SURVIVE 10 SEEDS. !!**
Re-checked 2026-08-21 against `results/final_ns3_grid.json`:
- "The single agent BEATS OSPF in both abilene regimes" — **FALSE**. Abilene feasible loss
  1.01 vs OSPF 0.17, delay 21.55 vs 11.45; overload loss 17.81 vs 14.35, delay 147.0 vs
  122.6. OSPF wins both regimes on loss AND delay. The single agent is better only on
  feasible UTILISATION (91.4 vs 96.7).
- "h32 vs single is an EVEN 3-3 SPLIT" — **FALSE**. It is **5-1 to MARL** (overload scored
  by loss, feasible by utilisation). Single takes only abilene feasible, and there only on
  utilisation while losing the same cell on loss (1.01 vs 0.56).
- "on abilene the single agent is the TIGHTER arm" — **FALSE**. MARL h32 has the smaller
  seed spread in ALL SIX cells at 10 seeds.
- "single abilene overload is BEST in cell" — **FALSE**. Best is ECMP 14.31, then OSPF
  14.35, then h32 14.90, then single 17.81.
WHAT SURVIVES: MARL's advantage is not an artifact of the reward form. Under the IDENTICAL
reward it still wins geant overload, geant feasible, g50 overload and g50 feasible.
NOT TESTED (state as a limitation): whether MARL would also improve under the single agent's
whole-reward form. Only the centralized arm was re-run.

## FINAL PACKET-LEVEL GRID — `results/final_ns3_grid.json`
558 ns-3 sims, 0 failures, `results/ns3f_*` = 3 arms x 10 seeds x 18 matrices, plus the 18
deterministic OSPF runs. Capacity-aware (`--metric weighted` in the ENV) + matched
hyperparameters. Arms: single `_singleH64gRM`, h32 `_tier2m15cm`, ECMP `_ecmp`.
Abilene ran at loads 16/22/28 (weighted OSPF has no overload regime below ~16), so its
packet row does NOT correspond to its analytical row (8/12/16). geant/germany50 do.
OSPF is deterministic given the matrix, so it carries no seed spread and was simulated once.

  geant OVERLOAD    loss%  OSPF 14.13 | ECMP 7.52 | single 11.23 | h32 **3.36**
  geant FEASIBLE    util%  OSPF 97.3  | ECMP 79.4 | single 90.5  | h32 **66.3**
                    (h32 loss 0.12 vs OSPF 0.16, delay 8.50 vs 9.46 — headroom is FREE here)
  g50   OVERLOAD    loss%  OSPF 27.29 | ECMP 20.68 | single 14.94 | h32 **10.63**
  g50   FEASIBLE    util%  OSPF 95.6  | ECMP 71.6 | single 69.6 | h32 **65.5**
                    BUT loss  OSPF 0.03 | single **0.03** | h32 1.09
                    -> SINGLE AGENT WINS THIS CELL ON QoS: OSPF's own loss with 26.0pt more
                       headroom. h32 buys 4pt more headroom and pays 1.09% loss for it.
                       Report the trade, not just the headroom.
  abilene FEASIBLE  loss%  OSPF **0.17** | ECMP 0.22 | h32 0.56 | single 1.01
                    delay  OSPF **11.45**| ECMP 12.82 | h32 13.66 | single 21.55
                    util   OSPF 96.7 | ECMP 96.8 | single **91.4** | h32 92.4
  abilene OVERLOAD  loss%  ECMP **14.31** | OSPF 14.35 | h32 14.90 | single 17.81
                    delay  h32 **112.8** | ECMP 116.3 | OSPF 122.6 | single 147.0

**ABILENE IS A NEGATIVE RESULT FOR BOTH LEARNED ARMS** on loss and delay — OSPF or ECMP wins
every abilene QoS cell. The only abilene gain is feasible-regime headroom (single 91.4,
h32 92.4, against OSPF 96.7). The mechanism is the slow-link section above.

**ECMP IS A STRONG BASELINE.** Real equal-COST ECMP beats OSPF in both overload cells
(geant 7.52 vs 14.13; germany50 20.68 vs 27.29) and takes abilene overload loss outright at
14.31. Any older note claiming ECMP is worse than OSPF used equal-HOP ECMP, a straw man.

MARL h32 is the reported decentralised arm: it wins 5 of the 6 head-to-head cells against
the single agent and is the tighter arm on seed spread in all six.

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

**DEFECT FOUND AND FIXED — kept because it explains why the abilene result moved.** The
OSPF baseline originally routed by HOP COUNT (`nx.shortest_path` with no weight), but real
OSPF uses cost = refBW/linkBW. abilene_sndlib has ONE 2.48G link among fourteen 9.92G links,
so hop-count OSPF routed straight through it and looked ~45pt worse than a correctly
configured OSPF (102.6% -> 57.2% mean offered load) — every early abilene "gain" was
measured against a straw man, and none of them survived. geant/germany50 have uniform 40G
links, so weighting changes nothing there, though germany50 does swing on equal-cost
TIE-BREAKING alone. The fix is `--metric weighted` everywhere (see "Capacity-aware routing"
above); every reported run uses it, and all current numbers are post-fix.

## Session 2026-08-21 — cleanup + paper audit
Three commits changed what is on disk. Read this before looking for a file.
- `0f7beeb` deleted 27 superseded scripts (dead figure duplicates, early-phase analysis,
  the monaco launchers, the older trainers, the Topology Zoo code). KEPT despite looking
  orphaned: `make_heatmap.py` (produces Figure 1 of the paper), `run_ns3_phase2.py` and
  `evaluate_ns3.py` (the live exporters print them as their next step).
- `2a978ac` tracked the 20 training logs behind the training-curve figure. They had been
  UNTRACKED, i.e. the figure's only input data had no git backup.
- `3b6067d` deleted 516 results/logs entries (174M -> 74M). KEPT: the 120 reported
  `ns3f_{ecmp,marlh32,singleRM}_*` dirs, the 10+10 reported policies, the five reported
  JSONs, the 20 training logs, and the `.done`/`_jobs.txt` run records. DROPPED beyond the
  superseded set, at explicit request: the h64 arm, the `_singleH64gcap` whole-reward arm,
  and the Topology Zoo artifacts. Restore anything with `git checkout 2a978ac -- <path>`.
- `787c5e8` corrected seven numbers in `paper/`. The audit that found them checked EVERY
  figure in the paper against the current grids: **165/165 values in the three result
  tables already matched** — the defects were isolated. What changed: a stale 3-seed GEANT
  loss in the Conclusion (2.9 -> 3.4); the abilene-overload "best method" (OSPF -> ECMP,
  14.31 < 14.35); the single-agent parameter count (268,420 -> 100,740, measured from the
  saved policy, which narrows the capacity gap from ~15x to ~5.7x); the work-budget
  multiplier (six times -> three times, measured 3.32 hops/demand by rolling the trained
  policy over the 17 training topologies); the training edge range (21 -> 18, polska);
  three training-convergence figures; and a duplicated appendix block that had left
  `tab:reward-decomp` and `fig:training-curve` multiply defined.
DO NOT re-derive these — they are measured, and the measurement commands are in the commit
messages.

## Next steps (ranked, after long deliberation)
1. **Specialist baseline** (decided 2026-07-28): KEEP the 17-topology zero-shot design as
   the headline — it is the contribution the survey identifies as unmet (GDDR/GROM fail to
   beat per-topology retraining). Add per-topology specialists at the SAME 1.5M budget as a
   REFERENCE arm to quantify the generalization gap (the ROAR-style comparison nobody
   reports). Needs a `--topos` flag on `train_marl_gnn_tier2.py` (currently hardcoded
   `TRAIN_TOPOS`). ~9 runs, ~3h parallel on geneva. NOT a replacement for the 17.
   (The old `*_sndlib_marl_real_seed*` dirs were an OLDER env/budget and were deleted on
   2026-08-21 — there is nothing on disk to compare against; specialists must be trained.)
2. **Finish the ns3-ospf branch** — see below. Citability, not correctness.
3. **Failure robustness**: env already supports `fail_links` (currently 0). Evaluate the
   SAVED policies zero-shot with 1 link failed. No retraining, ~2h. Biggest claim-per-hour.
4. **Within-episode traffic shifts.** The project premise says "dynamic traffic" but each
   ns-3 run holds ONE static matrix; we test generalization ACROSS matrices, not adaptation
   WITHIN an episode. State this as a limitation or close it.
5. Greedy-k3 oracle as a cheap upper reference (no ceiling in the table currently).

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
- Topology Zoo (245 topos, in `topologies/zoo_*.json`) was tried and is kept ONLY as an
  honest negative: more topological diversity did not help; germany50 degraded
  out-of-distribution. Not the main story.
  **THIS NEGATIVE IS NO LONGER REPRODUCIBLE FROM THIS REPO.** Both halves were deleted on
  2026-08-21: the code (`ingest_topzoo.py`, `dump_zoo_traffic.py`, `train_marl_gnn_zoo.py`)
  in commit 0f7beeb, and the artifacts (`results/marlgnn_zoo_seed0/`) in commit 3b6067d.
  Only the 246 `topologies/zoo_*.json` files and this written claim survive. If the result
  ever needs defending, restore both halves first:
    `git checkout 0f7beeb~1 -- ingest_topzoo.py dump_zoo_traffic.py train_marl_gnn_zoo.py`
    `git checkout 2a978ac -- results/marlgnn_zoo_seed0`

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