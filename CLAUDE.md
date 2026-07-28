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

**Headline results (all packet-level ns-3, 378 sims, 0 failures):**
- MARL beats BOTH baselines on loss in EVERY overload cell:
  abilene 7.19 -> 0.62%, geant 14.13 -> 3.65%, germany50 18.94 -> 11.60%.
- germany50 feasible: MARL h64 frees 19pt of headroom (util 96.9 -> 77.9) at no QoS cost.
- MARL is the STABLE method (seed spread ~4pt); the single-agent is seed-fragile
  (spread up to 62pt; 2 of 3 seeds collapse to shortest-path). This REVERSES the old
  "MARL miscoordinates at scale" finding, which was under-training.
- ECMP is WORSE than OSPF on germany50 in both regimes (23.40% vs 18.94% loss).
Grid: `results/matched_ns3_grid.json` (via `make_matched_grid.py`).

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
AFTER IT LANDS: re-run the ns-3 grid with `--metric weighted` and rebuild
`results/matched_ns3_grid.json`. Only then is the abilene verdict final — report whichever
way it falls.

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