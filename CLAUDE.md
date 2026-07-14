cat > ~/thesis/CLAUDE.md << 'EOF'
# MARL for Network Routing — Master's Thesis

## Project
MARL-based routing optimization on an SDN platform, evaluated against OSPF and
LP-optimum under dynamic traffic conditions, with QoS-aware reward design.
Novel angle: GNN as the agent backbone (one novel element — keep scope disciplined).

## Environment (VERIFIED WORKING — do not reconfigure without reason)
- PRIMARY MACHINE (as of 2026-07): monaco.ee.ucl.ac.uk — 40 CPU cores (4x Xeon
  E5-4620 v4 @2.1GHz), 512GB RAM, Rocky 9, CPU-ONLY (no GPU). Dedicated (not shared)
  -> use for training + eval. Cores are OLD/slow (~5x slower per-core than malmo) but
  plentiful; parallelize across cores. Our work needs NO GPU (tiny nets, CPU surrogate,
  CPU ns-3), so monaco is the right box and doesn't waste shared GPU machines.
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
  ns-3: raise per-sim timeout via NS3_TIMEOUT env (run_ns3_phase2.py reads it; default 900).
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

### In Progress
- 🔄 Phase 3: PPO training (50k timesteps on GPU 1, ~30-60 min, started 18:50 UTC)
  * Goal: Beat 13.6% baseline (α=0.3)
  * Checkpoints saved to results/models/
  * TensorBoard logs to results/tb_logs/

### Next (Deferred for later)
- Phase 2B: Upgrade to proper ns3-ai gym interface (protobuf, not urgent)
- Phase 4: MAPPO multi-agent coordination (after PPO validation)
- Phase 5: GNN backbone (the novel element, requires MAPPO working first)

## Scope decisions
- LP-optimum baseline SKIPPED. Research question is "MARL vs traditional routing";
  OSPF + ECMP are the deployed baselines. No theoretical ceiling, so we compare to
  what's actually used. Revisit if time permits (~half a day with CVXPY).

## Topology choice
- Start on Abilene (12 nodes, 15 links, all 10G). Move to GEANT (23 nodes, 39 links,
  mixed 10G/2.5G) once Abilene results are clean. Both already built.

## Working conventions
- Research design questions: discuss, don't just implement.
- Be honest where MARL loses to OSPF/MPLS-TE (predictability, convergence).
- One novel element only. Push back on scope creep.
- Decision interval 1-2s simulated; episode 60-120s.
- Topologies: Abilene (12 nodes) then GEANT (24 nodes). Traffic: gravity model.
EOF

cat ~/thesis/CLAUDE.md