# Checkpoint — 2026-06-18

## TL;DR
Completed the **genuine multi-agent RL** study end-to-end (phases M4 + M5). A
decentralized per-node MAPPO router — each node choosing the next hop from ONLY local
observations, no central controller — **beats OSPF under congestion in full ns-3**
(Abilene loss 21%→13%, delay 115→87 ms) and lands **within ~1 point of the centralized
single-agent GNN**. On capacity-limited GÉANT all methods tie and MARL does no harm.
Built the 3-way comparison + figures. Next topology: Germany/DFN.

## What we did today
1. **M4 — ns-3 validation of MARL** (`export_marl_routing.py`): roll out the agents'
   hop-by-hop paths → routing JSON (MARL path in the gnn_path slot) → run_ns3_phase2.py /
   abilene-validate.cc UNCHANGED. Real packet loss/delay measured.
2. **Reward fix**: penalise only DETOUR hops (non-progress / sideways moves = the extra
   hops over shortest), not every hop. So MARL keeps OSPF-short paths when uncongested
   and detours only to relieve congestion. delay_penalty=0.5 chosen (swept 0.05/0.5/1.0;
   0.5 best — env.step updated, train_marl default now 0.5).
3. **M5 — fair 3-way** (`make_3way_fig.py`): OSPF vs single-agent GNN vs MARL on
   IDENTICAL held-out seeds, both topologies.

## Key results (ns-3, held-out traffic, by regime)
### Abilene (load 4.0) — results/ns3_eval_{sa3way,marlv2_seed0}/
| Regime | Metric | OSPF | single-agent GNN | MARL (decentralized) |
|--------|--------|------|------------------|----------------------|
| Overload | loss  | 21.18% | 11.98% | **13.04%** |
| Overload | delay | 115.0 ms | 87.9 ms | **87.3 ms** |
| Feasible | loss  | 0.23% | 0.28% | 0.46% |
| Feasible | delay | 16.1 ms | 19.5 ms | 23.3 ms |

### GÉANT (load 1.4) — results/ns3_eval_{sageant3way,marlgeant_seed0}/
| Regime | Metric | OSPF | single-agent GNN | MARL |
|--------|--------|------|------------------|------|
| Overload | loss  | 5.95% | 5.42% | 6.37% |
| Overload | delay | 90.9 ms | 83.0 ms | 89.5 ms |
| Feasible | loss  | 0.28% | 0.28% | 0.39% |

**Story (3 tiers, honest):** OSPF < decentralized MARL ≲ centralized GNN. MARL beats
OSPF with LOCAL-only info (small coordination cost vs the centralized brain); on
capacity-limited GÉANT everything ties and MARL does no harm.

## Figures made today
- results/fig_3way_abilene.png — OSPF vs SA-GNN vs MARL, Abilene, by regime.
- results/fig_3way_geant.png — same for GÉANT (capacity-limited tie).

## MARL system (all phases done, seed-0)
- marl_routing/multiagent_routing_env.py (M1) — per-node hop-by-hop, stretch=1, loop-free,
  detour-only delay reward.
- marl_routing/mappo.py (M2) — ActorCritic (local actor + central critic) + MAPPO.
- train_marl.py (M3), export_marl_routing.py (M4), make_3way_fig.py (M5).
- Models: results/abilene_marl_v2_seed0/, results/geant_marl_v1_seed0/.
- RESULTS_SUMMARY.md §2b updated with the MARL 3-way.

## Honest open items (polish, not blockers)
1. **Multi-seed MARL** (seeds 1,2) for error bars — currently seed-0 only (single-agent
   has 3 seeds). Would tighten the Abilene feasible cost (one stubborn matrix, seed 1001).
2. delay_penalty sweep showed variance on one borderline feasible matrix — multi-seed
   averaging is the right fix, not more single-seed tuning.

## NEXT: third topology — Germany / DFN
- Goal: add a 3rd topology (the German research network, ~30–50 nodes) to test whether
  the Abilene-vs-GÉANT story generalises (well-connected win vs capacity-limited tie).
- NOTE: no germany/dfn topology data on the machine yet. Need a topology JSON
  (topologies/<name>.json) in the existing schema {nodes:[{id,name,city,lat,lon}],
  links:[{src,dst,capacity,delay}]}. Candidate real topologies: Germany50 (SNDlib, 50
  nodes/88 links, the canonical German TE benchmark) or a ~30-node DFN variant. Need to
  confirm exact topology + source (SNDlib / Topology Zoo) before building — don't
  fabricate node/link data.
- Once the JSON exists, the WHOLE pipeline is topology-agnostic: topology.py loads it,
  traffic.py + envs + train_gnn_qos.py + train_marl.py + ns-3 all take --topo <name>.
  Expect: build topology JSON → pick congesting load band → train single-agent + MARL →
  ns-3 3-way → figure. ~1 session.

## To resume
  cd ~/thesis
  # 1. obtain/confirm germany topology -> topologies/germany.json (existing schema)
  # 2. python -c "from marl_routing.topology import load; t=load('germany'); print(t.n_nodes)"
  # 3. find congesting load (ospf_max_util sweep) like geant_headroom.py
  # 4. train_gnn_qos.py --topo germany ... ; train_marl.py --topo germany ...
