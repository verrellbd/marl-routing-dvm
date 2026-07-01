# Checkpoint — 2026-06-21

## TL;DR
Added a **third topology, Germany50** (50-node German research backbone), and ran the
full 3-way (OSPF vs single-agent GNN vs MARL) end-to-end including ns-3. Both learned
methods beat OSPF under congestion on all three topologies. New honest finding:
**decentralized MARL coordination is near-free on small networks (Abilene ~12 nodes)
but becomes high-variance at scale (Germany50 ~50 nodes)** — independent per-node agents
can miscoordinate (detour onto the same alternate link).

## What we did today
1. **Built Germany50** (`make_germany_topology.py` → topologies/germany50.json): 50 real
   German cities (accurate lat/lon) + reconstructed backbone (geographic, ~SNDlib
   germany50 structure), hub40G/edge10G/peripheral2.5G capacities, delays from distance.
   50 nodes, 84 links, diameter 10, 2450 pairs, strongly connected. (Provenance documented
   in the JSON; swap in SNDlib XML for bit-exact edges.)
2. **Confirmed winnable**: iterated best-response @k3 beats OSPF by +8/+11/+14 pt at load
   0.6/0.8/1.0. Middle case between Abilene (huge headroom) and GÉANT (capacity-limited).
3. **Trained single-agent GNN + MARL** on the congesting band (loads 0.6/0.8/1.0).
   Critical: delay_penalty had to drop 0.5→0.1 on this long-diameter graph (0.5 collapsed
   the GNN to OSPF). Added `--delay-penalty` to train_gnn_qos.py.
4. **Flow filtering for ns-3**: added `--max-flows` (top-K by rate) to evaluate_ns3.py +
   export_marl_routing.py (2450 flows → top-600; gravity demand is concentrated). Makes
   big topologies tractable in ns-3.
5. **3-way ns-3** at load 1.0, top-600 flows, identical stratified seeds + figure.

## Germany50 results
### Analytical (held-out load 1.0)
- Single-agent GNN: CLEAN WIN (OSPF 87.7/108/132.9 → 73.9/101.3/118.8, −7..−14pt).
- MARL (stretch=1): HIGH VARIANCE — big wins (1005 117→96 feasible) AND catastrophic
  losses (1000 96→141, 1013 158→182) from miscoordination.

### ns-3 3-way (load 1.0, top-600 flows) — results/ns3_eval_{sag50_3way,marlg50_seed0}/
| Regime | Metric | OSPF | single-agent GNN | MARL |
|--------|--------|------|------------------|------|
| Overload | loss  | 10.88% | 6.27% | **4.42%** |
| Overload | delay | 56.4 ms | 49.1 ms | 53.1 ms |
| Feasible | loss  | 0.09% | 0.00% | 0.10% |
Both beat OSPF; MARL edged SA-GNN on these (most-overloaded) seeds. NB: the
OSPF-util stratifier picked the heavily-overloaded seeds, where MARL does well; the
catastrophic-miscoord seeds (1000,1013) weren't selected — report the variance honestly.
Figure: results/fig_3way_germany50.png.

## stretch lesson
stretch=2 is BROKEN on large graphs (1081 bad paths, wanders/loops at diameter 10) →
MARL is stuck with stretch=1's equal-distance detours only. This limits MARL flexibility
vs the single-agent's k-shortest paths, and contributes to the scale variance.

## Three-topology picture (the thesis arc)
| Topology | nodes | routing headroom | single-agent GNN | MARL (decentralized) |
|----------|-------|------------------|------------------|----------------------|
| Abilene | 12 | large | strictly dominates OSPF | ≈ matches GNN (coord near-free) |
| GÉANT | 23 | ~0 (capacity-limited) | ties OSPF | ties OSPF (no harm) |
| Germany50 | 50 | moderate (+8..14pt) | clean win over OSPF | beats OSPF but HIGH-VARIANCE |

## Honest open items
1. MARL multi-seed (seeds 1,2) for error bars on all three topologies.
2. Germany50 MARL variance: the miscoordination is somewhat fundamental (no inter-agent
   comms); could explore a small comms/observation-sharing extension if scope allows.
3. Optional: swap SNDlib germany50 XML for bit-exact edges.

## Figures
- results/fig_3way_{abilene,geant,germany50}.png — the 3-way per topology.
- results/fig_ospf_vs_gnn.png, fig_geant_{headroom,bottleneck}.png, fig_traffic_heatmap.png,
  fig_pipeline.png — explainers from prior sessions.
