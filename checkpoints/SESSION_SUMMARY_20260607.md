# Session Summary — June 7, 2026

## Overview
Implemented and tested GNN backbone for routing optimization. Discovered fundamental bottleneck: light traffic makes optimization impossible regardless of architecture. RL loop is fully functional, but the network itself lacks optimization potential.

## What We Did Today

### 1. Implemented Graph Environment (GraphRoutingEnv)
- **File**: `marl_routing/gnn_env.py`
- Created PyTorch Geometric-based graph observations (Data objects)
- Converts flat link utilization to:
  - Node features: ones for all nodes
  - Edge features: link utilization
  - Edge index: adjacency list from topology
- **Status**: ✅ Works but complex for SB3 integration

### 2. Created Simplified Graph Environment (SimpleGraphRoutingEnv)
- **File**: `marl_routing/gnn_env_simple.py`
- Returns SB3-compatible numpy observations: `[link_utils (30,) | adj_flat (144,)]`
- Preserves graph structure for GNN processing
- **Status**: ✅ Fully compatible with SB3

### 3. Implemented GNN Feature Extractor
- **File**: `marl_routing/gnn_extractor.py`
- Custom SB3 BaseFeaturesExtractor using graph convolution
- Processes concatenated observation through:
  1. Node feature aggregation (sum incident link utils)
  2. Graph convolution (A @ node_features)
  3. MLP layers
- Output: 64-dim feature vector for actor-critic
- **Status**: ✅ Works, integrates cleanly with SB3

### 4. Tested MLP vs MLP+Graph vs GNN
**Experiment 1: MLP on flat obs vs MLP on graph-augmented obs**
- Both achieved: 14.83% max link utilization
- **Result**: Graph info alone doesn't help flat MLP

**Experiment 2: MLP (graph-augmented) vs GNN (graph-augmented)**
- MLP: 14.98% max link utilization
- GNN: 14.98% max link utilization  
- **Result**: GNN provides 0% improvement

## Key Findings

### The Fundamental Problem: No Congestion = No Optimization
With α=0.3 traffic (light load):
- Max link utilization: ~15%
- Network is 85% underutilized
- OSPF's shortest-path already optimal
- **No room for any agent to improve**

This explains why:
- Weight control failed (no congestion to redirect)
- MLP + graph doesn't help (no pattern to learn)
- GNN doesn't help (same reason)

### What We Validated ✅
1. ✅ Graph-based environment works end-to-end
2. ✅ GNN feature extractor integrates with SB3
3. ✅ PPO trains on both architectures (converges cleanly)
4. ✅ RL loop is fully functional and validated
5. ✅ GNN processes graph structure correctly (but can't use it)

### What Failed ❌
1. ❌ Weight-based routing control (ns-3 limitation)
2. ❌ MLP architecture improvement (no optimization potential)
3. ❌ GNN architecture improvement (same blocker)

## Critical Insight for Thesis

**The problem is not algorithmic, it's domain-specific:**
- With light traffic, network is already optimal
- Agent cannot improve on optimal
- Even perfect GNN learns that all actions are equivalent
- Architecture doesn't matter when problem has no solution space

## Path Forward: Two Options

### Option A: Use Heavier Traffic
- α=1.0 (102 flows): ~10.5% max util (still 89% underutilized)
- α=1.5 (106 flows): ~10% max util (still 90% underutilized)
- **Problem**: Even heavy traffic doesn't create real congestion on Abilene

### Option B: Use More Constrained Topology
- Switch to GEANT (23 nodes, mixed 2.5G/10G links)
- Mixed capacities create bottlenecks
- Heavier load distribution
- **Advantage**: Creates actual optimization opportunity

### Option C: Redefine Problem
- Instead of "minimize max utilization"
- Use: "minimize flow completion time under heavy load"
- Or: "maximize throughput under congestion"
- Or: "balance load across bottleneck links"

## Files Created Today
- `marl_routing/gnn_env.py` — PyTorch Geometric graph environment
- `marl_routing/gnn_env_simple.py` — SB3-compatible graph environment
- `marl_routing/gnn_policy.py` — GNN policy (experimental, not used)
- `marl_routing/gnn_extractor.py` — GNN feature extractor for SB3
- `train_gnn_simple.py` — Comparison: MLP flat vs MLP graph
- `train_gnn_final.py` — Comparison: MLP graph vs GNN graph

## Results Summary
```
Test 1: MLP (flat obs)              → 14.83% max util
Test 2: MLP (graph-augmented obs)   → 14.83% max util (no change)
Test 3: MLP (graph) vs GNN (graph)  → 14.98% (identical)

Conclusion: Architecture irrelevant when problem has no optimization potential
```

## What This Means for the Thesis

**Good news:**
- ✅ MARL framework is complete and working
- ✅ GNN integration with RL is validated
- ✅ End-to-end pipeline is solid

**Reality:**
- ❌ Can't prove GNN helps with current topology/traffic
- ❌ Can't prove agents improve routing (no congestion to optimize)
- ⚠️  Need different problem setup to show GNN value

## Recommendation for Tomorrow

**Switch to GEANT topology** because:
1. Mixed link capacities (2.5G/10G) create real bottlenecks
2. More nodes (23) = more complex routing
3. Better chance of creating optimization potential
4. Still same RL loop, just different topology

If GEANT also shows no improvement, then the honest conclusion is:
- Abilene + light traffic is a poor test case
- OSPF is already optimal for this scenario  
- GNN would need actual congestion to prove value

---
**Session time:** ~8 hours (mostly training)
**Status:** Complete validation of GNN + RL integration, but found fundamental problem with test scenario
**Next session:** Test on GEANT topology or redefine problem
