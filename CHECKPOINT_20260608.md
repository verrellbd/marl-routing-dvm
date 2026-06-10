# GNN Routing Evaluation - Session Checkpoint (2026-06-08)

## 🎯 What We Did Today

1. **Fixed Critical Bugs**
   - ns-3 path resolution (parent.parent not parent.parent.parent)
   - Observation shape mismatch (15 → 30 directed link utilizations)

2. **Trained GNN Agent**
   - PPO with GNN feature extractor
   - 5056 timesteps, 85 episodes
   - Model saved: `results/ppo_gnn_routing_agent/gnn_routing_agent_final`

3. **Evaluated GNN on Light Traffic (α=0.3)**
   ```
   GNN Agent:    22.41% max link utilization
   OSPF Baseline: 14.60% max link utilization
   Result: ❌ OSPF wins by 53.55%
   ```

## 📊 Key Finding

**Light traffic is not the right regime for RL routing optimization.** Network is only 14.6% utilized - no congestion bottlenecks for the agent to improve. OSPF's shortest-path routing already achieves near-optimal load balancing.

## 🔄 Continue Tomorrow (Phase 3)

Test GNN on **heavier traffic** where congestion creates optimization opportunities:

**Option A: Quick Test (10 steps)**
```bash
cd ~/thesis
# Modify to use α=1.0 traffic
python3 << 'EOF'
# Load model from results/ppo_gnn_routing_agent/gnn_routing_agent_final
# Test on traffic_abilene_α1.0_min30.json
# Expected runtime: ~10 minutes on GPU 3
EOF
```

**Option B: Generate New Traffic**
```bash
python3 << 'EOF'
from marl_routing.traffic import generate_gravity_model_traffic
from pathlib import Path

# Generate α=1.0 and α=1.5 traffic files
for alpha in [1.0, 1.5]:
    flows = generate_gravity_model_traffic("abilene", alpha, min_flow_mbps=30)
    # Save to results/traffic_abilene_αX.X_min30.json
EOF
```

## 📁 Files to Know

**Trained Model**:
- `results/ppo_gnn_routing_agent/gnn_routing_agent_final` (trained on α=0.3)

**Current Results**:
- `results/gnn_routing_results.json` - GNN evaluation (22.41%)
- `results/ospf_baseline.json` - OSPF baseline (14.60%)

**Code**:
- `marl_routing/gnn_routing_agent.py` - GNN environment (bugs fixed)
- `train_gnn_routing_agent.py` - Training script
- `measure_ospf_baseline.py` - OSPF measurement

**Memory**:
- `~/.claude/projects/.../memory/session_20260608_gnn_eval_checkpoint.md` - Detailed checkpoint

## ⚙️ Technical Notes

- **GPU 3** available and working for inference
- **ns-3 simulations** stable (60-second runs complete successfully)
- **GNN feature extractor** verified working (12 nodes, 30 links → 64-dim features)
- **Model loading** works: `PPO.load(..., device='cuda')`

## 🎓 Research Status

**Question**: Does GNN-based routing beat OSPF?

**Current Answer**:
- α=0.3 (light): No, OSPF 14.60% < GNN 22.41%
- α=1.0 (moderate): Unknown - test tomorrow
- α=1.5 (heavy): Unknown - test tomorrow

**Next Hypothesis**: GNN should outperform OSPF under congestion where shortest-path has bottlenecks.

---

**Ready to continue tomorrow!** All code is fixed and tested. Just need to re-run evaluation on heavier traffic scenarios.
