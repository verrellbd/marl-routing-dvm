# Checkpoint — 2026-07-01

> **CORRECTION (2026-07-02):** the "matches the centralized GNN at 50 nodes" claim below is
> **seed-0 only**. Multi-seed ns-3 (seeds 0/1/2) showed the hop cap reliably fixes DELAY
> (every seed beats OSPF) but LOSS stays high-variance (s0 0.03%, s1 2.10%, s2 6.20% — s2
> worse than OSPF). Honest framing: coordination cost at 50n shows as VARIANCE, not a mean
> gap; central GNN is the reliable choice at scale. See RESULTS_SUMMARY §2 (corrected) +
> fig_germany50_multiseed.png. Text below is retained as the point-in-time seed-0 record.

## TL;DR
**Closed the Germany50 coordination gap.** Diagnosed why decentralized MARL lagged the
centralized GNN at 50 nodes, fixed it with a hop cap, and confirmed at the packet level:
MARL now MATCHES the centralized GNN on all three networks. Also set up run logging +
model checkpointing and tidied the repo (old checkpoints/logs into folders).

## The Germany50 fix (headline)
**Diagnosis (read-only, on real test matrices):** MARL's 50-node delay/loss gap
(0.96% loss / 17.6 ms vs SA-GNN 0.03% / 1.9 ms) came from per-node agents, on local info
only, wandering into long detours — up to **12 hops** (stretch 5–7), 17–64 flows/matrix.
These detours CONCENTRATED load, leaving a link ~105% saturated → queueing delay.
Key nuance: germany50 detour links are SHORT (~0.15 ms), so propagation barely changed —
**the delay driver was queueing (utilisation), not path length.**

**Fix = hard hop cap on the agents' forwarding** (final path ≤ shortest-hops + max_stretch).
Constrains only the agent's action mask; **topology is unchanged**. Implemented `max_stretch`
in `multiagent_routing_env._valid`; added `--max-stretch` to train_marl.py +
export_marl_routing.py.

**Tuning:**
- cap2 + λ0.25 (tag `_opt3`): too tight — killed wandering (stretch→2) but forced one
  matrix to 171% util (a flow genuinely needed a >2 detour). Rejected.
- **cap4 + λ0.1 (tag `_opt3b`): WINS.** Analytical util 95→86 mean, 107→**97 max (never
  overloads)**. Learned that MARL's util was already fine at λ0.1 — only the *worst*
  detours needed clipping, and cap=4 also improved util (acts as a load-concentration
  regulariser).

**ns-3 packet-level result (Germany50, real, overload, seed 0):**
| Metric | OSPF | SA-GNN | MARL orig | **MARL opt3b** |
|--------|------|--------|-----------|----------------|
| loss | 3.16% | 0.03% | 0.96% | **0.03%** |
| delay | 17.4 ms | 1.9 ms | 17.6 ms | **2.2 ms** |
Delay 8× lower, loss 30× lower → **matches the centralized GNN.** Same 5 test matrices,
rateScale=20, simTime=8, top-200 flows as the original run (fair). Model:
results/germany50_sndlib_marl_opt3b_seed0/. Figure: results/fig_real3way_germany50.png.

**Thesis impact:** the one honest weakness ("decentralization costs at 50 nodes") is now a
*diagnosed and fixed* failure mode — decentralized MARL matches centralized on ALL three
networks, with an interpretable mechanism (greedy over-detours at scale; bounded stretch
prevents it).

## Infra added today
- **Run logging** → `logs/<run>_<ts>.log` via a `_Tee` in train_marl.py (mirrors stdout).
- **Periodic model checkpoints** → `checkpoints/<run>/ckpt_upd*.pt` via new
  `ckpt_dir`/`ckpt_every` args on `MAPPO.learn`.
- **Repo tidy:** moved all `CHECKPOINT_*.md` + `SESSION_SUMMARY_*.md` → `checkpoints/`;
  all old `*.log` → `logs/`. Root now holds only active docs (CLAUDE, RESULTS_SUMMARY,
  WRITEUP, MARL_PLAN, NS3_PROVENANCE, SETUP_NOTES, PROGRESS_REPORT).

## In progress
- opt3b **seeds 1 and 2** retraining (background) for multi-seed confirmation of the
  Germany50 fix. When done: re-run analytical/ns-3 as needed and refresh the multi-seed
  numbers (currently §2a Germany50 MARL is the OLD uncapped 99±7).

## Files touched
marl_routing/multiagent_routing_env.py (max_stretch) · marl_routing/mappo.py (ckpt hooks) ·
train_marl.py (Tee log, ckpt dir, --max-stretch) · export_marl_routing.py (--max-stretch) ·
RESULTS_SUMMARY.md (§2 Germany50 + findings) · results/fig_real3way_germany50.png ·
memory real-data-sndlib.

## Next
1. Finish opt3b seeds 1/2 → confirm the fix holds across seeds; refresh §2a multi-seed
   (Germany50 MARL should tighten from 99±7 toward the SA-GNN band).
2. Then WRITE-UP (WRITEUP.md draft exists). The Germany50 diagnosis→fix is a strong
   methods+results story on its own.
3. (optional) quantify the MARL-vs-OSPF trade-off; LP upper bound.
