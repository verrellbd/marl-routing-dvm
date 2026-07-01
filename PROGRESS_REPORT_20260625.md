# Progress Report — MARL for Network Routing

**Student:** Dean Ariefin · **Date:** 25 June 2026 · **For:** supervisor meeting

---

## 1. One-paragraph summary

The system is **built, working end-to-end, and producing defensible results on real
data**. I have a decentralized multi-agent RL routing controller (one agent per router,
custom MAPPO) and a centralized single-agent GNN controller, both evaluated against the
deployed OSPF baseline. Across **three real backbone networks with real measured dynamic
traffic**, judged at the **packet level in ns-3**, both learned controllers beat OSPF under
congestion; the decentralized MARL is competitive with — and on GÉANT beats — the
centralized controller, with the coordination cost appearing only on the largest (50-node)
network. The result is **multi-seed robust**. The remaining work is mainly write-up.

---

## 1a. Progress since last week

**Last week** the topologies and traffic were still **synthetically generated** (a gravity
traffic model and reconstructed/approximate topologies — generated with Claude). **This
week** the entire study has been moved onto **real resources from SNDlib**
(https://sndlib.put.poznan.pl/home.action): real topologies, real link capacities, and
real measured **dynamic** traffic matrices. All final results now rest on real data only;
the synthetic results are retired.

The other major step this week: the **decentralized MARL controller (CTDE / MAPPO) is now
built and evaluated** — last week the MARL formulation had not been started.

---

## 1b. Status by thesis component

| Thesis component | Last week | This week | Note |
|---|---|---|---|
| Evaluation pipeline (ns-3 + analytical) | Done | ✅ Done | Train-fast, judge-in-ns-3 |
| GNN routing backbone (novel element) | Done | ✅ Done | Single-agent, PPO |
| Topologies + traffic | Done (synthetic gravity) | ✅ **Done — now REAL (SNDlib)** | 3 real nets incl. Germany50 (50n); real dynamic traffic; temporal split |
| QoS-aware reward | Done (v1) | ✅ Done | −max-util − hop penalty (see §2a) |
| Baseline: OSPF | Done | ✅ Done | Sole baseline (scope decision) |
| MARL formulation (CTDE / MAPPO) — RQ1 | **Not started** | ✅ **Done** | Per-node agents, custom MAPPO, evaluated on all 3 nets |
| Dynamic-conditions eval — RQ3 | Not started | ✅ Largely done | Real dynamic traffic + temporal train/test split; link-failure injection still to do |
| Reward design study — RQ2 | Partial | 🟡 Partial | One reward family tested; not yet swept |

---

## 2. What is done

**System**
- Decentralized MARL controller: per-node agents, hop-by-hop forwarding, custom MAPPO
  (centralized training / decentralized execution). This is the thesis's central system.
- Centralized single-agent GNN controller (PyTorch-Geometric backbone — the one novel
  modelling element), serving as the strong central upper-reference.
- OSPF baseline.
- Fast analytical training surrogate + full ns-3 packet-level evaluation harness
  (exact per-flow paths installed as static routes, FlowMonitor for loss/delay/throughput).

**Data — now entirely real (no synthetic/reconstructed data in final results)**
- Topologies, capacities, delays from SNDlib: Abilene (12n), GÉANT (22n), Germany50/DFN (50n).
- Real measured **dynamic** traffic: Abilene-Zhang (5-min, 6 months), GÉANT-Uhlig/TOTEM
  (15-min, 4 months), Germany50-DFN (5-min, 1 day).
- **Temporal train/test split** (train on first 70% of the timeline, test on last 30%) —
  every number is on *unseen real traffic from a later period*. Selection is deterministic.

**Results (packet-level ns-3, real traffic, overload regime)**

| Network | Metric | OSPF | central GNN | decentralized MARL |
|---------|--------|------|-------------|--------------------|
| Abilene | loss | 2.32% | **0.17%** | **0.18%** |
| Abilene | delay | 37.5 ms | **11.7 ms** | **12.6 ms** |
| GÉANT | loss | 7.46% | 1.44% | **0.86%** |
| GÉANT | delay | 17.9 ms | 15.5 ms | **14.9 ms** |
| Germany50 | loss | 3.16% | **0.03%** | 0.96% |
| Germany50 | delay | 17.4 ms | **1.9 ms** | 17.6 ms |

- Both learned methods beat OSPF under congestion on all three networks (loss cut ~5–100×).
- Decentralized MARL ≈ centralized GNN on small/mid networks; **coordination cost shows at
  50 nodes** (MARL controls loss but pays delay without a global view).

**Results — feasible regime (traffic fits; offered load < capacity)**

| Network | Metric | OSPF | central GNN | decentralized MARL |
|---------|--------|------|-------------|--------------------|
| Abilene | loss | 0.18% | 0.16% | 0.18% |
| GÉANT | loss | 0.12% | 0.14% | 0.15% |
| Germany50 | loss | 0.12% | 0.03% | 0.03% |

- When traffic fits, all methods **tie** (loss ≈ 0.1–0.2% everywhere) — by design: the
  QoS-aware reward keeps the learned policies on short paths when there is no congestion to
  relieve, so they only detour under overload. The gain is therefore concentrated exactly
  where it matters (overload) and costs nothing in the common feasible case.

**Reward in use right now**

Both controllers use one QoS-aware reward family:

> reward = −(marginal increase in max link-utilisation) − λ·(extra / detour hops)

- The first term telescopes over an episode to −(final bottleneck utilisation), so
  maximising return = minimising the worst-loaded link (the classic TE objective).
- The second term is a delay/QoS penalty (λ = 0.5 on Abilene/GÉANT, 0.1 on Germany50): it
  penalises hops beyond the shortest path, so the policy stays on OSPF-short paths when
  uncongested and only detours to relieve a bottleneck.

This is a single reward family — RQ2 (a systematic reward-design sweep) is still open.

**Multi-seed robustness (3 model seeds, overload max link-utilisation, mean ± std)**

| Network | OSPF | central GNN | MARL |
|---------|------|-------------|------|
| Abilene | 122% | 67 ± 2% | 64 ± 2% |
| GÉANT | 126% | 92 ± 2% | 97 ± 7% |
| Germany50 | 109% | 86 ± 2% | 99 ± 7% |

Both beat OSPF across every seed; the coordination-cost-at-scale pattern is robust.

---

## 3. Key findings to discuss

1. **Decentralization is near-free on small/mid networks** and becomes costly only at
   scale — a clean, honest characterization of *when* a distributed control plane is worth
   it, rather than an unconditional "MARL wins."
2. **The gain is concentrated in the overload regime.** When traffic fits, learned routing
   correctly declines to detour and ties OSPF. This is the intended QoS-aware behaviour.
3. Two methodological findings caught along the way: a simulation knob (rateScale) that
   preserves loss/utilisation but distorts delay (must be pinned across networks), and a
   millisecond-truncation bug in the ns-3 delay model that had zeroed Germany50's sub-ms
   link delays. Both are now controlled.

---

## 4. Plan for next week

1. **Improve Germany50** — the 50-node network is the one case where decentralized MARL
   pays a delay cost vs the centralized controller. Investigate and push for a better MARL
   result there (e.g. tune the per-node policy / λ, revisit the stretch constraint,
   possibly inter-agent signalling).
2. **Quantify the MARL-vs-OSPF trade-off** — turn the headline comparison into a clear,
   quantified trade-off across the three networks (gain under congestion vs cost/parity
   when feasible; the decentralization-vs-scale relationship).
3. **Start writing the research** — begin the dissertation write-up from the existing draft
   (`WRITEUP.md`): methodology + results chapters, figures, related work, citations.

**Questions for supervisor**
- Is OSPF a sufficient baseline, or should I prioritize the LP upper bound before writing?
- Scope check: is the single novel element (GNN backbone) within decentralized MARL the
  right framing, or should the emphasis shift more toward the MARL coordination story?
- Preferred thesis structure / any required chapter layout for the write-up.

---

*Supporting material: `WRITEUP.md` (draft chapters), `RESULTS_SUMMARY.md` (detailed
results + reproducibility), figures in `results/fig_real3way_*.png` and
`results/fig_multiseed_overload.png`.*
