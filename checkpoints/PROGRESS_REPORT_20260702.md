# Progress Report — MARL for Network Routing

**Student:** Dean Ariefin · **Date:** 2 July 2026 · **For:** supervisor meeting
**Compares against:** last week's report (25 June 2026)

---

## 1. One-paragraph summary

Last week the system was complete on real data and the one open weakness was Germany50,
where the decentralized MARL lagged the centralized GNN at 50 nodes. This week I
**diagnosed why and built a fix** (a hop cap on the agents' forwarding), and — importantly
— **ran it honestly across 3 seeds instead of 1**. The result is more nuanced than a single
seed suggested: the fix **reliably solves the delay problem** at 50 nodes but **loss stays
high-variance** — so decentralization's cost at scale is now precisely characterized as
*unreliability*, not a fixed gap. I also added run logging + model checkpointing and tidied
the repository.

---

## 2. Last week's plan vs. what got done

Last week I set three goals for this week. Status:

| Goal (set 25 Jun) | Status | Outcome |
|---|---|---|
| **1. Improve Germany50** | ✅ Done (with honest caveat) | Diagnosed the cause; hop-cap fix reliably solves *delay*, partly improves *loss* (high-variance) |
| **2. Quantify the MARL-vs-OSPF trade-off** | ✅ Done | Clear per-network verdict + a diagnosed mechanism for *why* MARL wins small/mid and loses reliability at scale |
| **3. Start writing the research** | 🔜 Next | Deferred one week — the Germany50 result changed, so I locked the science down first (correct call) |

---

## 3. What changed this week (the delta)

**Germany50 — from "open weakness" to "diagnosed + partially fixed"**

*Last week:* MARL at 50 nodes = 0.96% loss / 17.6 ms delay (vs centralized GNN 0.03% / 1.9 ms)
— reported as an unexplained coordination cost.

*This week:*
- **Diagnosed the cause** (measured, not guessed): with only local information, per-node
  agents wandered into long detours (up to **12 hops** vs OSPF's 3–4) that *concentrated*
  load onto shared links, leaving one saturated (~105%). The delay driver was **queueing
  (congestion), not path length** — Germany50's links are short.
- **Built a fix:** a **hop cap** on the agents' forwarding (final path ≤ shortest + 4 hops).
  This constrains only the agent's route choice — **the topology is unchanged**.
- **Tested it honestly across 3 training seeds** (this is the key rigor step vs last week's
  single seed).

**Germany50 result — hop-capped MARL, packet-level, 3 seeds:**

| | OSPF | centralized GNN | MARL (mean ± std) |
|---|---|---|---|
| Overload loss | 3.16% | 0.03% | 2.78 ± 2.56% |
| Overload delay | 17.4 ms | 1.9 ms | **8.2 ± 5.4 ms** |

Per seed: loss 0.03% / 2.10% / 6.20%; delay 2.2 / 7.1 / 15.2 ms.

**Honest reading:** the hop cap **reliably fixes delay** (every seed beats OSPF, and beats
the 17.6 ms uncapped MARL), but **loss is high-variance** — one seed matches the centralized
GNN, another is *worse* than OSPF. **The coordination cost at 50 nodes is real and shows up
as variance/unreliability, not a mean gap.** The centralized GNN remains the reliable choice
at scale.

> **Correction of last week's mid-week claim:** an interim seed-0-only result briefly looked
> like "MARL matches the centralized GNN at 50 nodes." The 3-seed run shows that is true only
> for the best seed. This is corrected in all documents — reported here for transparency.

---

## 4. The MARL-vs-OSPF trade-off, now quantified

Across the three real networks, the picture is now clear and mechanism-backed:

| Network | Size | MARL vs OSPF | MARL vs centralized GNN |
|---------|------|--------------|-------------------------|
| Abilene | 12 nodes | ✅ big win (loss 13×, delay 3×) | ✅ matches |
| GÉANT | 22 nodes | ✅ best method (loss 9×) | ✅ **beats** it |
| Germany50 | 50 nodes | ⚠️ delay wins; loss unreliable | ❌ loses (high-variance) |

**Headline:** decentralized, local-only routing **matches or beats a central controller up
to ~22 nodes**, and at 50 nodes becomes a **reliability trade-off**. *Why:* as the network
grows, each router's local view covers less of it, paths lengthen so local errors compound,
independent local fixes collide onto shared links, and a shared reward makes credit
assignment harder — all degrade together. The centralized GNN sidesteps all four via a
global view. This is a clean, defensible boundary result (exactly the "be honest where MARL
loses" the project calls for).

---

## 5. Also this week (engineering / rigor)

- **Run logging** → `logs/` (every training run captured to a timestamped log).
- **Model checkpointing** → `checkpoints/` (periodic snapshots during training).
- **Repository tidy** — old checkpoint notes and logs moved into `checkpoints/` and `logs/`.
- All results docs (RESULTS_SUMMARY §2/§2a), figures, and memory updated to the honest
  multi-seed numbers.

---

## 6. Plan for next week

1. **Write the research** (now the priority — the science is locked and honest). Draft the
   methodology + results chapters from the existing `WRITEUP.md`.
2. (optional) **Inter-agent communication** — the natural fix for Germany50's residual loss
   variance: let neighbouring agents exchange link-load signals so they stop colliding on
   shared bottlenecks. Directly targets the diagnosed failure mode; good research extension.
3. (optional) LP / best-response upper bound → state results as "X% of optimal."

**Questions for supervisor**
- Is the honest Germany50 result (delay fixed, loss high-variance) framed the right way, or
  should I push harder on closing the loss variance (inter-agent comms) before writing?
- Any required thesis chapter structure to follow for the write-up?

---

*Supporting material: `RESULTS_SUMMARY.md` (§2 multi-seed, corrected), figures
`results/fig_real3way_germany50.png` + `results/fig_germany50_multiseed.png`,
`checkpoints/CHECKPOINT_20260701.md`, `WRITEUP.md` (draft chapters).*
