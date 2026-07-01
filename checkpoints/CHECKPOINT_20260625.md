# Checkpoint — 2026-06-25

## TL;DR
Multi-seed lock-down complete and the result is robust. Wrote the first prose draft of the
thesis (methodology + results + discussion) and a supervisor-facing progress report.
Nothing running; state clean. Next: refine the write-up.

## What we did today
1. **Confirmed multi-seed robustness** (from prior session, verified intact): 18 models
   trained (3 networks × {GNN, MARL} × seeds 0/1/2) on real data. Analytical overload
   max-util, mean ± std over seeds:
   - Abilene: OSPF 122 / SA-GNN 67±2 / MARL 64±2
   - GÉANT: OSPF 126 / SA-GNN 92±2 / MARL 97±7
   - Germany50: OSPF 109 / SA-GNN 86±2 / MARL 99±7
   Both beat OSPF across all seeds; SA tight (±2), MARL higher-variance on larger nets (±7)
   → coordination-cost-at-scale is robust. Artifacts: `eval_multiseed_analytical.py`,
   `results/multiseed_analytical.json`, `make_multiseed_fig.py`,
   `results/fig_multiseed_overload.png`. RESULTS_SUMMARY.md §2a.
2. **Wrote `WRITEUP.md`** — first thesis-chapter prose draft: RQ & contribution, data &
   networks (real-sources-only), method (sequential MDP, MARL/CTDE, GNN backbone,
   train-fast/judge-in-ns-3, parameter consistency), results (3-way + multi-seed),
   discussion (why MARL wins GÉANT but lags at 50n), caveats, reproducibility. Draws only
   on verified numbers.
3. **Wrote `PROGRESS_REPORT_20260625.md`** — clean supervisor-facing 1-pager: summary,
   what's done, key findings, next-week plan, questions for supervisor. For tomorrow's
   meeting.

## Current state of the result (the anchor numbers)
Real-data, seed-free, delay-corrected ns-3 3-way (overload regime, loss / delay):
- Abilene:   OSPF 2.32% 37.5ms | SA-GNN 0.17% 11.7ms | MARL 0.18% 12.6ms
- GÉANT:     OSPF 7.46% 17.9ms | SA-GNN 1.44% 15.5ms | MARL 0.86% 14.9ms (MARL best)
- Germany50: OSPF 3.16% 17.4ms | SA-GNN 0.03% 1.9ms  | MARL 0.96% 17.6ms (SA-GNN best)
Feasible regime: all ~tie. Figures `results/fig_real3way_*.png`.

## Files added today
- `WRITEUP.md` — draft thesis chapters.
- `PROGRESS_REPORT_20260625.md` — supervisor progress report.
- `CHECKPOINT_20260625.md` — this file.

## Nothing changed in code/models today (write-up day). No jobs running.

## Next session (priority order)
1. **Write-up** — refine `WRITEUP.md` into final thesis format: related work, citations,
   figures, requested chapter layout (confirm structure with supervisor first).
2. (optional) ns-3 multi-seed loss/delay error bars (currently seed-0 anchor; analytical
   bars already cover all seeds). Expensive on Germany50.
3. (optional) LP / best-response upper bound → "X% of optimal".
4. (optional) Substantiate §5 mechanism: correlate MARL delay gap with diameter/bottlenecks.
5. (research stretch) inter-agent comms to close MARL's Germany50 delay gap.

See also: `CHECKPOINT_20260623.md`, `RESULTS_SUMMARY.md`, memory `real-data-sndlib`.
