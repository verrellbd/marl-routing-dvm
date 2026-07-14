# Checkpoint — 2026-07-14

## TL;DR
Moved primary machine to **monaco** (CPU-only, dedicated). Retrained all 18 models there and
re-ran the full ns-3 QoS on the fresh models → **complete fresh multi-seed results**. Cleaned
the repo of all superseded files. Ready for the write-up.

## Machine move
- **monaco.ee.ucl.ac.uk** is now primary: 40 CPU cores, 512GB, Rocky 9, **CPU-only**.
  Dedicated (not shared). ~5x slower per-core than malmo but many cores; our work needs no GPU.
- HOME is **NFS-shared** across all ee.ucl boxes → repo, venv, ~/.claude (transcripts + memory)
  identical everywhere. `claude --continue` from ~/thesis resumes sessions on any machine.
- CLAUDE.md updated (monaco primary, ns-3 rebuild/NS3_TIMEOUT notes, CPU-only training).

## What was done
1. **Retrained all 18 models on monaco** (parallel, ~25 min) via `train_all_monaco.sh`.
   Configs: abilene/geant MARL uncapped λ0.5 `_real`; germany50 MARL hop-capped
   (max-stretch 4, λ0.1) `_opt3b`; GNN λ per-net `_real`.
2. **Re-ran full ns-3 QoS on the fresh models** via `eval_ns3_all_monaco.sh`. Monaco is slow
   for ns-3: geant/germany50 timed out at 900s → re-ran with `NS3_TIMEOUT=3600`
   (run_ns3_phase2.py now reads that env var). All 18 fresh dirs have summary.json.
3. **Cleaned the repo**: deleted superseded scripts, 4 old models, 40 old ns-3 dirs, old
   figs/json, results/models/. Kept: 18 models, 18 fresh ns-3 dirs, current scripts + figs.

## Current results — FRESH multi-seed (overload, mean ± std over seeds 0/1/2)

### Link utilisation (analytical, max link-util %)
| Network | OSPF | SA-GNN | MARL |
|---------|-----:|-------:|-----:|
| Abilene (12n) | 122 | 65 ± 3 | 65 ± 2 |
| GÉANT (22n) | 126 | 98 ± 7 | 85 ± 0 |
| Germany50 (50n) | 109 | 82 ± 3 | 94 ± 11 |

### Quality of Service (ns-3, packet-level)
| Network | OSPF loss | SA-GNN loss | MARL loss | OSPF dly | SA-GNN dly | MARL dly |
|---------|----------:|------------:|----------:|---------:|-----------:|---------:|
| Abilene | 2.32% | 0.17 ± 0.00 | 0.18 ± 0.00 | 37.5 | 11.7 ± 0.1 | 12.6 ± 0.1 |
| GÉANT | 10.63% | 4.27 ± 1.49 | **0.91 ± 0.30** | 18.7 | 21.0 ± 4.9 | **12.2 ± 1.1** |
| Germany50 | 3.16% | **0.06 ± 0.03** | 2.00 ± 1.89 | 17.4 | **2.8 ± 0.9** | 9.3 ± 7.1 |

### Scale story (the headline)
- **Abilene (12n):** MARL ≈ SA-GNN, both crush OSPF.
- **GÉANT (22n): MARL wins clearly** (loss 0.91% vs SA 4.27% vs OSPF 10.63%; SA delay 21ms
  is worse than OSPF). Decentralization best at mid-scale.
- **Germany50 (50n): SA-GNN wins** (0.06%); MARL delay-improved but loss-limited & high-variance.
- Feasible regime: all tie ~0.1%.

## Open items / caveats
- **GÉANT SA-GNN is a noisy/poor outcome** (4.27 ± 1.49% loss; util 98 ± 7 — one seed converged
  badly). Decide before thesis: report honestly, or retrain/add seeds to fix the outlier.
- **fig_real3way_* still show OLD numbers** → regenerate with the fresh QoS (make_3way_fig.py).
- **§4.1 utilisation is analytical**, §4.2 QoS is ns-3 — label clearly in write-up (training is
  on the analytical surrogate by design; all *reported QoS* is ns-3).

## Next
1. Regenerate figures (fig_real3way_*, fig_multiseed_overload) with fresh numbers.
2. Update RESULTS_SUMMARY.md §2/§2a to fresh numbers.
3. WRITE-UP: Results chapter (3 subsections drafted in conversation / WRITEUP.md), Method
   (thesis_chapter3_method.md).
4. (optional) investigate GÉANT SA-GNN outlier; ns-3 link-util extraction if going all-ns-3.
