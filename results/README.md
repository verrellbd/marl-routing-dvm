# What is in `results/`

Written 2026-07-28, after deleting five superseded measurement batches. The thesis
reports **one** batch. Anything not listed under "Live" below is not reported.

## Live — these are what the thesis uses

| path | what it is |
|---|---|
| `final_ns3_grid.json` | **The results table.** 378 ns-3 sims, 0 failures. Every number in the Results chapter comes from here. |
| `ns3f_<arm>_<topo>_s<seed>/` | Raw per-simulation measurements behind that grid. 48 dirs. |
| `width_selection.json` | MARL hidden-width selection, scored on the **training** topologies (not the test backbones). |

Arms in `ns3f_*`: `ospf`, `ecmp`, `single`, `marlh32`, `marlh64`.
Topos: `abilene`, `geant`, `germany50`, plus `g50feas` (Germany50's feasible regime,
measured at lower loads because its normal loads are all overload).

## Live policies

| tag | arm | status |
|---|---|---|
| `marlgnn_tier2m15cm_seed{0,1,2}` | MARL h=32 | **reported policy** |
| `marlgnn_tier2m15h64cm_seed{0,1,2}` | MARL h=64 | sensitivity check |
| `single_singleH64gcap_seed{0,1,2}` | single-agent | baseline |
| `single_singleH64gRM_seed{0,1,2}` | single-agent, MARL reward form | reward-form ablation |

All four were trained by `train_matched_hparams.sh` (except `RM`, which adds
`--reward-form marl`) at a matched 1.5M steps with identical PPO hyperparameters.

## Superseded policies — kept, not reported

Older training runs are still on disk because they cost hours of compute and are the
record of how the work developed. **Do not quote them.** They differ from the live
arms in at least one of: capacity-blind path construction (`--metric hop`), unmatched
PPO hyperparameters, a smaller budget, or per-topology rather than zero-shot training.

Rough guide to the tags: `_tier2m15` / `_singleH64g` (capacity-blind), `_cap`
(capacity-aware but MARL's own PPO defaults), `*_sndlib_*_real_seed*` (per-topology
specialists, older env), `topoagn_*` / `marlgnn_zoo_*` / `single_singletmgen_*`
(earlier phases).

## Deleted on 2026-07-28

Five measurement batches whose numbers did not survive re-measurement, plus the
launchers that produced them. All recoverable from git history:

```
git log --oneline --diff-filter=D -- results/ns3m_marlh32_geant_s0
git checkout <commit>^ -- results/ns3m_'*'
```

Removed: `ns3m_*` (84 dirs — the capacity-blind matched grid and the Abilene
diagnostic re-runs), `ns3_eval_real{marl,sa}_fresh_*` (18 dirs — per-topology
specialists), `ns3_topoagn_*` (5 dirs — earliest topology-agnostic runs),
`matched_ns3_grid.json`, and the scripts `run_ns3_matched.sh`, `run_ns3_ecmp.sh`,
`run_ns3_ecmpnative.sh`, `run_ns3_g50feas.sh`, `run_ns3_abilene_weighted.sh`,
`run_ns3_abilene_capaware.sh`, `train_capaware.sh`, `make_matched_grid.py`,
`eval_g50_feasible.py`.

Why they went: `matched_ns3_grid.json` had four headline claims and three of them
reversed under correct measurement. Keeping it invited quoting the wrong number.
