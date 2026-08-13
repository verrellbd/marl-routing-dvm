#!/usr/bin/env python3
"""Rebuild the packet-level grid from the ns3f_* simulation outputs.

There was no committed aggregator for results/final_ns3_grid.json -- it was produced
ad hoc -- so this script first REPRODUCES the existing 3-seed grid before being trusted
on ten. Run with --validate to check that; it prints any cell that disagrees.

Aggregation, matching how the chapter describes it: each seed contributes its own mean
over the matrices of a regime, and the reported figure is the mean over seeds with the
standard deviation ACROSS SEEDS. Seeds are the unit of dispersion, not matrices.

Utilisation is corrected by the 7/8 active-window factor (flows run [2s, 9s] of an 8 s
simulation), exactly as the Experiments chapter states.

OSPF is read from the marlh32 directories: it is deterministic given the matrix, so it
carries no seed dispersion and was never re-simulated for seeds 3-9.

  python rebuild_ns3_grid.py --validate           # check against the 3-seed grid
  python rebuild_ns3_grid.py --seeds 10
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

WINDOW = 7.0 / 8.0          # flows active [2,9] within an 8 s run

# reported arm -> (directory stem, state-file prefix)
ARMS = {
    "ospf":                  ("marlh32",  "ns3_ospf"),   # deterministic, seed 0 only
    "ecmp":                  ("ecmp",     "ns3_ecmp"),
    "single":                ("singleRM", "ns3_gnn"),    # --reward-form marl (reported)
    "marlh32":               ("marlh32",  "ns3_gnn"),
    "marlh64":               ("marlh64",  "ns3_gnn"),    # 3 seeds only, not reported
    "single_whole_ablation": ("single",   "ns3_gnn"),    # superseded whole-reward arm
}

# topology -> {regime: directory short name}
WHERE = {
    "abilene":   {"overload": "abilene",   "feasible": "abilene"},
    "geant":     {"overload": "geant",     "feasible": "geant"},
    "germany50": {"overload": "germany50", "feasible": "g50feas"},
}


def regime_of(short, matrix, arm_stem):
    """Read the regime label the exporter wrote into the routing file."""
    p = Path(f"results/ns3f_{arm_stem}_{short}_s0/routing_seed{matrix}.json")
    if not p.exists():
        return None
    return json.load(open(p))["regime"]


def seed_means(arm, topo, regime, seeds):
    """[(loss, delay, maxutil)] -- one triple per seed, each a mean over that
    regime's matrices. Seeds with no data are skipped rather than counted as zero."""
    stem, pref = ARMS[arm]
    short = WHERE[topo][regime]
    out = []
    for s in seeds:
        d = Path(f"results/ns3f_{stem}_{short}_s{s}")
        if not d.is_dir():
            continue
        rows = []
        for f in sorted(d.glob(f"{pref}_*.json")):
            m = int(re.search(rf"{pref}_(\d+)", f.name).group(1))
            if regime_of(short, m, stem) != regime:
                continue
            j = json.load(open(f))
            rows.append((j["loss_pct"], j["mean_delay_ms"],
                         max(j["link_utils"]) / WINDOW))
        if rows:
            out.append(tuple(np.mean(rows, axis=0)))
    return out


def build(seeds):
    grid = {}
    dropped = set()
    for topo in WHERE:
        for regime in ("feasible", "overload"):
            cell = {}
            for arm in ARMS:
                # OSPF is deterministic; one seed is the whole story
                use = [0] if arm == "ospf" else seeds
                v = seed_means(arm, topo, regime, use)
                if not v:
                    continue
                # A grid built at N seeds contains only arms that have N seeds.
                # Mixing counts silently (marlh64 at 3 inside a 10-seed file) is a
                # trap for anyone who reads the file later.
                if arm != "ospf" and len(v) < len(seeds):
                    dropped.add(f"{arm}({len(v)})")
                    continue
                a = np.array(v)
                cell[arm] = {
                    "loss_pct":    round(float(a[:, 0].mean()), 2),
                    "loss_sd":     round(float(a[:, 0].std(ddof=0)), 2),
                    "delay_ms":    round(float(a[:, 1].mean()), 2),
                    "delay_sd":    round(float(a[:, 1].std(ddof=0)), 2),
                    "maxutil_pct": round(float(a[:, 2].mean()), 2),
                    "maxutil_sd":  round(float(a[:, 2].std(ddof=0)), 2),
                    "seeds":       len(v),
                }
            grid[f"{topo}/{regime}"] = cell
    if dropped:
        print(f"  [dropped, fewer than {len(seeds)} seeds] {', '.join(sorted(dropped))}")
    return grid


ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--out", default="results/final_ns3_grid.json")
ap.add_argument("--validate", action="store_true",
                help="rebuild with 3 seeds and diff against final_ns3_grid.json")
A = ap.parse_args()

if A.validate:
    ref = json.load(open("results/final_ns3_grid.json"))
    got = build(list(range(3)))
    bad = 0
    for cell, arms in ref.items():
        for arm, vals in arms.items():
            for k, want in vals.items():
                # OSPF was simulated once per seed dir and averaged over identical
                # copies; we read it once. The metrics agree, only the count differs.
                if arm == "ospf" and k == "seeds":
                    continue
                have = got.get(cell, {}).get(arm, {}).get(k)
                if have is None or abs(have - want) > 0.011:
                    print(f"  MISMATCH {cell:22} {arm:22} {k:12} "
                          f"ref={want} rebuilt={have}")
                    bad += 1
    print("VALIDATION: exact match" if not bad else f"VALIDATION: {bad} field(s) differ")
    raise SystemExit(0 if not bad else 1)

grid = build(list(range(A.seeds)))
Path(A.out).write_text(json.dumps(grid, indent=1))
print(f"wrote {A.out}")
for cell, arms in grid.items():
    n = {a: v["seeds"] for a, v in arms.items()}
    print(f"  {cell:22} seeds per arm: {n}")
