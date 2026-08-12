#!/usr/bin/env python3
"""Paired significance tests, MARL (h=32) against OSPF, on the final ns-3 grid.

Each learned-arm directory holds its own simulation AND the OSPF simulation for
the SAME traffic matrix, so every comparison is naturally paired: the matrix is
held fixed and only the routing changes. That is the right unit -- matrices differ
enormously in offered load, so an unpaired test would drown the routing effect in
between-matrix variance.

Pairing is on the matrix, with the learned arm's three seeds averaged first: the
seeds are three draws of one policy, not three independent observations of the
matrix, so treating them as separate samples would inflate n.

Regimes are labelled by OSPF's own packet loss, which separates cleanly (feasible
< 0.3%, overload > 12%) and agrees with the analytical labelling used throughout.
Nine matrices per regime across the three backbones, so Wilcoxon signed-rank has a
smallest attainable two-sided p of 0.004.

  python significance_test.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ARM = "marlh32"
DIRS = {           # directory stem -> topology label
    "abilene": "Abilene", "geant": "GEANT",
    "germany50": "Germany50", "g50feas": "Germany50",
}
SEEDS = (0, 1, 2)


def read(stem, seed, kind, m):
    p = Path(f"results/ns3f_{ARM}_{stem}_s{seed}/ns3_{kind}_{m}.json")
    return json.load(open(p)) if p.exists() else None


def maxutil(rec):
    return max(rec["link_utils"])


rows = []   # (regime, topology, matrix, ospf_loss, marl_loss, ospf_util, marl_util)
for stem, topo in DIRS.items():
    mats = sorted(int(re.search(r"ospf_(\d+)", f).group(1))
                  for f in glob.glob(f"results/ns3f_{ARM}_{stem}_s0/ns3_ospf_*.json"))
    for m in mats:
        o = read(stem, 0, "ospf", m)
        g = [read(stem, s, "gnn", m) for s in SEEDS]
        g = [x for x in g if x is not None]
        if o is None or not g:
            continue
        regime = "overload" if o["loss_pct"] > 1.0 else "feasible"
        rows.append((regime, topo, m,
                     o["loss_pct"], float(np.mean([x["loss_pct"] for x in g])),
                     maxutil(o), float(np.mean([maxutil(x) for x in g]))))

print(f"{'regime':9} {'topology':10} {'mat':>4} "
      f"{'OSPF loss':>10} {'MARL loss':>10} {'OSPF util':>10} {'MARL util':>10}")
for r in sorted(rows):
    print(f"{r[0]:9} {r[1]:10} {r[2]:>4} "
          f"{r[3]:>10.2f} {r[4]:>10.2f} {r[5]:>10.1f} {r[6]:>10.1f}")

for regime, metric, oi, mi, lower_better in [
        ("overload", "packet loss (%)", 3, 4, True),
        ("feasible", "max link utilisation (%)", 5, 6, True)]:
    sel = [r for r in rows if r[0] == regime]
    o = np.array([r[oi] for r in sel])
    g = np.array([r[mi] for r in sel])
    d = g - o
    w = stats.wilcoxon(g, o)                     # paired, two-sided
    t = stats.ttest_rel(g, o)
    print(f"\n=== {regime}: {metric}, MARL vs OSPF (n={len(sel)} matrices) ===")
    print(f"  OSPF mean {o.mean():.2f}   MARL mean {g.mean():.2f}   "
          f"mean diff {d.mean():+.2f}   median diff {np.median(d):+.2f}")
    print(f"  MARL better on {int((d < 0).sum())}/{len(d)} matrices")
    print(f"  Wilcoxon signed-rank W={w.statistic:.1f}, p={w.pvalue:.4f}")
    print(f"  paired t-test      t={t.statistic:.2f}, p={t.pvalue:.4f}")

    # Abilene is the network the paper reports as the failure case; quote the
    # restricted test only alongside the pooled one, never instead of it.
    sub = [r for r in sel if r[1] != "Abilene"]
    o2 = np.array([r[oi] for r in sub]); g2 = np.array([r[mi] for r in sub])
    w2 = stats.wilcoxon(g2, o2)
    print(f"  excluding Abilene (n={len(sub)}): better on "
          f"{int((g2 - o2 < 0).sum())}/{len(sub)}, W={w2.statistic:.1f}, "
          f"p={w2.pvalue:.4f}")
