#!/usr/bin/env python3
"""Paired Wilcoxon signed-rank tests on bottleneck offered load.

Pairs are (model seed, test matrix): every policy is evaluated on the identical
matrix, so OSPF / SA-GNN / MARL can be compared pairwise. Reuses the offered-load
computation from make_offered_util_fig.py so the p-values and Table 1 come from
exactly the same numbers.

Caveat reported in the text: the 9 (resp. 6) pairs are 3 seeds x 3 (resp. 2)
matrices, so pairs sharing a matrix are not independent. The p-values are
indicative of consistency across the evaluation set, not a population test.
"""
import importlib.util
import json
import os

import numpy as np
from scipy.stats import wilcoxon

spec = importlib.util.spec_from_file_location("m", "make_offered_util_fig.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def paired(net):
    """-> {regime: {method: array}} aligned on (seed, matrix)."""
    cap = m.caps(net)
    rec = {}
    for s in m.SEEDS:
        for tag, d in (("sa", f"{m.R}/ns3_eval_realsa_fresh_{net}_s{s}"),
                       ("marl", f"{m.R}/ns3_eval_realmarl_fresh_{net}_s{s}")):
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.startswith("routing_seed"):
                    continue
                r = json.load(open(f"{d}/{fn}"))
                k = (s, fn)
                e = rec.setdefault(k, {"regime": r["regime"]})
                if tag == "sa":
                    e["ospf"] = m.offered_maxutil(r["flows"], "ospf_path", cap)
                    e["sa"] = m.offered_maxutil(r["flows"], "gnn_path", cap)
                else:
                    e["marl"] = m.offered_maxutil(r["flows"], "gnn_path", cap)
    out = {}
    for reg in m.REGIMES:
        ks = [k for k, v in rec.items() if v["regime"] == reg and len(v) == 4]
        out[reg] = {t: np.array([rec[k][t] for k in ks])
                    for t in ("ospf", "sa", "marl")}
    return out


def main():
    for net, _ in m.NETS:
        d = paired(net)
        for reg in m.REGIMES:
            a = d[reg]
            n = len(a["ospf"])
            if n == 0:
                continue
            ps = {f"{x} vs {y}": wilcoxon(a[x], a[y]).pvalue
                  for x, y in (("ospf", "sa"), ("ospf", "marl"), ("sa", "marl"))}
            body = "  ".join(f"{k}: p={v:.4f}" for k, v in ps.items())
            print(f"{net:18s} {reg:9s} n={n:2d}  {body}")


if __name__ == "__main__":
    main()
