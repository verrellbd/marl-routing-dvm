#!/usr/bin/env python3
"""Greedy best-response reference: at every hop take the admissible neighbour that
minimises the resulting bottleneck.

This is the arm the methodology chapter describes as an intermediate reference between
OSPF and the learned policies. It shares the MARL action space exactly -- the same
per-hop admissible set, the same cost-based stretch limits -- so it isolates what the
learned policy buys over acting greedily on the same choices. It is not a deployable
method: it needs the global link state after every hop of every demand.

Cells and loads mirror fill_offered_grid.py so the numbers drop into the same table.

  python eval_greedy.py --out results/greedy_grid.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from marl_routing.real_traffic import real_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topology import load as load_topology

CELLS = [
    ("abilene_sndlib",   [8.0, 12.0, 16.0],        "feasible"),
    ("abilene_sndlib",   [16.0, 22.0, 28.0],       "overload"),
    ("geant_sndlib",     [3.0, 5.0, 7.0],          "both"),
    ("germany50_sndlib", [35.0, 50.0, 65.0],       "overload"),
    ("germany50_sndlib", [15.0, 20.0, 25.0, 30.0], "feasible"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/greedy_grid.json")
ap.add_argument("--max-flows", type=int, default=200)
A = ap.parse_args()


def pairs_of(t):
    n = load_topology(t).n_nodes
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def cap(mats, k):
    out = []
    for m in mats:
        r = np.asarray(m, float).copy()
        if (r > 0).sum() > k:
            r[r < np.sort(r)[::-1][k - 1]] = 0.0
        out.append(r)
    return out


def greedy_action(env):
    """Admissible neighbour minimising the bottleneck that results from taking it."""
    valid = env._valid()
    best, best_i = None, None
    for i, ok in enumerate(valid):
        if not ok or i >= len(env.nbr_arc[env.cur_node]):
            continue
        ai = env.nbr_arc[env.cur_node][i]
        after = env.load.copy()
        after[ai] += env.cur_rate
        mu = float((100.0 * after / env.b.cap).max())
        if best is None or mu < best:
            best, best_i = mu, i
    if best_i is None:                       # dead end: env.step applies its own fallback
        return 0
    return best_i


def main():
    out = {}
    for topo, loads, want in CELLS:
        p = pairs_of(topo)
        mats = cap(real_matrices(topo, p, loads, n_per_scale=6, split="test"), A.max_flows)
        env = TopoAgnosticMARLEnv([(topo, p, mats)], seed=0, normalize_reward=True,
                                  metric="weighted")
        ospf, greedy = [], []
        for m in mats:
            ospf.append(env.ospf_max_util(m, 0))
            env.set_matrix(m, 0); env.reset()
            done, info = False, {}
            while not done:
                _, _, _, _, done, info = env.step(greedy_action(env))
            greedy.append(info["max_util"])
        ospf = np.array(ospf); greedy = np.array(greedy)
        short = topo.replace("_sndlib", "")
        for regime, sel in [("overload", ospf >= 100), ("feasible", ospf < 100)]:
            if want in (regime, "both") and sel.sum():
                out.setdefault(f"{short}/{regime}", {})["greedy"] = {
                    "mean": round(float(greedy[sel].mean()), 1), "sd": 0.0,
                    "n": int(sel.sum())}
                print(f"  {short}/{regime:8s} greedy {greedy[sel].mean():7.1f} "
                      f"(OSPF {ospf[sel].mean():6.1f}, n={sel.sum()})", flush=True)
    Path(A.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {A.out}")


if __name__ == "__main__":
    main()
