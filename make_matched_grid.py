#!/usr/bin/env python3
"""Aggregate the matched-1.5M-budget comparison into one grid.

Reads results/ns3m_<arm>_<topo>_s<seed>/ produced by run_ns3_matched.sh + run_ns3_ecmp.sh
and reports, per (topology, regime, arm), the PACKET-LEVEL metrics:
    loss_pct, mean_delay_ms, throughput_mbps, and ns-3 max link utilisation.

ns-3 utilisation notes (see thesis METHOD):
  * it SATURATES at ~100% — a real link cannot carry more than capacity; excess offered
    load becomes loss. So in the overload regime every method pins near 100% and the
    differentiator is loss, not utilisation. Utilisation differentiates in FEASIBLE.
  * ACTIVE_FRAC: flows are active 7s of the 8s measurement window, so a link at 100% line
    rate reads 87.5%. Divide by 7/8 to recover true utilisation (uniform, so it cancels in
    ratios, but we correct it for absolute honesty).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ACTIVE_FRAC = 7.0 / 8.0
ARMS = ["ospf", "ecmp", "single", "marlh32", "marlh64"]
TOPOS = ["abilene", "geant", "germany50"]


def load(fp):
    try:
        return json.loads(Path(fp).read_text())
    except Exception:
        return None


def collect():
    """-> data[(topo, regime, arm)] = list of dicts(loss, delay, tput, util)"""
    data = defaultdict(list)
    for d in sorted(Path("results").glob("ns3m_*")):
        parts = d.name.split("_")          # ns3m, arm, topo, sN
        if len(parts) < 4:
            continue
        arm, topo = parts[1], parts[2]
        # germany50's feasible regime was exported separately (its default TEST_LOADS are
        # all in overload), into …_g50feas_… dirs -> fold back into the germany50 rows.
        topo = "germany50" if topo == "g50feas" else topo
        for rf in sorted(d.glob("routing_seed*.json")):
            i = rf.stem.replace("routing_seed", "")
            regime = (load(rf) or {}).get("regime", "feasible")
            # the learned/ecmp routing lives in ns3_gnn_* (or ns3_ecmp_* for the ECMP dirs)
            learned = load(d / f"ns3_gnn_{i}.json") or load(d / f"ns3_ecmp_{i}.json")
            if learned:
                data[(topo, regime, arm)].append(rec(learned))
            # OSPF is re-simulated in every non-ECMP dir; collect once per (topo, regime)
            o = load(d / f"ns3_ospf_{i}.json")
            if o:
                data[(topo, regime, "ospf")].append(rec(o))
    return data


def rec(j):
    return {"loss": j["loss_pct"], "delay": j["mean_delay_ms"],
            "tput": j["throughput_mbps"],
            "util": j["max_offered_util_pct"] / ACTIVE_FRAC}


def main():
    data = collect()
    for metric, unit, lower_better in [("util", "%", True), ("loss", "%", True),
                                       ("delay", "ms", True), ("tput", "Mbps", False)]:
        print(f"\n=== ns-3 {metric} ({unit}) {'lower' if lower_better else 'higher'} is better ===")
        print(f"{'topology / regime':26s} " + "".join(f"{a:>14s}" for a in ARMS))
        for t in TOPOS:
            for reg in ("overload", "feasible"):
                cells = []
                for a in ARMS:
                    v = data.get((t, reg, a))
                    cells.append(f"{np.mean([x[metric] for x in v]):9.2f}(n{len(v):2d})"
                                 if v else " " * 14)
                if any(c.strip() for c in cells):
                    print(f"{t + ' / ' + reg:26s} " + "".join(cells))
    out = {f"{t}|{r}|{a}": {m: float(np.mean([x[m] for x in v])) for m in
                            ("util", "loss", "delay", "tput")} | {"n": len(v)}
           for (t, r, a), v in data.items()}
    Path("results/matched_ns3_grid.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print("\n[saved] results/matched_ns3_grid.json")


if __name__ == "__main__":
    main()
