#!/usr/bin/env python3
"""
Aggregate ns-3 QoS results across multiple training seeds -> mean +/- std per
regime/metric. Reads results/ns3_eval_qos_seed{S}/summary.json for each seed.
Gives error bars on the headline numbers so dominance can be stated rigorously.
"""
import json
from pathlib import Path

import numpy as np

SEEDS = [0, 1, 2]
METRICS = ["ospf_loss", "gnn_loss", "ospf_delay_ms", "gnn_delay_ms"]


def main():
    per_seed = {}
    for s in SEEDS:
        f = Path(f"results/ns3_eval_qos_seed{s}/summary.json")
        if f.exists():
            per_seed[s] = json.loads(f.read_text())["by_regime"]
    if not per_seed:
        print("no per-seed summaries found"); return

    out = {"seeds": list(per_seed), "by_regime": {}}
    print("=" * 70)
    print(f"MULTI-SEED ns-3 QoS (mean +/- std over {len(per_seed)} seeds)")
    print("=" * 70)
    for reg in ("overload", "feasible"):
        out["by_regime"][reg] = {}
        print(f"\n  {reg.upper()}")
        for m in METRICS:
            vals = [per_seed[s][reg][m] for s in per_seed if per_seed[s][reg][m] is not None]
            if not vals:
                continue
            mean, std = float(np.mean(vals)), float(np.std(vals))
            out["by_regime"][reg][m] = {"mean": round(mean, 3), "std": round(std, 3)}
        b = out["by_regime"][reg]
        unit = lambda m: "%" if "loss" in m else " ms"
        print(f"    loss : OSPF {b['ospf_loss']['mean']:.2f}%        "
              f"GNN-QoS {b['gnn_loss']['mean']:.2f}% +/- {b['gnn_loss']['std']:.2f}")
        print(f"    delay: OSPF {b['ospf_delay_ms']['mean']:.1f} ms    "
              f"GNN-QoS {b['gnn_delay_ms']['mean']:.1f} +/- {b['gnn_delay_ms']['std']:.1f} ms")

    Path("results/ns3_eval_multiseed.json").write_text(json.dumps(out, indent=2))
    print(f"\n  -> results/ns3_eval_multiseed.json")


if __name__ == "__main__":
    main()
