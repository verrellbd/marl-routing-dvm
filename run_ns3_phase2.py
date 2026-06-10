#!/usr/bin/env python3
"""
Lightweight ns-3 runner (NO torch import -> small RSS -> avoids fork-OOM on the
shared machine). Reads pre-exported routing JSONs (gnn_path + ospf_path per flow)
from a directory, runs ns-3 for both routings, and aggregates QoS by congestion
regime. Use after evaluate_ns3.py has written the routing_seed*.json files.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

OVERLOAD_SEEDS = {1009, 1013, 1018}
FEASIBLE_SEEDS = {1004, 1011, 1008}
NS3_DIR = Path("~/thesis/ns-3-dev").expanduser()
TOPO = Path("~/thesis/topologies/abilene.json").expanduser().resolve()
RATESCALE, SIMTIME = 20, 8


def run_ns3(routing_file, routing, state_file):
    routing_file, state_file = Path(routing_file).resolve(), Path(state_file).resolve()
    cmd = ["./ns3", "run",
           f"scratch/abilene-validate/abilene-validate --topo={TOPO} "
           f"--routing_file={routing_file} --routing={routing} "
           f"--state={state_file} --simTime={SIMTIME} --rateScale={RATESCALE}"]
    r = subprocess.run(cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=400)
    if not state_file.exists():
        raise RuntimeError(f"ns-3 ({routing}) no state. stderr:\n{r.stderr[-400:]}")
    return json.loads(state_file.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with routing_seed*.json")
    args = ap.parse_args()
    d = Path(args.dir).expanduser().resolve()

    rows = []
    for rf in sorted(d.glob("routing_seed*.json")):
        seed = int(rf.stem.replace("routing_seed", ""))
        regime = "overload" if seed in OVERLOAD_SEEDS else "feasible"
        o = run_ns3(rf, "ospf", d / f"ns3_ospf_{seed}.json")
        g = run_ns3(rf, "gnn",  d / f"ns3_gnn_{seed}.json")
        rows.append({"seed": seed, "regime": regime,
                     "ospf_loss": round(o["loss_pct"], 2), "gnn_loss": round(g["loss_pct"], 2),
                     "ospf_delay_ms": round(o["mean_delay_ms"], 2), "gnn_delay_ms": round(g["mean_delay_ms"], 2),
                     "ospf_tput": round(o["throughput_mbps"], 0), "gnn_tput": round(g["throughput_mbps"], 0)})
        print(f"  [{regime:8}] seed {seed}: loss OSPF {o['loss_pct']:.2f}% -> GNN {g['loss_pct']:.2f}%"
              f"  | delay OSPF {o['mean_delay_ms']:.1f} -> GNN {g['mean_delay_ms']:.1f} ms")

    def grp(reg, k):
        v = [r[k] for r in rows if r["regime"] == reg]
        return round(float(np.mean(v)), 2) if v else None
    summary = {"by_regime": {reg: {k: grp(reg, k) for k in
               ("ospf_loss", "gnn_loss", "ospf_delay_ms", "gnn_delay_ms", "ospf_tput", "gnn_tput")}
               for reg in ("overload", "feasible")}, "per_matrix": rows}
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== by regime ===")
    for reg in ("overload", "feasible"):
        b = summary["by_regime"][reg]
        if b["ospf_loss"] is None:
            continue
        print(f"  {reg.upper():9} loss OSPF {b['ospf_loss']:.2f}% -> GNN {b['gnn_loss']:.2f}%"
              f"   delay OSPF {b['ospf_delay_ms']:.1f} -> GNN {b['gnn_delay_ms']:.1f} ms")
    print(f"  -> {d/'summary.json'}")


if __name__ == "__main__":
    main()
