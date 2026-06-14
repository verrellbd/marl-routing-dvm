#!/usr/bin/env python3
"""
Full ns-3 evaluation of the trained GNN policy across MANY held-out traffic
matrices. For each unseen matrix we install both routings (OSPF shortest path and
the GNN's exact per-flow paths) in ns-3 and measure real QoS: packet loss, mean
delay, delivered throughput. Aggregates OSPF vs GNN as mean ± std.

This makes ns-3 (not the analytical model) the judge of the result.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import networkx as nx
from stable_baselines3 import PPO

from marl_routing.traffic import generate_matrix
from marl_routing.topology import load as load_topology
from marl_routing.sequential_routing_env import MultiTrafficSequentialEnv
from marl_routing.gnn_routing_agent import compute_ksp

TOPO = "abilene"
LOAD_FACTOR = 3.0
K_PATHS = 3
# Stratified held-out seeds: 3 OVERLOADED (OSPF>100%) + 3 FEASIBLE spanning 55-97%,
# so we see where the GNN helps (congestion) vs is neutral (uncongested).
OVERLOAD_SEEDS = [1009, 1013, 1018]
FEASIBLE_SEEDS = [1004, 1011, 1008]
TEST_SEEDS = OVERLOAD_SEEDS + FEASIBLE_SEEDS
_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="results/generalization/gnn_generalist")
_ap.add_argument("--tag", default="", help="suffix for output dir (e.g. _qos)")
_ap.add_argument("--export-only", action="store_true",
                 help="write routing JSONs then exit (run ns-3 separately via run_ns3_phase2.py)")
_ARGS, _ = _ap.parse_known_args()

MODEL = _ARGS.model
NS3_DIR = Path("~/thesis/ns-3-dev").expanduser()
RATESCALE, SIMTIME = 20, 8
OUT = Path(f"~/thesis/results/ns3_eval{_ARGS.tag}").expanduser().resolve(); OUT.mkdir(parents=True, exist_ok=True)


def gnn_paths_for(model, env, pairs, pair_paths, rates):
    """Run the GNN deterministically; return chosen node-path per non-zero flow."""
    env.set_matrix(rates)
    obs, _ = env.reset()
    chosen = {}
    done = False
    while not done:
        pi = env.order[env.pos]
        a, _ = model.predict(obs, deterministic=True)
        chosen[pi] = int(a) % K_PATHS
        obs, _, done, _, _ = env.step(a)
    flows = []
    for pi, (s, d) in enumerate(pairs):
        if rates[pi] <= 0:
            continue
        flows.append({
            "src": int(s), "dst": int(d), "rate_mbps": float(rates[pi]),
            "start": 2.0, "stop": 18.0,
            "gnn_path": [int(x) for x in pair_paths[pi][chosen.get(pi, 0)]],
            "ospf_path": [int(x) for x in nx.shortest_path(env.topo.graph, s, d)],
        })
    return flows


def run_ns3(routing_file, routing, state_file):
    routing_file = Path(routing_file).resolve()
    state_file = Path(state_file).resolve()
    topo = Path("~/thesis/topologies/abilene.json").expanduser().resolve()
    cmd = ["./ns3", "run",
           f"scratch/abilene-validate/abilene-validate "
           f"--topo={topo} "
           f"--routing_file={routing_file} --routing={routing} "
           f"--state={state_file} --simTime={SIMTIME} --rateScale={RATESCALE}"]
    r = subprocess.run(cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=400)
    if not state_file.exists():
        raise RuntimeError(f"ns-3 ({routing}) produced no state. stderr tail:\n{r.stderr[-400:]}")
    return json.loads(state_file.read_text())


def main():
    import gc
    topo = load_topology(TOPO)
    n = topo.n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    pair_paths = [compute_ksp(topo.graph, s, d, k=K_PATHS) for (s, d) in pairs]
    env = MultiTrafficSequentialEnv(TOPO, pairs, [np.zeros(len(pairs))], k_paths=K_PATHS)

    # ---- Phase 1: with model loaded, extract routing JSONs for every seed ----
    model = PPO.load(MODEL, device="cpu")
    meta = {}
    for seed in TEST_SEEDS:
        T = generate_matrix(TOPO, LOAD_FACTOR, seed=seed)
        rates = np.array([T[s, d] for (s, d) in pairs])
        flows = gnn_paths_for(model, env, pairs, pair_paths, rates)
        rf = OUT / f"routing_seed{seed}.json"
        rf.write_text(json.dumps({"seed": seed, "flows": flows}, indent=2))
        meta[seed] = {"routing_file": str(rf), "ospf_util": round(env.ospf_max_util(rates), 1)}
    # free PyTorch before forking ns-3 (avoids fork OOM on the shared machine)
    del model; gc.collect()

    if _ARGS.export_only:
        print(f"[export-only] wrote {len(TEST_SEEDS)} routing JSONs to {OUT}")
        print(f"  now run: python run_ns3_phase2.py --dir {OUT}")
        return

    # ---- Phase 2: run ns-3 for both routings (no torch in memory) ----
    rows = []
    for seed in TEST_SEEDS:
        rf = meta[seed]["routing_file"]
        o = run_ns3(rf, "ospf", OUT / f"ns3_ospf_{seed}.json")
        g = run_ns3(rf, "gnn",  OUT / f"ns3_gnn_{seed}.json")
        regime = "overload" if meta[seed]["ospf_util"] >= 100 else "feasible"
        rows.append({
            "seed": seed, "regime": regime, "ospf_analytical_util": meta[seed]["ospf_util"],
            "ospf_loss": round(o["loss_pct"], 2), "gnn_loss": round(g["loss_pct"], 2),
            "ospf_delay_ms": round(o["mean_delay_ms"], 2), "gnn_delay_ms": round(g["mean_delay_ms"], 2),
            "ospf_tput": round(o["throughput_mbps"], 0), "gnn_tput": round(g["throughput_mbps"], 0),
        })
        print(f"  [{regime:8}] seed {seed} (OSPF {meta[seed]['ospf_util']:.0f}%): "
              f"loss OSPF {o['loss_pct']:.2f}% -> GNN {g['loss_pct']:.2f}%  | "
              f"delay OSPF {o['mean_delay_ms']:.1f} -> GNN {g['mean_delay_ms']:.1f} ms")

    def grp(reg, k):
        v = [r[k] for r in rows if r["regime"] == reg]
        return round(float(np.mean(v)), 2) if v else None
    summary = {"topo": TOPO, "load_factor": LOAD_FACTOR, "rateScale": RATESCALE,
               "by_regime": {reg: {
                   "ospf_loss": grp(reg, "ospf_loss"), "gnn_loss": grp(reg, "gnn_loss"),
                   "ospf_delay_ms": grp(reg, "ospf_delay_ms"), "gnn_delay_ms": grp(reg, "gnn_delay_ms"),
               } for reg in ("overload", "feasible")},
               "per_matrix": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 66)
    print("ns-3 EVALUATION by regime (real packet-level QoS)")
    print("=" * 66)
    for reg in ("overload", "feasible"):
        b = summary["by_regime"][reg]
        if b["ospf_loss"] is None:
            continue
        print(f"  {reg.upper():9}  loss: OSPF {b['ospf_loss']:.2f}% -> GNN {b['gnn_loss']:.2f}%"
              f"   delay: OSPF {b['ospf_delay_ms']:.1f} -> GNN {b['gnn_delay_ms']:.1f} ms")
    print(f"\n  -> {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
