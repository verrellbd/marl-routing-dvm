#!/usr/bin/env python3
"""
Multi-seed analytical evaluation: for each topology, load the seed 0/1/2 models
(single-agent GNN + MARL) and compute max link-utilisation on the DETERMINISTIC real
test matrices, stratified by congestion regime. Reports OSPF vs SA-GNN vs MARL as
mean +/- std ACROSS MODEL SEEDS — the error bars on the headline routing metric,
cheaply (no ns-3). ns-3 remains the single-seed high-fidelity anchor.

  python eval_multiseed_analytical.py
"""
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from marl_routing.topology import load as load_topology
from marl_routing.real_traffic import real_matrices
from marl_routing.sequential_routing_env import MultiTrafficSequentialEnv
from marl_routing.multiagent_routing_env import MultiAgentRoutingEnv
from marl_routing.mappo import MAPPO

# topo -> (eval load scale, n test matrices)
TOPOS = {"abilene_sndlib": 12.0, "geant_sndlib": 5.0, "germany50_sndlib": 35.0}
SEEDS = [0, 1, 2]
K_PATHS = 3


def sa_maxutil(model, seqenv, rates):
    seqenv.set_matrix(rates); obs, _ = seqenv.reset(); done = False
    info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = seqenv.step(a)
    return info["max_util"]


def main():
    out = {}
    for topo, load in TOPOS.items():
        n = load_topology(topo).n_nodes
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        mats = real_matrices(topo, pairs, [load], n_per_scale=20, split="test")
        seqenv = MultiTrafficSequentialEnv(topo, pairs, [np.zeros(len(pairs))], k_paths=K_PATHS)
        maenv = MultiAgentRoutingEnv(topo, pairs, [np.zeros(len(pairs))], stretch=1)

        ospf = np.array([seqenv.ospf_max_util(r) for r in mats])
        regime = np.where(ospf >= 100, "overload", "feasible")

        sa_by_seed, marl_by_seed = [], []
        for s in SEEDS:
            sa_model = PPO.load(f"results/{topo}_qos_real_seed{s}/gnn_generalist_qos", device="cpu")
            sa = np.array([sa_maxutil(sa_model, seqenv, r) for r in mats]); del sa_model
            ma = MAPPO(maenv); ma.load(f"results/{topo}_marl_real_seed{s}/mappo_actor_critic.pt")
            af = ma.act_fn(True)
            marl = []
            for r in mats:
                maenv.rollout_paths(af, r); marl.append(maenv.cur_max)
            sa_by_seed.append(sa); marl_by_seed.append(np.array(marl))
        sa_by_seed = np.array(sa_by_seed); marl_by_seed = np.array(marl_by_seed)  # [seed, matrix]

        rec = {}
        for reg in ["overload", "feasible"]:
            m = regime == reg
            if m.sum() == 0:
                continue
            # per seed: mean over matrices in this regime; then mean+/-std over seeds
            sa_seed_means = sa_by_seed[:, m].mean(axis=1)
            ma_seed_means = marl_by_seed[:, m].mean(axis=1)
            rec[reg] = {
                "n_matrices": int(m.sum()),
                "ospf": round(float(ospf[m].mean()), 1),
                "sa_gnn": [round(float(sa_seed_means.mean()), 1), round(float(sa_seed_means.std()), 1)],
                "marl": [round(float(ma_seed_means.mean()), 1), round(float(ma_seed_means.std()), 1)],
            }
        out[topo] = rec
        print(f"\n=== {topo} (load {load}) — max link-util %, mean+/-std over seeds {SEEDS} ===")
        for reg, v in rec.items():
            print(f"  {reg:9} (n={v['n_matrices']}): OSPF {v['ospf']:.0f}  "
                  f"SA-GNN {v['sa_gnn'][0]:.0f}±{v['sa_gnn'][1]:.0f}  "
                  f"MARL {v['marl'][0]:.0f}±{v['marl'][1]:.0f}")
    Path("results/multiseed_analytical.json").write_text(json.dumps(out, indent=2))
    print("\n-> results/multiseed_analytical.json")


if __name__ == "__main__":
    main()
