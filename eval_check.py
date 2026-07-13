#!/usr/bin/env python3
"""Quick post-training check: for each network, compare OSPF vs single-agent GNN vs MARL
on the real held-out TEST matrices (overload regime), as max link-utilisation, mean +/- std
over the three training seeds. Lower is better; < 100% = feasible.

Run on monaco:  python eval_check.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import torch
torch.set_num_threads(2)
from stable_baselines3 import PPO

from marl_routing.topology import load as load_topology
from marl_routing.real_traffic import real_matrices
from marl_routing.sequential_routing_env import MultiTrafficSequentialEnv
from marl_routing.multiagent_routing_env import MultiAgentRoutingEnv
from marl_routing.mappo import MAPPO

# topo -> (eval load scale, marl tag, max_stretch used in training)
CFG = {
    "abilene_sndlib":  (12.0, "_real",  None),
    "geant_sndlib":    (5.0,  "_real",  None),
    "germany50_sndlib":(35.0, "_opt3b", 4),
}
# rough expected overload numbers from our established runs (for a sanity comparison)
EXPECT = {"abilene_sndlib": (122, 67, 64),
          "geant_sndlib": (126, 92, 97),
          "germany50_sndlib": (109, 86, "~high-var")}
SEEDS = [0, 1, 2]


def sa_maxutil(model, seq, r):
    seq.set_matrix(r); o, _ = seq.reset(); d = False; inf = {}
    while not d:
        a, _ = model.predict(o, deterministic=True); o, _, d, _, inf = seq.step(a)
    return inf["max_util"]


def main():
    print(f"{'Network':17}{'OSPF':>6}{'SA-GNN':>14}{'MARL':>16}   (overload max-util %, mean±std over seeds)")
    print("-" * 78)
    for topo, (load, mtag, ms) in CFG.items():
        n = load_topology(topo).n_nodes
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        mats = real_matrices(topo, pairs, [load], n_per_scale=20, split="test")
        seq = MultiTrafficSequentialEnv(topo, pairs, [np.zeros(len(pairs))], k_paths=3)
        mae = MultiAgentRoutingEnv(topo, pairs, [np.zeros(len(pairs))], stretch=1, max_stretch=ms)
        ospf = np.array([seq.ospf_max_util(r) for r in mats]); ov = ospf >= 100
        sa_s, ma_s = [], []
        for s in SEEDS:
            gm = PPO.load(f"results/{topo}_qos_real_seed{s}/gnn_generalist_qos", device="cpu")
            sa_s.append(np.array([sa_maxutil(gm, seq, r) for r in mats])[ov].mean()); del gm
            mm = MAPPO(mae); mm.load(f"results/{topo}_marl{mtag}_seed{s}/mappo_actor_critic.pt")
            af = mm.act_fn(True); ma = []
            for r in mats:
                mae.rollout_paths(af, r); ma.append(mae.cur_max)
            ma_s.append(np.array(ma)[ov].mean())
        eo, es, em = EXPECT[topo]
        print(f"{topo:17}{ospf[ov].mean():6.0f}{np.mean(sa_s):9.0f} ± {np.std(sa_s):<3.0f}"
              f"{np.mean(ma_s):10.0f} ± {np.std(ma_s):<3.0f}   (expected ~ OSPF {eo} / SA {es} / MARL {em})")
    print("-" * 78)
    print("GOOD if: both SA-GNN and MARL are well below OSPF (and below 100 = feasible).")
    print("MARL on germany50 is expected higher-variance than SA-GNN (coordination cost at scale).")


if __name__ == "__main__":
    main()
