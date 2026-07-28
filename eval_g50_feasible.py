#!/usr/bin/env python3
"""Fill the missing germany50 FEASIBLE row of the ANALYTICAL grid.

The trainers evaluate germany50 only at TEST_LOADS 35/50/65, which are all above capacity
(OSPF 100-222%) -> the zero-shot summary has an overload row and no feasible row, unlike
abilene/geant. This re-evaluates the SAVED policies at lower loads and reports the feasible
subset, so all three topologies have both regimes.

  python eval_g50_feasible.py            -> results/g50_feasible_analytical.json
"""
import json
from pathlib import Path

import numpy as np

from marl_routing.graph_routing_env import GraphSeqRoutingEnv
from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topo_gnn_extractor import TopoAgnosticGNNExtractor  # noqa: F401
from marl_routing.topology import load as load_topology
from stable_baselines3 import PPO

TOPO = "germany50_sndlib"
LOADS = [15.0, 20.0, 25.0, 30.0]
N_PER_SCALE = 6


def main():
    n = load_topology(TOPO).n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    mats = real_matrices(TOPO, pairs, LOADS, n_per_scale=N_PER_SCALE, split="test")

    ref = GraphSeqRoutingEnv([(TOPO, pairs, mats)], k_paths=3, seed=0)
    b = ref.bundles[0]
    ospf = np.array([b.ospf_max_util(m) for m in mats])
    ecmp = np.array([b.ecmp_max_util(m) for m in mats])
    keep = ospf < 100
    print(f"[germany50] {keep.sum()}/{len(mats)} matrices feasible "
          f"(OSPF {ospf[keep].min():.1f}-{ospf[keep].max():.1f}%)")
    fmats = [m for m, k in zip(mats, keep) if k]
    out = {"topo": TOPO, "loads": LOADS, "n": int(keep.sum()),
           "ospf_mean": float(ospf[keep].mean()), "ecmp_mean": float(ecmp[keep].mean()),
           "arms": {}}
    print(f"  OSPF {ospf[keep].mean():6.2f}%   ECMP {ecmp[keep].mean():6.2f}%")

    for seed in (0, 1, 2):
        # --- single-agent (SB3) ---
        p = Path(f"results/single_singleH64g_seed{seed}/policy.zip")
        if p.exists():
            env = GraphSeqRoutingEnv([(TOPO, pairs, fmats)], k_paths=3, seed=seed + 1)
            model = PPO.load(p, device="cpu")
            vals = []
            for m in fmats:
                env.set_matrix(m, 0)
                obs, _ = env.reset(); done = False; info = {}
                while not done:
                    a, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _, info = env.step(int(a))
                vals.append(info["max_util"])
            out["arms"][f"single_s{seed}"] = float(np.mean(vals))
            print(f"  single  seed{seed}: {np.mean(vals):6.2f}%")

        # --- MARL arms ---
        for tag, hid in (("tier2m15", 32), ("tier2m15h64", 64)):
            mp = Path(f"results/marlgnn_{tag}_seed{seed}/policy.pt")
            if not mp.exists():
                continue
            env = TopoAgnosticMARLEnv([(TOPO, pairs, fmats)], seed=seed + 1, stretch=2)
            mappo = GNNMAPPO(env, hidden=hid, rounds=3, seed=seed)
            mappo.load(mp)
            pol = mappo.act_fn(True)
            vals = []
            for m in fmats:
                env.set_matrix(m, 0)
                obs, mask, _ = env.reset(); done = False; info = {}
                while not done:
                    obs, mask, _, _, done, info = env.step(pol(env))
                vals.append(info["max_util"])
            name = "marlh32" if hid == 32 else "marlh64"
            out["arms"][f"{name}_s{seed}"] = float(np.mean(vals))
            print(f"  {name} seed{seed}: {np.mean(vals):6.2f}%")

    for arm in ("single", "marlh32", "marlh64"):
        v = [out["arms"][k] for k in out["arms"] if k.startswith(arm + "_")]
        if v:
            out[f"{arm}_mean"] = float(np.mean(v))
            out[f"{arm}_std"] = float(np.std(v))
            print(f"[mean] {arm:8s} {np.mean(v):6.2f} +- {np.std(v):.2f} "
                  f"(vs OSPF {ospf[keep].mean() - np.mean(v):+.1f}pt, "
                  f"vs ECMP {ecmp[keep].mean() - np.mean(v):+.1f}pt)")
    Path("results/g50_feasible_analytical.json").write_text(json.dumps(out, indent=2))
    print("\n[saved] results/g50_feasible_analytical.json")


if __name__ == "__main__":
    main()
