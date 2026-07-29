#!/usr/bin/env python3
"""Single-agent counterpart of train_marl_gnn_tier2.py (for the OSPF|ECMP|single|MARL table).

Trains ONE topology-agnostic single-agent GNN policy (SB3-PPO + k-shortest-path selector)
on the 17 SNDlib topologies (TMgen or nominal traffic), then tests ZERO-SHOT on the 3 core
backbones with REAL MEASURED traffic, reporting OSPF + ECMP + single-agent max-util.

  python train_single_tier2.py --seed 0 --traffic tmgen --timesteps 500000 --tag _singletmgen
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from marl_routing.graph_routing_env import GraphSeqRoutingEnv
from marl_routing.nominal_traffic import nominal_matrices
from marl_routing.real_traffic import real_matrices
from marl_routing.tmgen_traffic import tmgen_matrices
from marl_routing.topo_gnn_extractor import TopoAgnosticGNNExtractor
from marl_routing.topology import load as load_topology

TRAIN_TOPOS = [
    "atlanta_sndlib", "cost266_sndlib", "dfn-bwin_sndlib", "dfn-gwin_sndlib",
    "di-yuan_sndlib", "france_sndlib", "india35_sndlib", "janos-us_sndlib",
    "newyork_sndlib", "nobel-eu_sndlib", "nobel-germany_sndlib", "nobel-us_sndlib",
    "norway_sndlib", "pdh_sndlib", "polska_sndlib", "ta1_sndlib", "zib54_sndlib",
]
TEST_TOPOS = ["abilene_sndlib", "geant_sndlib", "germany50_sndlib"]
TEST_LOADS = {"abilene_sndlib": [8., 12., 16.], "geant_sndlib": [3., 5., 7.],
              "germany50_sndlib": [35., 50., 65.]}

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--timesteps", type=int, default=500000)
ap.add_argument("--traffic", choices=["tmgen", "nominal"], default="tmgen")
ap.add_argument("--k-paths", type=int, default=3)
ap.add_argument("--hidden", type=int, default=32)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--n-envs", type=int, default=8)
ap.add_argument("--n-steps", type=int, default=256)
ap.add_argument("--ent-coef", type=float, default=0.01,
                help="PPO entropy bonus; raise (0.03-0.05) to prevent premature collapse "
                     "to shortest-path when training across many diverse topologies")
ap.add_argument("--raw-reward", action="store_true",
                help="disable per-episode reward normalisation (the proven topoagn_ctrl "
                     "config; normalisation shrinks the congestion signal below the delay "
                     "penalty -> shortest-path collapse)")
ap.add_argument("--max-flows", type=int, default=500)
ap.add_argument("--reward-form", choices=["whole", "marl"], default="whole",
                help="'whole' normalises congestion AND delay by the episode's OSPF "
                     "bottleneck (this env's own design, introduced to stop shortest-path "
                     "collapse); 'marl' matches the MARL env exactly -- congestion "
                     "normalised, flat 0.5 per DETOUR hop unnormalised. Ablation for the "
                     "last unmatched axis between the two arms.")
ap.add_argument("--metric", choices=["hop", "weighted"], default="hop",
                help="path-cost metric. 'weighted' = OSPF cost refBW/linkBW, making the\n"
                     "candidate paths and detour limits CAPACITY-AWARE; 'hop' is the\n"
                     "legacy capacity-blind behaviour. Identical on uniform-capacity topos.")
ap.add_argument("--tag", default="_singletmgen")
A = ap.parse_args()


class Progress(BaseCallback):
    """Log a real training percentage + ETA (SB3 is otherwise silent) and checkpoint."""

    def __init__(self, total, out, every=25000):
        super().__init__()
        self.total, self.out, self.every = total, out, every
        self.next_at, self.t0 = every, time.time()

    def _on_step(self):
        n = self.num_timesteps
        if n >= self.next_at:
            self.next_at += self.every
            el = time.time() - self.t0
            eta = el * (self.total - n) / max(n, 1)
            print(f"[progress] {n}/{self.total} ({100*n/self.total:5.1f}%) "
                  f"elapsed {el/60:.1f}min eta {eta/60:.1f}min", flush=True)
            self.model.save(self.out / "policy_ckpt")
        return True


def pairs_of(t):
    n = load_topology(t).n_nodes
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def cap(mats, k):
    if not k:
        return mats
    out = []
    for m in mats:
        r = np.asarray(m, float).copy()
        if (r > 0).sum() > k:
            r[r < np.sort(r)[::-1][k - 1]] = 0.0
        out.append(r)
    return out


def train_traffic(t, p):
    if A.traffic == "tmgen":
        return cap(tmgen_matrices(t, p, n_patterns=3,
                                  load_scales=(0.6, 0.8, 1.0, 1.2, 1.5), seed=A.seed), A.max_flows)
    return cap(nominal_matrices(t, p, seed=A.seed), A.max_flows)


def evaluate(env, mats, model):
    ospf, ecmp, single = [], [], []
    for m in mats:
        ospf.append(env.ospf_max_util(m, 0)); ecmp.append(env.ecmp_max_util(m, 0))
        env.set_matrix(m, 0)
        obs, _ = env.reset(); done = False; info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(a))
        single.append(info["max_util"])
    ospf, ecmp, single = map(np.array, (ospf, ecmp, single))
    out = {}
    for regime, sel in [("overload", ospf >= 100), ("feasible", ospf < 100)]:
        if sel.sum() == 0:
            continue
        o, e, g = ospf[sel], ecmp[sel], single[sel]
        out[regime] = {"ospf_mean": float(o.mean()), "ecmp_mean": float(e.mean()),
                       "single_mean": float(g.mean()),
                       "delta_ospf_pt": float(o.mean() - g.mean()),
                       "delta_ecmp_pt": float(e.mean() - g.mean()),
                       "win_ospf_pct": float((g < o).mean() * 100),
                       "win_ecmp_pct": float((g < e).mean() * 100), "n": int(sel.sum())}
        print(f"  [{regime:8s}] OSPF {o.mean():6.1f}% ECMP {e.mean():6.1f}% SINGLE {g.mean():6.1f}%"
              f"  vs-OSPF {o.mean()-g.mean():+.1f} vs-ECMP {e.mean()-g.mean():+.1f} (n={sel.sum()})",
              flush=True)
    return out


def main():
    out = Path(f"results/single{A.tag}_seed{A.seed}"); out.mkdir(parents=True, exist_ok=True)
    train_specs = [(t, pairs_of(t), train_traffic(t, pairs_of(t))) for t in TRAIN_TOPOS]

    def mk(rank):
        return lambda: GraphSeqRoutingEnv(train_specs, k_paths=A.k_paths,
                                          seed=A.seed * 100 + rank, normalize_reward=not A.raw_reward,
                                          metric=A.metric, reward_form=A.reward_form)
    venv = DummyVecEnv([mk(i) for i in range(A.n_envs)])
    print(f"[single] train 17 SNDlib ({A.traffic}); zero-shot {TEST_TOPOS}; seed={A.seed} "
          f"buffer={A.n_steps*A.n_envs} -> {A.timesteps//(A.n_steps*A.n_envs)} updates", flush=True)

    pk = {"features_extractor_class": TopoAgnosticGNNExtractor,
          "features_extractor_kwargs": {"k_paths": A.k_paths, "hidden_dim": A.hidden,
                                        "rounds": A.rounds}}
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=A.n_steps, batch_size=256,
                n_epochs=10, gamma=0.995, gae_lambda=0.95, ent_coef=A.ent_coef, seed=A.seed,
                device="cpu", policy_kwargs=pk, verbose=0)
    model.learn(total_timesteps=A.timesteps, progress_bar=False,
                callback=Progress(A.timesteps, out))
    model.save(out / "policy")
    print(f"[saved] {out/'policy'}.zip", flush=True)

    results = {"train_topos": TRAIN_TOPOS, "test_topos": TEST_TOPOS, "seed": A.seed,
               "traffic": A.traffic, "zero_shot": {}}
    print("\n[eval] ZERO-SHOT on core backbones (real measured):", flush=True)
    for t in TEST_TOPOS:
        p = pairs_of(t)
        mats = real_matrices(t, p, TEST_LOADS[t], n_per_scale=6, split="test")
        ev = GraphSeqRoutingEnv([(t, p, mats)], k_paths=A.k_paths, seed=A.seed + 1,
                                normalize_reward=not A.raw_reward, metric=A.metric,
                                reward_form=A.reward_form)
        print(f"[{t}]", flush=True)
        results["zero_shot"][t] = evaluate(ev, mats, model)
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
