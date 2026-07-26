#!/usr/bin/env python3
"""Train ONE topology-agnostic GNN-actor MARL (MAPPO) policy across networks.

The multi-agent counterpart of train_topo_agnostic.py: per-node agents share one GNN
policy with L-round message passing (Option B — the paper-style multi-hop awareness that
the earlier MLP-actor MARL lacked). Trains on a MIX of backbones, evaluates on SEEN
topologies (both regimes) + ZERO-SHOT on a held-out network.

  python train_marl_gnn.py --seed 0 --updates 150 --tag _marlgnn

Saves results/marlgnn{tag}_seed{seed}/policy.pt + summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv
from marl_routing.topology import load as load_topology

# same congesting-regime loads as train_topo_agnostic.py so results are comparable
LOADS = {
    "abilene_sndlib": [8.0, 12.0, 16.0],
    "geant_sndlib": [3.0, 5.0, 7.0],
    "germany50_sndlib": [35.0, 50.0, 65.0],
}

ap = argparse.ArgumentParser()
ap.add_argument("--train-topos", default="abilene_sndlib,geant_sndlib")
ap.add_argument("--holdout", default="germany50_sndlib")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--updates", type=int, default=150)
ap.add_argument("--rollout", type=int, default=4096)
ap.add_argument("--n-per-scale", type=int, default=8)
ap.add_argument("--hidden", type=int, default=32)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--stretch", type=int, default=2)
ap.add_argument("--delay-penalty", type=float, default=0.5)
ap.add_argument("--max-flows", type=int, default=500,
                help="cap flows per TRAINING matrix to top-N by rate (keeps episodes "
                     "inside the rollout on large topologies e.g. germany50 ~1300 flows; "
                     "0 = no cap. Eval is always UNFILTERED so zero-shot stays comparable).")
ap.add_argument("--tag", default="_marlgnn")
A = ap.parse_args()


def pairs_of(topo):
    n = load_topology(topo).n_nodes
    return [(s, d) for s in range(n) for d in range(n) if s != d]


def _cap_flows(mats, k):
    """Zero out all but the top-k flows by rate in each matrix (tractable episodes)."""
    if not k:
        return mats
    out = []
    for m in mats:
        r = np.asarray(m, dtype=float).copy()
        if (r > 0).sum() > k:
            thr = np.sort(r)[::-1][k - 1]
            r[r < thr] = 0.0
        out.append(r)
    return out


def spec(topo, split, n_per_scale, max_flows=0):
    p = pairs_of(topo)
    mats = real_matrices(topo, p, LOADS[topo], n_per_scale=n_per_scale, split=split)
    if max_flows:
        mats = _cap_flows(mats, max_flows)
    return (topo, p, mats)


def evaluate(env, mats, policy, label):
    """Stratify test matrices into overload (OSPF>=100%) / feasible and report the
    MARL vs OSPF max-util margin in each regime."""
    ospf, marl = [], []
    for m in mats:
        o = env.ospf_max_util(m, 0)
        env.set_matrix(m, 0)
        obs, mask, _ = env.reset()
        done = False; info = {}
        while not done:
            a = policy(env)
            obs, mask, _, _, done, info = env.step(a)
        ospf.append(o); marl.append(info["max_util"])
    ospf = np.array(ospf); marl = np.array(marl)
    out = {}
    for regime, sel in [("overload", ospf >= 100), ("feasible", ospf < 100)]:
        if sel.sum() == 0:
            continue
        o = ospf[sel]; g = marl[sel]
        win = float((g < o).mean() * 100)
        out[regime] = {"ospf_mean": float(o.mean()), "marl_mean": float(g.mean()),
                       "delta_pt": float(o.mean() - g.mean()), "win_pct": win,
                       "n": int(sel.sum())}
        print(f"  [{label:22s} {regime:8s}] OSPF {o.mean():6.1f}%  ->  MARL {g.mean():6.1f}%"
              f"   delta {o.mean()-g.mean():+6.1f}pt   beats OSPF {win:.0f}% (n={sel.sum()})")
    return out


def main():
    train_topos = [t for t in A.train_topos.split(",") if t]
    out = Path(f"results/marlgnn{A.tag}_seed{A.seed}"); out.mkdir(parents=True, exist_ok=True)

    train_specs = [spec(t, "train", A.n_per_scale, A.max_flows) for t in train_topos]
    env = TopoAgnosticMARLEnv(train_specs, seed=A.seed, delay_penalty=A.delay_penalty,
                              stretch=A.stretch, normalize_reward=True)
    print(f"[train] topos={train_topos} holdout={A.holdout} seed={A.seed} "
          f"updates={A.updates} rollout={A.rollout} hidden={A.hidden} rounds={A.rounds}")

    mappo = GNNMAPPO(env, hidden=A.hidden, rounds=A.rounds, rollout_steps=A.rollout,
                     n_epochs=6, minibatch=512, seed=A.seed)
    mappo.learn(total_steps=A.rollout * A.updates, log_every=5,
                ckpt_dir=out / "ckpts", ckpt_every=30)
    mappo.save(out / "policy.pt")
    print(f"[saved] {out/'policy.pt'}")

    policy = mappo.act_fn(True)
    results = {"train_topos": train_topos, "holdout": A.holdout, "seed": A.seed,
               "seen": {}, "zero_shot": {}}

    print("\n[eval] SEEN topologies, held-out test traffic:")
    for t in train_topos:
        s = spec(t, "test", 6)
        ev = TopoAgnosticMARLEnv([s], seed=A.seed + 1, delay_penalty=A.delay_penalty,
                                 stretch=A.stretch, normalize_reward=True)
        results["seen"][t] = evaluate(ev, s[2], policy, t)

    if A.holdout:
        print(f"\n[eval] ZERO-SHOT on {A.holdout} (never trained on):")
        s = spec(A.holdout, "test", 6)
        ev = TopoAgnosticMARLEnv([s], seed=A.seed + 1, delay_penalty=A.delay_penalty,
                                 stretch=A.stretch, normalize_reward=True)
        results["zero_shot"][A.holdout] = evaluate(ev, s[2], policy, A.holdout)

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out/'summary.json'}")


if __name__ == "__main__":
    main()
