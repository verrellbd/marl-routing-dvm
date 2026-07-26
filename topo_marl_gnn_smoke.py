#!/usr/bin/env python3
"""Option B smoke test: GNN-actor MARL (multi-hop message passing) on abilene+geant.

Gives MARL the paper-style L-round node-embedding exchange it was missing. Trains one
shared policy across two SEEN backbones on REAL SNDlib traffic and reports, each few
updates, the test-matrix max-util vs OSPF per topology. Success = crossing OSPF parity
(reaching >=0 pt, i.e. util <= OSPF) and staying there stably — unlike the MLP-actor MARL
which sat -50..-150 pt below OSPF throughout.
"""
import argparse

import numpy as np

from marl_routing.marl_gnn import GNNMAPPO
from marl_routing.real_traffic import real_matrices
from marl_routing.topo_agnostic_marl_env import TopoAgnosticMARLEnv

ap = argparse.ArgumentParser()
ap.add_argument("--updates", type=int, default=60)
ap.add_argument("--rollout", type=int, default=4096)
ap.add_argument("--hidden", type=int, default=32)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--stretch", type=int, default=2)
ap.add_argument("--delay-penalty", type=float, default=0.5)
ap.add_argument("--loads", default="2,3,4")
ap.add_argument("--n-per-scale", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()

TOPOS = ["abilene_sndlib", "geant_sndlib"]
loads = [float(x) for x in A.loads.split(",")]


def build_specs(split):
    specs = []
    for t in TOPOS:
        from marl_routing.topology import load as load_topology
        n = load_topology(t).n_nodes
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        mats = real_matrices(t, pairs, loads, n_per_scale=A.n_per_scale, split=split)
        specs.append((t, pairs, mats))
    return specs


train_specs = build_specs("train")
env = TopoAgnosticMARLEnv(train_specs, seed=A.seed, delay_penalty=A.delay_penalty,
                          stretch=A.stretch, normalize_reward=True)

# separate single-topo eval envs on TEST traffic (never touch the trainer env)
eval_envs, tests = {}, {}
for idx, (t, pairs, _) in enumerate(train_specs):
    tmats = real_matrices(t, pairs, [3.0], n_per_scale=1, split="test")
    ev = TopoAgnosticMARLEnv([(t, pairs, tmats)], seed=100 + idx, delay_penalty=A.delay_penalty,
                             stretch=A.stretch, normalize_reward=True)
    eval_envs[t] = ev
    tests[t] = tmats[0]


def rollout_maxutil(ev, policy, rates):
    ev.set_matrix(rates, 0)
    obs, mask, _ = ev.reset()
    done = False
    while not done:
        a = policy(ev)
        obs, mask, _, _, done, info = ev.step(a)
    return info["max_util"]


mappo = GNNMAPPO(env, hidden=A.hidden, rounds=A.rounds, rollout_steps=A.rollout,
                 n_epochs=6, minibatch=512, seed=A.seed)


def ev():
    pol = mappo.act_fn(True)
    parts = []
    for t in TOPOS:
        mu = rollout_maxutil(eval_envs[t], pol, tests[t])
        ospf = eval_envs[t].ospf_max_util(tests[t], 0)
        # convention: positive pt = BETTER than OSPF (lower util); negative = worse
        parts.append(f"{t.split('_')[0]}:{ospf - mu:+.0f}pt(mu{mu:.0f}/ospf{ospf:.0f})")
    return "  ".join(parts)


print(f"[gnn-smoke] rounds={A.rounds} hidden={A.hidden} stretch={A.stretch} "
      f"rollout={A.rollout} updates={A.updates}")
for t in TOPOS:
    print(f"  {t}: OSPF test max-util {eval_envs[t].ospf_max_util(tests[t], 0):.1f}%")
mappo.learn(total_steps=A.rollout * A.updates, log_every=2, eval_fn=ev)
