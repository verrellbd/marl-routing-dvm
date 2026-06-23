#!/usr/bin/env python3
"""
Train the multi-agent (MAPPO) per-node routing policy — the genuine MARL system.

Mirrors train_gnn_qos.py (mixed-load training, multi-seed, same topologies) but the
policy is decentralized: each node-agent chooses the next hop from LOCAL observations,
trained with a centralized critic (CTDE). Saves the actor-critic for ns-3 evaluation
(paths are produced by rolling out the agents' hop-by-hop decisions).

  python train_marl.py --topo abilene --loads 2,3,4 --seed 0 --timesteps 600000 --tag _v1
"""
import argparse
from pathlib import Path

import numpy as np

from marl_routing.traffic import generate_matrix
from marl_routing.topology import load as load_topology
from marl_routing.multiagent_routing_env import MultiAgentRoutingEnv
from marl_routing.mappo import MAPPO

_ap = argparse.ArgumentParser()
_ap.add_argument("--topo", default="abilene", help="abilene | geant")
_ap.add_argument("--loads", default="2,3,4", help="comma-list of training load factors")
_ap.add_argument("--seed", type=int, default=0)
_ap.add_argument("--timesteps", type=int, default=600_000)
_ap.add_argument("--delay-penalty", type=float, default=0.5,
                 help="penalty per DETOUR hop (non-progress); keeps short paths when "
                      "uncongested, detours only to relieve congestion. 0.5 = best on Abilene.")
_ap.add_argument("--stretch", type=int, default=1)
_ap.add_argument("--traffic", default="gravity", choices=["gravity", "real"],
                 help="gravity = synthetic model; real = measured SNDlib/Abilene-Zhang "
                      "matrices (load factors become magnitude SCALES on real demand).")
_ap.add_argument("--tag", default="")
_ARGS = _ap.parse_args()

TOPO = _ARGS.topo
TRAIN_LOADS = [float(x) for x in _ARGS.loads.split(",")]
TRAIN_SEEDS = list(range(0, 20))
RESULTS = Path(f"results/{TOPO}_marl{_ARGS.tag}_seed{_ARGS.seed}")
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    topo = load_topology(TOPO); n = topo.n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if _ARGS.traffic == "real":
        from marl_routing.real_traffic import real_matrices
        train_mats = real_matrices(TOPO, pairs, TRAIN_LOADS, n_per_scale=20, split="train")
    else:
        train_mats = [np.array([generate_matrix(TOPO, lf, seed=s)[a, b] for a, b in pairs])
                      for lf in TRAIN_LOADS for s in TRAIN_SEEDS]

    env = MultiAgentRoutingEnv(TOPO, pairs, train_mats, seed=_ARGS.seed,
                               delay_penalty=_ARGS.delay_penalty, stretch=_ARGS.stretch)
    eval_env = MultiAgentRoutingEnv(TOPO, pairs, train_mats, seed=_ARGS.seed + 999,
                                    delay_penalty=_ARGS.delay_penalty, stretch=_ARGS.stretch)
    print(f"[train-marl] topo={TOPO} seed={_ARGS.seed} loads={TRAIN_LOADS} "
          f"delay_penalty={_ARGS.delay_penalty} {len(train_mats)} matrices, "
          f"{n} node-agents, {_ARGS.timesteps} steps")

    mappo = MAPPO(env, hidden=128, rollout_steps=4096, n_epochs=8, minibatch=512,
                  gamma=0.99, ent_coef=0.01, seed=_ARGS.seed)

    # held-out check on UNSEEN traffic (real: later-period 5-min matrices; gravity: unseen seeds)
    if _ARGS.traffic == "real":
        from marl_routing.real_traffic import real_matrices
        held_list = real_matrices(TOPO, pairs, [TRAIN_LOADS[-1]], n_per_scale=5, split="test")
        held_rates = {i: r for i, r in enumerate(held_list)}
    else:
        held = [1000, 1005, 1009, 1013, 1018]
        held_rates = {s: np.array([generate_matrix(TOPO, TRAIN_LOADS[-1], seed=s)[a, b]
                                   for a, b in pairs]) for s in held}

    def ev():
        wins = 0; deltas = []
        for s, r in held_rates.items():
            ospf = eval_env.ospf_max_util(r)
            eval_env.rollout_paths(mappo.act_fn(True), r)
            marl = eval_env.cur_max
            deltas.append(ospf - marl)
            wins += marl < ospf - 0.5
        return f"held-out: MARL beats OSPF {wins}/{len(held_rates)}, mean Δ {np.mean(deltas):+.1f}pt"

    mappo.learn(total_steps=_ARGS.timesteps, log_every=5, eval_fn=ev)
    mappo.save(RESULTS / "mappo_actor_critic.pt")
    print(f"\n[saved] {RESULTS/'mappo_actor_critic.pt'}")

    print("\n[final held-out analytical check vs OSPF + greedy]")
    for s, r in held_rates.items():
        ospf = eval_env.ospf_max_util(r); greedy = eval_env.greedy_max_util(r)
        eval_env.rollout_paths(mappo.act_fn(True), r); marl = eval_env.cur_max
        tag = "WIN" if marl < ospf - 0.5 else ("tie" if abs(marl - ospf) <= 0.5 else "loss")
        print(f"  seed {s}: OSPF {ospf:5.1f}%  greedy {greedy:5.1f}%  MARL {marl:5.1f}%  [{tag}]")


if __name__ == "__main__":
    main()
