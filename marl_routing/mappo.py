#!/usr/bin/env python3
"""
Custom MAPPO (CTDE) for the per-node multi-agent routing environment.

The node-agents take turns (one hop decision per env step) and share one policy
(parameter sharing — the agent identity enters through the local observation). This
makes the cooperative team problem reduce cleanly to PPO over the decision sequence,
with the multi-agent structure expressed as:

  * ACTOR  pi(a | obs, mask)  — sees only LOCAL observation -> decentralized execution.
  * CRITIC V(gstate)          — sees the GLOBAL link state -> centralized training only,
                                low-variance value for credit assignment (CTDE).

Team reward telescopes to -(final max link utilisation) - delay terms, so maximising
return == minimising the network bottleneck, comparable to OSPF / the single-agent GNN.

This module provides ActorCritic + MAPPO (collect -> GAE -> clipped PPO update) and a
smoke test that trains briefly on Abilene and checks it drives max-util below OSPF.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(sizes, act=nn.Tanh, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, gstate_dim, act_dim, hidden=128):
        super().__init__()
        self.actor = _mlp([obs_dim, hidden, hidden, act_dim])
        self.critic = _mlp([gstate_dim, hidden, hidden, 1])
        self.act_dim = act_dim

    def _dist(self, obs, mask):
        logits = self.actor(obs)
        logits = logits.masked_fill(mask < 0.5, -1e9)  # forbid invalid next hops
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs, mask, gstate, deterministic=False):
        d = self._dist(obs, mask)
        a = torch.argmax(d.logits, -1) if deterministic else d.sample()
        return a, d.log_prob(a), self.critic(gstate).squeeze(-1)

    def evaluate(self, obs, mask, gstate, action):
        d = self._dist(obs, mask)
        return d.log_prob(action), d.entropy(), self.critic(gstate).squeeze(-1)


class MAPPO:
    def __init__(self, env, hidden=128, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip=0.2, ent_coef=0.01, vf_coef=0.5, n_epochs=10, minibatch=512,
                 rollout_steps=4096, seed=0, device="cpu"):
        self.env = env
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.n_epochs, self.minibatch, self.rollout_steps = n_epochs, minibatch, rollout_steps
        self.device = device
        torch.manual_seed(seed); np.random.seed(seed)
        self.ac = ActorCritic(env.obs_dim, env.gstate_dim, env.act_dim, hidden).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self._obs = self._mask = self._gstate = None  # rolling env state

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    # ----------------------------------------------------------- rollout + GAE
    def collect(self):
        O, M, G, A, LP, R, V, D = [], [], [], [], [], [], [], []
        if self._obs is None:
            self._obs, self._mask, self._gstate = self.env.reset()
        for _ in range(self.rollout_steps):
            obs = self._t(self._obs); mask = self._t(self._mask); gst = self._t(self._gstate)
            a, lp, v = self.ac.act(obs.unsqueeze(0), mask.unsqueeze(0), gst.unsqueeze(0))
            a_i = int(a.item())
            nobs, nmask, ngst, r, done, info = self.env.step(a_i)
            O.append(self._obs); M.append(self._mask); G.append(self._gstate)
            A.append(a_i); LP.append(float(lp.item())); R.append(float(r))
            V.append(float(v.item())); D.append(float(done))
            if done:
                self._obs, self._mask, self._gstate = self.env.reset()
            else:
                self._obs, self._mask, self._gstate = nobs, nmask, ngst
        # bootstrap value of the state we stopped at
        with torch.no_grad():
            last_v = float(self.ac.critic(self._t(self._gstate).unsqueeze(0)).item())
        return dict(O=np.array(O, np.float32), M=np.array(M, np.float32),
                    G=np.array(G, np.float32), A=np.array(A, np.int64),
                    LP=np.array(LP, np.float32), R=np.array(R, np.float32),
                    V=np.array(V, np.float32), D=np.array(D, np.float32), last_v=last_v)

    def gae(self, R, V, D, last_v):
        n = len(R); adv = np.zeros(n, np.float32); gae = 0.0
        for t in reversed(range(n)):
            next_v = last_v if t == n - 1 else V[t + 1]
            nonterminal = 1.0 - D[t]
            delta = R[t] + self.gamma * next_v * nonterminal - V[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
        return adv, adv + V

    # ------------------------------------------------------------------ update
    def update(self, b):
        adv, ret = self.gae(b["R"], b["V"], b["D"], b["last_v"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O, M, G = self._t(b["O"]), self._t(b["M"]), self._t(b["G"])
        A = torch.as_tensor(b["A"], device=self.device)
        LP = self._t(b["LP"]); ADV = self._t(adv); RET = self._t(ret)
        n = len(A); idx = np.arange(n)
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.minibatch):
                mb = idx[s:s + self.minibatch]
                lp, ent, v = self.ac.evaluate(O[mb], M[mb], G[mb], A[mb])
                ratio = torch.exp(lp - LP[mb])
                pl = -torch.min(ratio * ADV[mb],
                                torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV[mb]).mean()
                vl = ((v - RET[mb]) ** 2).mean()
                loss = pl + self.vf_coef * vl - self.ent_coef * ent.mean()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()
        return float(pl.item()), float(vl.item()), float(ent.mean().item())

    def learn(self, total_steps, log_every=1, eval_fn: Callable = None):
        updates = max(1, total_steps // self.rollout_steps)
        for u in range(updates):
            b = self.collect()
            pl, vl, ent = self.update(b)
            if (u + 1) % log_every == 0 or u == updates - 1:
                extra = f"  {eval_fn():s}" if eval_fn else ""
                print(f"  upd {u+1:3}/{updates}  ep_ret~{b['R'].sum()/max(1,b['D'].sum()):6.2f}"
                      f"  pl {pl:+.3f}  vl {vl:.3f}  ent {ent:.3f}{extra}", flush=True)

    # ---------------------------------------------------------- policy export
    def act_fn(self, deterministic=True):
        def f(obs, mask):
            with torch.no_grad():
                d = self.ac._dist(self._t(obs).unsqueeze(0), self._t(mask).unsqueeze(0))
                return int(torch.argmax(d.logits, -1).item() if deterministic else d.sample().item())
        return f

    def save(self, path):
        torch.save(self.ac.state_dict(), path)

    def load(self, path):
        self.ac.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    # ---- smoke test: brief MAPPO training on Abilene, must beat OSPF analytically ----
    from marl_routing.topology import load as load_topology
    from marl_routing.traffic import generate_matrix
    from marl_routing.multiagent_routing_env import MultiAgentRoutingEnv

    topo = "abilene"; t = load_topology(topo); n = t.n_nodes
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    train_mats = [np.array([generate_matrix(topo, 3.0, seed=s)[a, b] for a, b in pairs])
                  for s in range(8)]
    env = MultiAgentRoutingEnv(topo, pairs, train_mats, delay_penalty=0.0, stretch=1, seed=0)
    # SEPARATE env for evaluation — never touch the trainer's env (it holds the rolling
    # rollout state; resetting it mid-training corrupts collection).
    eval_env = MultiAgentRoutingEnv(topo, pairs, train_mats, stretch=1, seed=1)
    test = np.array([generate_matrix(topo, 3.0, seed=100)[a, b] for a, b in pairs])
    ospf = eval_env.ospf_max_util(test); greedy = eval_env.greedy_max_util(test)

    mappo = MAPPO(env, rollout_steps=4096, n_epochs=6, minibatch=512, seed=0)

    def ev():
        eval_env.rollout_paths(mappo.act_fn(True), test)
        return f"test max-util MARL {eval_env.cur_max:5.1f}% (OSPF {ospf:.1f}, greedy {greedy:.1f})"

    print(f"[smoke] Abilene load3 seed100: OSPF {ospf:.1f}%  greedy {greedy:.1f}%")
    mappo.learn(total_steps=4096 * 12, log_every=2, eval_fn=ev)
