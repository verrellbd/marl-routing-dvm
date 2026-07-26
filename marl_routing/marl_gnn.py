#!/usr/bin/env python3
"""GNN-actor MAPPO for the topology-agnostic per-node routing MARL (Option B).

This is the "give MARL a fair shot" variant motivated by the closest prior work
(Alanazi & Zareei, MA-DQN+GNN for MANETs). Their agents run L layers of GNN message
passing so each node sees a MULTI-HOP neighbourhood before choosing a next hop; our
earlier MARL actor was a plain MLP on immediate-incident-link features only (more myopic).

Here the ACTOR is a shared-weight GNN:
  * encode per-node structural features -> h0
  * R rounds of mean-aggregation message passing (h += relu(W[h, A h]))
  * for the deciding node, score each neighbour from [h_cur, h_nbr, local_arc_feats]
The CRITIC stays a centralized MLP on the global padded arc state (CTDE, unchanged idea).

Everything is topology-invariant: adjacency is structure only (no node identity), node
features are structural (is_current / is_dst / dist / util / headroom / degree). One
network trains on a MIX of topologies and can transfer zero-shot.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from marl_routing.topo_agnostic_marl_env import MAX_DEG, NF


def _mlp(sizes, act=nn.Tanh, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class GNNActorCritic(nn.Module):
    def __init__(self, node_f, gstate_dim, hidden=32, rounds=3, act_dim=MAX_DEG,
                 local_f=NF):
        super().__init__()
        self.rounds = rounds
        self.act_dim = act_dim
        self.local_f = local_f
        self.enc = nn.Linear(node_f, hidden)
        self.msg = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(rounds)])
        # score a neighbour from [h_current, h_neighbour, local arc feats] -> scalar logit
        self.head = _mlp([2 * hidden + local_f, hidden, 1])
        self.critic = _mlp([gstate_dim, hidden, hidden, 1])

    def _embed(self, node_feat, adj):
        # node_feat [B,N,F], adj [B,N,N] (row-normalised, self-loops) -> h [B,N,H]
        h = torch.relu(self.enc(node_feat))
        for lin in self.msg:
            agg = torch.bmm(adj, h)
            h = h + torch.relu(lin(torch.cat([h, agg], dim=-1)))
        return h

    def _logits(self, node_feat, adj, cur_idx, nbr_idx, local, mask):
        h = self._embed(node_feat, adj)                       # [B,N,H]
        B, _, H = h.shape
        ar = torch.arange(B, device=h.device)
        h_cur = h[ar, cur_idx]                                 # [B,H]
        safe = nbr_idx.clamp(min=0)                            # [B,MAX_DEG]
        h_nbr = torch.gather(h, 1, safe.unsqueeze(-1).expand(-1, -1, H))  # [B,MAX_DEG,H]
        local_f = local[:, :MAX_DEG * self.local_f].reshape(B, MAX_DEG, self.local_f)
        cur_rep = h_cur.unsqueeze(1).expand(-1, MAX_DEG, -1)
        head_in = torch.cat([cur_rep, h_nbr, local_f], dim=-1)  # [B,MAX_DEG,2H+NF]
        logit = self.head(head_in).squeeze(-1)                 # [B,MAX_DEG]
        return logit.masked_fill(mask < 0.5, -1e9)

    def _dist(self, node_feat, adj, cur_idx, nbr_idx, local, mask):
        return Categorical(logits=self._logits(node_feat, adj, cur_idx, nbr_idx, local, mask))

    @torch.no_grad()
    def act(self, nf, adj, cur, nbr, local, mask, gstate, deterministic=False):
        d = self._dist(nf, adj, cur, nbr, local, mask)
        a = torch.argmax(d.logits, -1) if deterministic else d.sample()
        return a, d.log_prob(a), self.critic(gstate).squeeze(-1)

    def evaluate(self, nf, adj, cur, nbr, local, mask, gstate, action):
        d = self._dist(nf, adj, cur, nbr, local, mask)
        return d.log_prob(action), d.entropy(), self.critic(gstate).squeeze(-1)


class GNNMAPPO:
    def __init__(self, env, hidden=32, rounds=3, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip=0.2, ent_coef=0.01, vf_coef=0.5, n_epochs=10, minibatch=512,
                 rollout_steps=4096, seed=0, device="cpu"):
        self.env = env
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.n_epochs, self.minibatch, self.rollout_steps = n_epochs, minibatch, rollout_steps
        self.device = device
        torch.manual_seed(seed); np.random.seed(seed)
        self.ac = GNNActorCritic(env.node_f_dim, env.gstate_dim, hidden=hidden,
                                 rounds=rounds, act_dim=env.act_dim).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self._need_reset = True

    def _tf(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _ti(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.int64, device=self.device)

    def _cur_graph(self):
        return self.env._graph_obs()

    def collect(self):
        NF_, ADJ, CUR, NBR, LOC, MSK, G, A, LP, R, V, D = ([] for _ in range(12))
        if self._need_reset:
            self._obs, self._mask, self._gstate = self.env.reset()
            self._need_reset = False
        for _ in range(self.rollout_steps):
            nf, adj, _, cur, nbr = self._cur_graph()
            loc, msk, gst = self._obs, self._mask, self._gstate
            a, lp, v = self.ac.act(
                self._tf(nf).unsqueeze(0), self._tf(adj).unsqueeze(0),
                self._ti([cur]), self._ti(nbr).unsqueeze(0),
                self._tf(loc).unsqueeze(0), self._tf(msk).unsqueeze(0),
                self._tf(gst).unsqueeze(0))
            a_i = int(a.item())
            nobs, nmask, ngst, r, done, info = self.env.step(a_i)
            NF_.append(nf); ADJ.append(adj); CUR.append(cur); NBR.append(nbr)
            LOC.append(loc); MSK.append(msk); G.append(gst)
            A.append(a_i); LP.append(float(lp.item())); R.append(float(r))
            V.append(float(v.item())); D.append(float(done))
            if done:
                self._obs, self._mask, self._gstate = self.env.reset()
            else:
                self._obs, self._mask, self._gstate = nobs, nmask, ngst
        with torch.no_grad():
            last_v = float(self.ac.critic(self._tf(self._gstate).unsqueeze(0)).item())
        return dict(NF=np.array(NF_, np.float32), ADJ=np.array(ADJ, np.float32),
                    CUR=np.array(CUR, np.int64), NBR=np.array(NBR, np.int64),
                    LOC=np.array(LOC, np.float32), MSK=np.array(MSK, np.float32),
                    G=np.array(G, np.float32), A=np.array(A, np.int64),
                    LP=np.array(LP, np.float32), R=np.array(R, np.float32),
                    V=np.array(V, np.float32), D=np.array(D, np.float32), last_v=last_v)

    def gae(self, R, V, D, last_v):
        n = len(R); adv = np.zeros(n, np.float32); g = 0.0
        for t in reversed(range(n)):
            next_v = last_v if t == n - 1 else V[t + 1]
            nt = 1.0 - D[t]
            delta = R[t] + self.gamma * next_v * nt - V[t]
            g = delta + self.gamma * self.lam * nt * g
            adv[t] = g
        return adv, adv + V

    def update(self, b):
        adv, ret = self.gae(b["R"], b["V"], b["D"], b["last_v"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        NF_ = self._tf(b["NF"]); ADJ = self._tf(b["ADJ"]); CUR = self._ti(b["CUR"])
        NBR = self._ti(b["NBR"]); LOC = self._tf(b["LOC"]); MSK = self._tf(b["MSK"])
        G = self._tf(b["G"]); A = self._ti(b["A"]); LP = self._tf(b["LP"])
        ADV = self._tf(adv); RET = self._tf(ret)
        n = len(A); idx = np.arange(n)
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.minibatch):
                mb = idx[s:s + self.minibatch]
                lp, ent, v = self.ac.evaluate(NF_[mb], ADJ[mb], CUR[mb], NBR[mb],
                                              LOC[mb], MSK[mb], G[mb], A[mb])
                ratio = torch.exp(lp - LP[mb])
                pl = -torch.min(ratio * ADV[mb],
                                torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV[mb]).mean()
                vl = ((v - RET[mb]) ** 2).mean()
                loss = pl + self.vf_coef * vl - self.ent_coef * ent.mean()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()
        return float(pl.item()), float(vl.item()), float(ent.mean().item())

    def learn(self, total_steps, log_every=1, eval_fn: Callable = None,
              ckpt_dir=None, ckpt_every=None):
        updates = max(1, total_steps // self.rollout_steps)
        if ckpt_dir is not None:
            from pathlib import Path
            ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
        for u in range(updates):
            b = self.collect()
            pl, vl, ent = self.update(b)
            self._need_reset = True  # eval_fn resets env; force fresh rollout next collect
            if (u + 1) % log_every == 0 or u == updates - 1:
                extra = f"  {eval_fn():s}" if eval_fn else ""
                print(f"  upd {u+1:3}/{updates}  ep_ret~{b['R'].sum()/max(1,b['D'].sum()):7.2f}"
                      f"  pl {pl:+.3f}  vl {vl:.3f}  ent {ent:.3f}{extra}", flush=True)
            if ckpt_dir is not None and ckpt_every and ((u + 1) % ckpt_every == 0 or u == updates - 1):
                self.save(ckpt_dir / f"ckpt_upd{u+1:04d}.pt")

    # policy export: act_fn drives an eval env via its _graph_obs + obs/mask
    def act_fn(self, deterministic=True):
        def f(env):
            nf, adj, _, cur, nbr = env._graph_obs()
            with torch.no_grad():
                d = self.ac._dist(self._tf(nf).unsqueeze(0), self._tf(adj).unsqueeze(0),
                                  self._ti([cur]), self._ti(nbr).unsqueeze(0),
                                  self._tf(env._obs()).unsqueeze(0),
                                  self._tf(env._mask()).unsqueeze(0))
                return int(torch.argmax(d.logits, -1).item() if deterministic
                           else d.sample().item())
        return f

    def save(self, path):
        torch.save(self.ac.state_dict(), path)

    def load(self, path):
        self.ac.load_state_dict(torch.load(path, map_location=self.device))
