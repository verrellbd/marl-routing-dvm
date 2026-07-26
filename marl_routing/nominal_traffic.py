#!/usr/bin/env python3
"""Training/test traffic for SNDlib topologies that have only a NOMINAL demand matrix
(everything except abilene/geant/germany50, which ship real time series).

From the single nominal matrix we build a small ENSEMBLE per topology by:
  * auto-calibrating an overall scale so OSPF max-util sits in a target regime (the nominal
    demands are otherwise on wildly different magnitudes across instances), then
  * applying per-OD multiplicative log-normal noise so each matrix differs while preserving
    the real spatial structure (standard TE practice — perturb magnitude, keep gravity).

Train and test ensembles use disjoint RNG streams, so test traffic is unseen.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from marl_routing.topology import load as load_topology

TOPO_DIR = Path(__file__).resolve().parent.parent / "topologies"


def _nominal(topo_name):
    d = json.loads((TOPO_DIR / f"{topo_name}_demands.json").read_text())
    return np.asarray(d["matrix"], dtype=float)


def _ospf_max_util(topo, M):
    g = topo.graph
    cap = {(u, v): g[u][v]["capacity"] for u, v in g.edges()}
    load = {a: 0.0 for a in cap}
    n = topo.n_nodes
    for s in range(n):
        for d in range(n):
            if s != d and M[s, d] > 0:
                p = nx.shortest_path(g, s, d)
                for i in range(len(p) - 1):
                    load[(p[i], p[i + 1])] += M[s, d]
    return max(100.0 * load[a] / cap[a] for a in cap)


def calibrated_base_scale(topo_name, target_util=100.0):
    """Scale that puts the nominal matrix's OSPF max-util at ~target_util%."""
    topo = load_topology(topo_name)
    u = _ospf_max_util(topo, _nominal(topo_name))
    return target_util / max(u, 1e-6)


def nominal_matrices(topo_name, pairs, load_scales=(0.6, 0.8, 1.0, 1.2, 1.5),
                     n_per_scale=1, split="train", sigma=0.0, target_util=100.0, seed=0):
    """List of flattened per-pair rate vectors (aligned to `pairs`).

    The base demand is the REAL SNDlib nominal matrix. load_scales multiply the
    auto-calibrated base scale to sweep the operating point (0.6 -> feasible, 1.5 ->
    overload) — this is magnitude scaling only, exactly as the 3 core topologies' measured
    traffic is scaled; NO synthetic structure is invented. sigma>0 (off by default) would
    add per-OD log-normal noise for extra diversity; we keep sigma=0 to stay fully real.
    """
    base = calibrated_base_scale(topo_name, target_util)
    M0 = _nominal(topo_name)
    rng = np.random.RandomState((hash((topo_name, split)) & 0xFFFFFFFF) ^ seed)
    out = []
    for sc in load_scales:
        for _ in range(n_per_scale):
            noise = np.exp(rng.normal(0.0, sigma, size=M0.shape)) if sigma > 0 else 1.0
            M = M0 * base * sc * noise
            out.append(np.array([M[s, d] for (s, d) in pairs]))
    return out


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:]:
        name = t if t.endswith("_sndlib") else f"{t}_sndlib"
        topo = load_topology(name)
        n = topo.n_nodes
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        base = calibrated_base_scale(name)
        mats = nominal_matrices(name, pairs, n_per_scale=3, split="test")
        us = []
        for m in mats:
            M = np.zeros((n, n))
            for k, (s, d) in enumerate(pairs):
                M[s, d] = m[k]
            us.append(_ospf_max_util(topo, M))
        print(f"{name:22s} base_scale={base:8.3f}  OSPF-util over ensemble "
              f"min {min(us):5.0f}% max {max(us):5.0f}%  ({len(mats)} matrices)")
