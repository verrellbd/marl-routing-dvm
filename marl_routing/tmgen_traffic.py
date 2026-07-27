#!/usr/bin/env python3
"""Modulated-gravity traffic matrices (via TMgen) for any topology.

TMgen implements Roughan's modulated-gravity model: a gravity spatial pattern (traffic
between two PoPs proportional to the product of their volumes) modulated by a diurnal
temporal signal. Each independent draw gives a DIFFERENT spatial hot-spot pattern, so we
draw several per topology to get genuine spatial diversity — unlike scaling one fixed
matrix. Each pattern is then calibrated so OSPF max-util hits a target, and swept across
load factors into the feasible->overload regime.

Reference: M. Roughan, "Simplifying the synthesis of Internet traffic matrices",
ACM SIGCOMM CCR 2005; TMgen (Heorhiadi et al.).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import networkx as nx
from tmgen.models import modulated_gravity_tm

from marl_routing.topology import load as load_topology


def _ospf_max_util(G, cap, arc_index, pairs, rates) -> float:
    load = np.zeros(len(arc_index))
    for i, (s, d) in enumerate(pairs):
        if rates[i] > 0:
            p = nx.shortest_path(G, s, d)
            for j in range(len(p) - 1):
                load[arc_index[(p[j], p[j + 1])]] += rates[i]
    return float((100.0 * load / cap).max())


def tmgen_matrices(topo_name, pairs, n_patterns=4, load_scales=(0.7, 1.0, 1.3),
                   seed=0, target_util=100.0, spatial_variance=600.0,
                   mean_traffic=1000.0):
    """Return a list of flattened per-pair rate vectors (aligned to `pairs`).

    n_patterns independent modulated-gravity draws (distinct spatial patterns) x
    load_scales (volume sweep). Each pattern is scaled so its OSPF max-util == target_util
    at load 1.0, then multiplied by each load factor -> spans feasible..overload.
    """
    topo = load_topology(topo_name)
    G = topo.graph
    n = topo.n_nodes
    arcs = list(G.edges())
    arc_index = {a: i for i, a in enumerate(arcs)}
    cap = np.array([G[u][v]["capacity"] for (u, v) in arcs], float)

    out = []
    for k in range(n_patterns):
        np.random.seed(seed * 10007 + k)               # reproducible, distinct per pattern
        tm = modulated_gravity_tm(num_nodes=n, num_tms=1, mean_traffic=mean_traffic,
                                  spatial_variance=spatial_variance,
                                  temporal_variance=0.01)
        A = np.asarray(tm.matrix)[:, :, 0].astype(float)
        np.fill_diagonal(A, 0.0)
        A[A < 0] = 0.0
        raw = np.array([A[s, d] for (s, d) in pairs])
        u = _ospf_max_util(G, cap, arc_index, pairs, raw)
        base = (target_util / u) if u > 0 else 1.0
        for sc in load_scales:
            out.append(raw * base * sc)
    return out


def save_traffic(specs, outdir, meta: dict):
    """Persist generated training traffic as a reproducible artifact.

    specs: list of (topo_name, pairs, matrices). Writes one compressed .npz per topology
    (matrices as float32 rate-vectors + the pair index) plus a manifest.json documenting
    the exact generation parameters so the corpus is fully reproducible from seed alone.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    index = []
    for name, pairs, mats in specs:
        M = np.asarray(mats, dtype=np.float32)                 # (n_matrices, n_pairs)
        P = np.asarray(pairs, dtype=np.int32)                  # (n_pairs, 2)
        np.savez_compressed(outdir / f"{name}.npz", matrices=M, pairs=P)
        index.append({"topo": name, "n_matrices": int(M.shape[0]),
                      "n_pairs": int(M.shape[1]),
                      "nonzero_flows_per_matrix": [int((m > 0).sum()) for m in M]})
    (outdir / "manifest.json").write_text(json.dumps(
        {"meta": meta, "topologies": index}, indent=2))
    return outdir


def load_traffic(outdir):
    """Inverse of save_traffic -> list of (topo_name, pairs, matrices)."""
    outdir = Path(outdir)
    man = json.loads((outdir / "manifest.json").read_text())
    specs = []
    for entry in man["topologies"]:
        name = entry["topo"]
        z = np.load(outdir / f"{name}.npz")
        pairs = [tuple(int(x) for x in p) for p in z["pairs"]]
        mats = [z["matrices"][i] for i in range(z["matrices"].shape[0])]
        specs.append((name, pairs, mats))
    return specs
