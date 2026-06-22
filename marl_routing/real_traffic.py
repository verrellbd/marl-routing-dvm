"""
Real traffic loader — uses ONLY the user-provided SNDlib data:
  * static demand matrices parsed from topologies/<name>_sndlib_demands.json
  * dynamic Abilene matrices (Yin Zhang, 5-min over 6 months) read straight from the
    .tgz of SNDlib-native demand files in topologies/.

Real measured traffic on the over-provisioned backbones is LIGHT vs link capacity, so
(as in standard TE studies) we keep the real spatial/temporal STRUCTURE and scale the
magnitude by `load_scale` to set the operating point. Nothing here is synthetic — the
matrices are the measured demands; only the overall level is scaled.
"""
from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import numpy as np

from marl_routing.topology import load as load_topology

TOPO_DIR = Path(__file__).resolve().parent.parent / "topologies"
# dynamic-traffic archives (real measured demand-matrix time series), keyed by topo base
TGZ = {
    "abilene": TOPO_DIR / "directed-abilene-zhang-5min-over-6months-ALL-native (1).tgz",
    "geant": TOPO_DIR / "directed-geant-uhlig-15min-over-4months-ALL-native.tgz",
    "germany50": TOPO_DIR / "directed-germany50-DFN-aggregated-5min-over-1day-native.tgz",
}
ABILENE_TGZ = TGZ["abilene"]  # back-compat


def _tgz_for(topo_name: str) -> Path:
    for base, path in TGZ.items():
        if topo_name.startswith(base):
            return path
    raise KeyError(f"no dynamic-traffic archive for {topo_name}")


def _name2id(topo_name: str):
    """Map SNDlib node names -> our integer ids (topology JSON order)."""
    j = json.loads((TOPO_DIR / f"{topo_name}.json").read_text())
    return {nd["name"]: nd["id"] for nd in j["nodes"]}, len(j["nodes"])


def _parse_demands(text: str, name2id: dict, n: int) -> np.ndarray:
    """Parse a SNDlib-native DEMANDS section into an n x n matrix (Mbps)."""
    m = re.search(r"DEMANDS\s*\((.*?)\n\)", text, re.S)
    D = np.zeros((n, n))
    if not m:
        return D
    for ln in m.group(1).splitlines():
        g = re.match(r"\s*\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\S+\s+([\d.]+)", ln)
        if g and g.group(1) in name2id and g.group(2) in name2id:
            D[name2id[g.group(1)], name2id[g.group(2)]] = float(g.group(3))
    return D


def load_static(topo_name: str, load_scale: float = 1.0) -> np.ndarray:
    """The single real demand matrix shipped in the SNDlib instance (geant/germany50)."""
    D = np.array(json.loads((TOPO_DIR / f"{topo_name}_demands.json").read_text())["matrix"])
    return D * load_scale


def sample_abilene_dynamic(k: int = 60, seed: int = 0, load_scale: float = 1.0,
                           topo_name: str = "abilene_sndlib"):
    """Sample k real 5-min Abilene matrices spread evenly across the 6-month archive.
    Returns a list of (n x n) matrices (Mbps, scaled). Deterministic given seed."""
    name2id, n = _name2id(topo_name)
    with tarfile.open(ABILENE_TGZ, "r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.name.endswith(".txt")),
                         key=lambda m: m.name)
        idx = np.linspace(0, len(members) - 1, k).astype(int)
        rng = np.random.RandomState(seed)
        idx = np.unique(np.clip(idx + rng.randint(-2, 3, size=k), 0, len(members) - 1))
        mats = []
        for i in idx:
            text = tar.extractfile(members[i]).read().decode()
            mats.append(_parse_demands(text, name2id, n) * load_scale)
    return mats, [members[i].name.split("zhang-5min-")[-1].replace(".txt", "") for i in idx]


def real_matrices(topo_name: str, pairs, load_scales, n_per_scale: int = 20,
                  seed: int = 0, split: str = "train"):
    """Return a list of flattened pair-rate vectors from REAL traffic, scaled by each
    load factor. Abilene uses the dynamic 5-min archive with a TEMPORAL split (train =
    first 70% of the 6-month timeline, test = last 30% -> generalization to unseen real
    traffic). GEANT/Germany50 (one real matrix each) use that matrix at each load scale.
    """
    name2id, n = _name2id(topo_name)
    tgz = _tgz_for(topo_name)
    with tarfile.open(tgz, "r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.name.endswith(".txt")),
                         key=lambda m: m.name)
        cut = int(0.7 * len(members))            # temporal split: first 70% train, last 30% test
        pool = members[:cut] if split == "train" else members[cut:]
        rng = np.random.RandomState(seed)
        out = []
        for sc in load_scales:
            picks = list(rng.permutation(len(pool)))   # draw until we have enough non-empty
            got = 0
            for i in picks:
                if got >= n_per_scale:
                    break
                text = tar.extractfile(pool[i]).read().decode()
                D = _parse_demands(text, name2id, n) * sc
                vec = np.array([D[a, b] for a, b in pairs])
                if vec.sum() <= 0:                      # skip empty snapshots (some are zero)
                    continue
                out.append(vec); got += 1
    return out


if __name__ == "__main__":
    # characterise the real Abilene dynamic traffic: magnitude range + congesting scale
    import networkx as nx
    topo = load_topology("abilene_sndlib"); G = topo.graph; n = topo.n_nodes
    arcs = list(G.edges()); ai = {a: i for i, a in enumerate(arcs)}
    cap = np.array([G[u][v]["capacity"] for u, v in arcs], float)

    def ospf_maxutil(D):
        load = np.zeros(len(arcs))
        for s in range(n):
            for d in range(n):
                if D[s, d] > 0:
                    p = nx.shortest_path(G, s, d)
                    for i in range(len(p) - 1):
                        load[ai[(p[i], p[i + 1])]] += D[s, d]
        return (100 * load / cap).max()

    mats, stamps = sample_abilene_dynamic(k=40, seed=0)
    totals = [M.sum() for M in mats]
    utils = [ospf_maxutil(M) for M in mats]
    print(f"sampled {len(mats)} real 5-min matrices, span {stamps[0]} .. {stamps[-1]}")
    print(f"  total demand (Mbps): min {min(totals):.0f}  median {np.median(totals):.0f}  max {max(totals):.0f}")
    print(f"  raw OSPF max-util %: min {min(utils):.2f}  median {np.median(utils):.2f}  max {max(utils):.2f}")
    med = np.median(utils)
    print(f"  -> load_scale for ~120% OSPF (congested): ~{120/med:.0f}x")
