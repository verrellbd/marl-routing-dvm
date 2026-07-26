#!/usr/bin/env python3
"""Parse SNDlib *native* topology files into our topology JSON schema + nominal demand.

The 3 headline networks (abilene/geant/germany50) ship real measured traffic TIME SERIES
(the .tgz archives) and hand-set realistic capacities. The other SNDlib instances give us
STRUCTURE + a single nominal demand matrix each — enough to use them as extra training
topologies and as zero-shot generalisation targets.

Capacity convention: SNDlib models capacity as a design variable (most instances have
pre-installed capacity 0 + modular options). For a congestion study the max-util metric is
a RATIO, so absolute capacity is immaterial once demand is scaled to the operating regime;
we therefore assign a UNIFORM capacity (default 10 Gbps) to every link. Delay is derived
from great-circle distance between node coordinates (5 us/km propagation), or a nominal
constant when coordinates are absent.

  python -m marl_routing.sndlib_native nobel-eu        # -> topologies/nobel-eu_sndlib.json
                                                       #    + nobel-eu_sndlib_demands.json
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import networkx as nx

TOPO_DIR = Path(__file__).resolve().parent.parent / "topologies"
NATIVE_DIR = TOPO_DIR / "sndlib-networks-native"
UNIFORM_CAP_MBPS = 10000.0     # 10 Gbps on every link
PROP_MS_PER_KM = 0.005         # ~5 us/km fibre propagation
# these ship real measured traffic + hand-set realistic capacities; NEVER regenerate them
# with the uniform-capacity convention (would silently break existing results).
PROTECTED = {"abilene_sndlib", "geant_sndlib", "germany50_sndlib"}


def _section(text, name):
    m = re.search(name + r"\s*\((.*?)\n\)", text, re.S)
    return m.group(1) if m else ""


def _haversine_km(a, b):
    (lon1, lat1), (lon2, lat2) = a, b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def parse_native(base: str):
    """Return (nodes, links, demand) from topologies/sndlib-networks-native/<base>.txt.
    nodes: [(name, lon|None, lat|None)]; links: [(u_name, v_name)]; demand: {(u,v): mbps}."""
    text = (NATIVE_DIR / f"{base}.txt").read_text()
    nodes = []
    for ln in _section(text, "NODES").splitlines():
        g = re.match(r"\s*(\S+)\s*(?:\(\s*([-\d.]+)\s+([-\d.]+)\s*\))?", ln)
        if g and g.group(1):
            lon = float(g.group(2)) if g.group(2) else None
            lat = float(g.group(3)) if g.group(3) else None
            nodes.append((g.group(1), lon, lat))
    links = []
    for ln in _section(text, "LINKS").splitlines():
        g = re.match(r"\s*\S+\s*\(\s*(\S+)\s+(\S+)\s*\)", ln)
        if g:
            links.append((g.group(1), g.group(2)))
    demand = {}
    for ln in _section(text, "DEMANDS").splitlines():
        g = re.match(r"\s*\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\S+\s+([\d.]+)", ln)
        if g:
            demand[(g.group(1), g.group(2))] = float(g.group(3))
    return nodes, links, demand


def to_json(base: str, uniform_cap=UNIFORM_CAP_MBPS, out_name=None):
    """Write topologies/<out_name>.json (+ _demands.json) in our schema. Returns out_name."""
    nodes, links, demand = parse_native(base)
    name2id = {nm: i for i, (nm, _, _) in enumerate(nodes)}
    coord = {i: (lo, la) for i, (_, lo, la) in enumerate(nodes)}
    have_coords = all(coord[i][0] is not None for i in coord)

    jnodes = []
    for i, (nm, lo, la) in enumerate(nodes):
        d = {"id": i, "name": nm}
        if lo is not None:
            d["lon"], d["lat"] = lo, la
        jnodes.append(d)

    jlinks = []
    for (u, v) in links:
        if u not in name2id or v not in name2id:
            continue
        iu, iv = name2id[u], name2id[v]
        if have_coords:
            km = _haversine_km(coord[iu], coord[iv])
            delay = max(0.1, round(km * PROP_MS_PER_KM, 3))
        else:
            delay = 1.0
        jlinks.append({"src": iu, "dst": iv, "capacity": float(uniform_cap), "delay": delay})

    n = len(nodes)
    out_name = out_name or f"{base}_sndlib"
    if out_name in PROTECTED:
        raise ValueError(f"{out_name} is protected (real traffic + hand-set capacities); "
                         f"refusing to overwrite with the uniform-capacity convention")
    doc = {"name": out_name, "capacity_unit": "Mbps", "delay_unit": "ms",
           "nodes": jnodes, "links": jlinks}

    # validate strong connectivity of the bidirected graph before writing
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for l in jlinks:
        g.add_edge(l["src"], l["dst"]); g.add_edge(l["dst"], l["src"])
    if not nx.is_strongly_connected(g):
        comps = list(nx.strongly_connected_components(g))
        raise ValueError(f"{base}: not strongly connected ({len(comps)} components, "
                         f"largest {max(len(c) for c in comps)}/{n})")

    (TOPO_DIR / f"{out_name}.json").write_text(json.dumps(doc, indent=2))

    # nominal demand matrix (Mbps) as {name}_demands.json for load_static / matrix gen
    M = [[0.0] * n for _ in range(n)]
    for (u, v), r in demand.items():
        if u in name2id and v in name2id:
            M[name2id[u]][name2id[v]] = r
    (TOPO_DIR / f"{out_name}_demands.json").write_text(
        json.dumps({"matrix": M, "n_demands": len(demand)}))

    return out_name, n, len(jlinks), len(demand), nx.diameter(g)


if __name__ == "__main__":
    import sys
    for base in sys.argv[1:]:
        try:
            nm, n, l, d, diam = to_json(base)
            print(f"[ok] {base:16s} -> {nm}.json  {n} nodes, {l} arcs, "
                  f"{d} demands, diameter {diam}")
        except Exception as e:
            print(f"[SKIP] {base:16s} {type(e).__name__}: {e}")
