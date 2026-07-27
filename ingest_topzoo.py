#!/usr/bin/env python3
"""Convert Internet Topology Zoo GML files into our topology JSON format.

Topology Zoo gives STRUCTURE only (no capacity, no traffic). We:
  * read the GML, collapse to a simple undirected graph (drop self-loops / parallels),
  * keep the largest connected component,
  * add both directions per edge (our env is directed),
  * assign uniform capacity (10 Gbps) since Zoo has none,
  * FILTER to what the topology-agnostic env can hold (<=56 nodes, <=176 arcs, deg<=12)
    and require strong connectivity,
  * EXCLUDE anything matching our 3 evaluation backbones (abilene/geant/germany).

Writes topologies/zoo_<name>.json  (loadable by marl_routing.topology.load).
"""
import glob
import json
import os
import re

import networkx as nx

# keep filter caps in lock-step with the env's padding sizes
from marl_routing.topo_agnostic_marl_env import N_MAX, A_MAX, MAX_DEG

SRC = "topologies/topology-zoo"
OUT = "topologies"
CAP_MBPS = 10000.0          # uniform 10 Gbps (Zoo has no capacities)
DELAY = 1.0
N_MIN = 3                   # include everything down to tiny graphs (only 3 eval nets excluded)
EXCLUDE = re.compile(r"abilene|geant|géant|germany", re.I)


def _read_gml_any(path):
    """Read a Zoo GML tolerating parallel edges (force multigraph, then collapse)."""
    for label in ("id", None):
        try:
            return nx.read_gml(path, label=label)
        except Exception:
            pass
    # duplicate-edge failures: inject `multigraph 1` so nx keeps parallels, then collapse
    txt = open(path, encoding="utf-8", errors="ignore").read()
    txt = re.sub(r"(graph\s*\[)", r"\1\n  multigraph 1", txt, count=1)
    return nx.parse_gml(txt, label="id")


def load_simple(path):
    """GML -> simple undirected Graph with integer nodes 0..n-1 (largest CC)."""
    g = _read_gml_any(path)
    g = nx.Graph(g)                       # collapse multi-edges to simple
    g.remove_edges_from(nx.selfloop_edges(g))
    if g.number_of_nodes() == 0:
        return None
    cc = max(nx.connected_components(g), key=len)
    g = g.subgraph(cc).copy()
    return nx.convert_node_labels_to_integers(g)


def main():
    kept, skipped = [], {}
    for path in sorted(glob.glob(f"{SRC}/*.gml")):
        name = os.path.basename(path)[:-4]
        if EXCLUDE.search(name):
            skipped[name] = "excluded (eval backbone)"; continue
        try:
            g = load_simple(path)
        except Exception as e:
            skipped[name] = f"parse error: {e}"; continue
        if g is None:
            skipped[name] = "empty"; continue
        n = g.number_of_nodes(); arcs = 2 * g.number_of_edges()
        md = max((d for _, d in g.degree()), default=0)
        if not (N_MIN <= n <= N_MAX):
            skipped[name] = f"n={n} out of [{N_MIN},{N_MAX}]"; continue
        if arcs > A_MAX:
            skipped[name] = f"arcs={arcs}>{A_MAX}"; continue
        if md > MAX_DEG:
            skipped[name] = f"deg={md}>{MAX_DEG}"; continue
        # directed version (both ways) must be strongly connected
        dg = nx.DiGraph()
        dg.add_nodes_from(g.nodes())
        for u, v in g.edges():
            dg.add_edge(u, v); dg.add_edge(v, u)
        if not nx.is_strongly_connected(dg):
            skipped[name] = "not strongly connected"; continue

        doc = {
            "name": f"zoo_{name}",
            "capacity_unit": "Mbps", "delay_unit": "ms",
            "nodes": [{"id": i, "name": str(i)} for i in range(n)],
            "links": [{"src": int(u), "dst": int(v), "capacity": CAP_MBPS, "delay": DELAY}
                      for u, v in g.edges()],
        }
        with open(f"{OUT}/zoo_{name}.json", "w") as f:
            json.dump(doc, f)
        kept.append((f"zoo_{name}", n, arcs, md))

    print(f"[ingest] kept {len(kept)} topologies, skipped {len(skipped)}")
    print("  node range:", min(k[1] for k in kept), "-", max(k[1] for k in kept))
    # summarise skip reasons
    from collections import Counter
    reasons = Counter(v.split(":")[0].split("(")[0].strip() for v in skipped.values())
    for r, c in reasons.most_common():
        print(f"  skip: {r}  x{c}")
    # write the manifest of trainable names
    names = [k[0] for k in kept]
    with open(f"{OUT}/zoo_manifest.json", "w") as f:
        json.dump(names, f, indent=1)
    print(f"[manifest] {OUT}/zoo_manifest.json ({len(names)} topos)")


if __name__ == "__main__":
    main()
