"""Topology loader: parses topology JSON into a NetworkX graph.

Single source of truth shared by the OSPF baseline trace, the LP optimum
solver, and the MARL routing environment. The ns-3 C++ side reads the same
JSON via nlohmann/json so link IDs stay aligned across stacks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

TOPO_DIR = Path(__file__).resolve().parent.parent / "topologies"


@dataclass(frozen=True)
class Topology:
    name: str
    graph: nx.DiGraph  # directed: each undirected JSON link becomes two arcs
    capacity_unit: str
    delay_unit: str

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_directed_links(self) -> int:
        return self.graph.number_of_edges()

    def node_name(self, nid: int) -> str:
        return self.graph.nodes[nid]["name"]


def load(name: str) -> Topology:
    path = TOPO_DIR / f"{name}.json"
    with path.open() as f:
        data = json.load(f)

    g = nx.DiGraph()
    for n in data["nodes"]:
        g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})

    for link in data["links"]:
        u, v = link["src"], link["dst"]
        attrs = {"capacity": link["capacity"], "delay": link["delay"]}
        g.add_edge(u, v, **attrs)
        g.add_edge(v, u, **attrs)

    topo = Topology(
        name=data["name"],
        graph=g,
        capacity_unit=data["capacity_unit"],
        delay_unit=data["delay_unit"],
    )
    _validate(topo)
    return topo


def _validate(topo: Topology) -> None:
    g = topo.graph
    if not nx.is_strongly_connected(g):
        comps = list(nx.strongly_connected_components(g))
        raise ValueError(f"{topo.name} is not strongly connected: {len(comps)} components")
    for u, v, d in g.edges(data=True):
        if d["capacity"] <= 0 or d["delay"] < 0:
            raise ValueError(f"bad link {u}->{v}: {d}")


def summary(topo: Topology) -> str:
    g = topo.graph
    degrees = [d for _, d in g.degree()]
    caps = [d["capacity"] for _, _, d in g.edges(data=True)]
    return (
        f"{topo.name}: {topo.n_nodes} nodes, {topo.n_directed_links // 2} undirected links\n"
        f"  degree min/mean/max = {min(degrees)}/{sum(degrees)/len(degrees):.2f}/{max(degrees)}\n"
        f"  capacity min/max    = {min(caps)}/{max(caps)} {topo.capacity_unit}\n"
        f"  diameter (hops)     = {nx.diameter(g)}"
    )


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "geant"
    print(summary(load(name)))
