"""Draw a topology to PNG using geographic coordinates."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from marl_routing.topology import load


def draw(name: str, out: Path) -> Path:
    topo = load(name)
    g = topo.graph

    pos = {n: (g.nodes[n]["lon"], g.nodes[n]["lat"]) for n in g.nodes}
    labels = {n: g.nodes[n]["name"] for n in g.nodes}

    # collapse directed pairs to one undirected edge for drawing
    undirected = nx.Graph()
    for u, v, d in g.edges(data=True):
        if not undirected.has_edge(u, v):
            undirected.add_edge(u, v, **d)

    caps = [d["capacity"] for _, _, d in undirected.edges(data=True)]
    cmin, cmax = min(caps), max(caps)
    widths = [1.0 + 3.0 * (d["capacity"] - cmin) / max(1, cmax - cmin)
              for _, _, d in undirected.edges(data=True)]
    colors = ["#888" if d["capacity"] == cmax else "#d62728"
              for _, _, d in undirected.edges(data=True)]

    fig, ax = plt.subplots(figsize=(13, 9))
    nx.draw_networkx_edges(undirected, pos, width=widths, edge_color=colors, alpha=0.7, ax=ax)
    nx.draw_networkx_nodes(undirected, pos, node_size=520, node_color="#1f77b4",
                           edgecolors="white", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(undirected, pos, labels=labels, font_size=9,
                            font_color="white", font_weight="bold", ax=ax)

    ax.set_title(f"{topo.name}: {topo.n_nodes} nodes, {undirected.number_of_edges()} links "
                 f"(grey = {cmax} Mbps, red = {cmin} Mbps)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "geant"
    out = Path(__file__).resolve().parent.parent / "results" / f"{name}_topology.png"
    print(f"Wrote {draw(name, out)}")
