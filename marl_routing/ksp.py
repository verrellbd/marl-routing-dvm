"""K-Shortest-Paths (KSP) computation for routing.

For each (src, dst) pair, compute k shortest paths using delay metric.
Used as the agent's action set: agent picks which path to route each flow on.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

from marl_routing.topology import Topology, load


@dataclass(frozen=True)
class PathSpec:
    """Single path from src to dst."""

    path: List[int]  # Node indices [src, ..., dst]
    delay_ms: float  # Total delay in milliseconds
    hops: int  # Number of links


def compute_ksp(
    topo: Topology,
    k: int = 3,
) -> Dict[Tuple[int, int], List[PathSpec]]:
    """Compute k shortest paths for all (src, dst) pairs using delay metric.

    Uses k-Dijkstra: run Dijkstra k times, blocking the last link of the
    previous path each iteration to find alternative routes.

    Args:
        topo: Topology object (has graph with delay weights)
        k: Number of paths to compute per (src, dst)

    Returns:
        Dict mapping (src, dst) to list of k PathSpec objects, sorted by delay
    """
    g = topo.graph
    n_nodes = topo.n_nodes
    ksp_dict: Dict[Tuple[int, int], List[PathSpec]] = {}

    # For each (src, dst) pair
    for src in range(n_nodes):
        for dst in range(n_nodes):
            if src == dst:
                continue  # Skip self-loops

            paths = _ksp_pair(g, src, dst, k)
            ksp_dict[(src, dst)] = paths

    return ksp_dict


def _ksp_pair(
    graph: nx.DiGraph,
    src: int,
    dst: int,
    k: int,
) -> List[PathSpec]:
    """Compute k shortest paths from src to dst using k-Dijkstra.

    Algorithm:
      1. Run Dijkstra once → Path 1 (shortest)
      2. Block the last link of Path 1, run Dijkstra → Path 2
      3. Block the last link of Path 2, run Dijkstra → Path 3
      ... repeat k times

    Args:
        graph: NetworkX DiGraph with edge attribute "delay"
        src: Source node
        dst: Destination node
        k: Number of paths to find

    Returns:
        List of k PathSpec objects, sorted by delay (ascending)
    """
    paths: List[PathSpec] = []
    blocked_edges: set[tuple[int, int]] = set()

    for _ in range(k):
        # Create a copy of graph with blocked edges removed
        g_temp = graph.copy()
        for u, v in blocked_edges:
            if g_temp.has_edge(u, v):
                g_temp.remove_edge(u, v)

        # Run Dijkstra with delay weights
        try:
            path = nx.dijkstra_path(g_temp, src, dst, weight="delay")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No more paths exist
            break

        # Compute path metrics
        delay_ms = 0.0
        for i in range(len(path) - 1):
            delay_ms += g_temp[path[i]][path[i + 1]]["delay"]
        hops = len(path) - 1

        paths.append(PathSpec(path=path, delay_ms=delay_ms, hops=hops))

        # Block the last link for next iteration (to find alternative)
        if len(path) > 1:
            last_link = (path[-2], path[-1])
            blocked_edges.add(last_link)

    return sorted(paths, key=lambda p: p.delay_ms)


def save_ksp_json(ksp_dict: Dict[Tuple[int, int], List[PathSpec]], out: Path) -> Path:
    """Save KSP dict to JSON file."""
    out.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    data = {}
    for (src, dst), paths in ksp_dict.items():
        key = f"{src}_{dst}"
        data[key] = [asdict(p) for p in paths]

    with out.open("w") as f:
        json.dump(data, f, indent=2)

    return out


def load_ksp_json(file: Path) -> Dict[Tuple[int, int], List[dict]]:
    """Load KSP dict from JSON file.

    Returns:
        Dict mapping (src, dst) tuple to list of path dicts
    """
    with file.open("r") as f:
        data = json.load(f)

    ksp_dict = {}
    for key, paths in data.items():
        src, dst = map(int, key.split("_"))
        ksp_dict[(src, dst)] = paths

    return ksp_dict


def summary(ksp_dict: Dict[Tuple[int, int], List[PathSpec]]) -> str:
    """Print KSP statistics."""
    k_values = [len(paths) for paths in ksp_dict.values()]
    max_k = max(k_values) if k_values else 0
    avg_k = sum(k_values) / len(k_values) if k_values else 0

    # Path diversity
    single_path_pairs = sum(1 for paths in ksp_dict.values() if len(paths) == 1)
    multi_path_pairs = len(ksp_dict) - single_path_pairs

    return (
        f"KSP Summary:\n"
        f"  Total (src,dst) pairs: {len(ksp_dict)}\n"
        f"  Pairs with 1 path (no alternatives): {single_path_pairs}\n"
        f"  Pairs with >1 path (have alternatives): {multi_path_pairs}\n"
        f"  Paths per pair: min={min(k_values) if k_values else 0}, "
        f"avg={avg_k:.2f}, max={max_k}\n"
    )


if __name__ == "__main__":
    import sys

    topo_name = sys.argv[1] if len(sys.argv) > 1 else "abilene"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"Computing {k}-shortest-paths for {topo_name}...")
    topo = load(topo_name)
    ksp_dict = compute_ksp(topo, k=k)
    print(summary(ksp_dict))

    out = Path(__file__).resolve().parent.parent / "results" / f"ksp_{topo_name}_k{k}.json"
    save_ksp_json(ksp_dict, out)
    print(f"\nSaved to {out}")
