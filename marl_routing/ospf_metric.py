"""OSPF cost metrics.

Our exporters originally routed the OSPF baseline by HOP COUNT (`nx.shortest_path` with no
weight). Real OSPF uses cost = reference-bandwidth / link-bandwidth, so a slow link is
expensive and gets routed around. On a topology with uniform link capacities the two are
identical; on a heterogeneous one they are not, and hop-count OSPF is a much weaker
baseline than anything deployed.

Concretely on abilene_sndlib (14x9.92G + 1x2.48G) the 2.48G link carries 83.6% mean
utilisation under hop-count OSPF and 0.0% under weighted OSPF.

ECMP inherits the same metric: real ECMP splits among paths of equal OSPF COST, not equal
hop count.
"""
import networkx as nx


def weighted_graph(G, ref_bw=None):
    """Copy of G with edge attribute 'w' = ref_bw / capacity (OSPF cost).

    ref_bw defaults to the largest capacity present, so the fastest links get cost 1.
    """
    if ref_bw is None:
        ref_bw = max(G[u][v]["capacity"] for u, v in G.edges())
    W = G.copy()
    for u, v in W.edges():
        W[u][v]["w"] = ref_bw / W[u][v]["capacity"]
    return W


def dist_to_all(G, W, metric="hop"):
    """For every destination d: {node: distance from node to d} under `metric`.

    Under "weighted" this is OSPF COST distance, not hops. The learned policies use it for
    their progress/detour test, so switching the metric here is what makes them
    capacity-aware: on abilene the 2.48G link costs 4, so a route through it looks 4 units
    "further" than a 1-hop fast link and the policy's candidate set stops depending on it.
    """
    R = G.reverse(copy=False) if metric != "weighted" else W.reverse(copy=False)
    if metric == "weighted":
        return {d: nx.single_source_dijkstra_path_length(R, d, weight="w") for d in G.nodes()}
    return {d: nx.single_source_shortest_path_length(R, d) for d in G.nodes()}


def edge_cost(W, u, v, metric="hop"):
    """Cost of one arc under `metric` (1 hop, or the OSPF cost)."""
    return float(W[u][v]["w"]) if metric == "weighted" else 1.0


def shortest_path(G, W, s, d, metric="hop"):
    """OSPF path under the chosen metric. `W` is weighted_graph(G) (ignored for hop)."""
    if metric == "weighted":
        return nx.shortest_path(W, s, d, weight="w")
    return nx.shortest_path(G, s, d)


def all_shortest_paths(G, W, s, d, metric="hop"):
    """Equal-cost path set under the chosen metric — the candidates ECMP splits over."""
    if metric == "weighted":
        return list(nx.all_shortest_paths(W, s, d, weight="w"))
    return list(nx.all_shortest_paths(G, s, d))


def max_util(G, W, pairs, arc_index, cap, rates, metric="hop"):
    """Max link utilisation (%) if every demand follows its OSPF path under `metric`."""
    import numpy as np
    load = np.zeros(len(cap))
    for i, (s, d) in enumerate(pairs):
        if rates[i] <= 0:
            continue
        p = shortest_path(G, W, s, d, metric)
        for j in range(len(p) - 1):
            load[arc_index[(p[j], p[j + 1])]] += rates[i]
    return float((100.0 * load / cap).max())
