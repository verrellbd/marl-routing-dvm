"""Gravity model traffic matrix generator.

T_ij ∝ (w_i^out × w_j^in) / sum(w_k)

Weights come from a log-normal distribution by default (realistic hot spots),
or uniform (sanity baseline). The matrix is scaled by a load factor so you can
sweep congestion levels (α=0.3 light, 1.0 moderate, 1.5 heavy).

Output is a list of ns-3 OnOff flows (src, dst, rate_mbps, start, stop).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from marl_routing.topology import load


@dataclass(frozen=True)
class Flow:
    src: int
    dst: int
    rate_mbps: float
    start: float
    stop: float


def generate_matrix(
    topo_name: str,
    load_factor: float = 1.0,
    weight_dist: str = "lognorm",
    seed: int = 42,
) -> np.ndarray:
    """Generate gravity-model demand matrix.

    Args:
        topo_name: name of topology (e.g. "abilene")
        load_factor: scale matrix by this (1.0 = baseline)
        weight_dist: "lognorm" (hot spots) or "uniform" (flat)
        seed: for reproducibility

    Returns:
        (n_nodes, n_nodes) demand matrix in Mbps
    """
    topo = load(topo_name)
    n = topo.n_nodes
    rng = np.random.RandomState(seed)

    # --- Generate node weights ---
    if weight_dist == "lognorm":
        w = rng.lognormal(mean=0, sigma=1.0, size=n)
        w /= w.mean()
    elif weight_dist == "uniform":
        w = np.ones(n)
    else:
        raise ValueError(f"unknown weight_dist: {weight_dist}")

    # --- Gravity model: T_ij = (w_i^out × w_j^in) / sum(w) ---
    w_out = w.reshape(-1, 1)
    w_in = w.reshape(1, -1)
    T = (w_out * w_in) / w.sum()

    # Zero out diagonal (no self-traffic)
    np.fill_diagonal(T, 0)

    # Normalize to baseline: 10% of total link capacity across all edges
    # (this is the baseline at load_factor=1.0)
    topo_g = topo.graph
    mean_cap = np.mean([d["capacity"] for _, _, d in topo_g.edges(data=True)])
    target_total = n * mean_cap * 0.1
    T *= target_total / T.sum()

    # Now scale by load factor (0.3 = 30% of target, 1.5 = 150%)
    T *= load_factor

    return T


def matrix_to_flows(
    matrix: np.ndarray,
    topo_name: str,
    sim_time: float = 60.0,
    ramp_time: float = 2.0,
    min_flow_mbps: float = 0.0,
) -> list[Flow]:
    """Convert demand matrix to flow list for ns-3.

    Each non-zero entry becomes an OnOff flow. Flows start at ramp_time
    and run until sim_time - ramp_time (to avoid startup/shutdown transients).

    Args:
        matrix: (n, n) demand matrix in Mbps
        topo_name: topology name
        sim_time: total simulation time in seconds
        ramp_time: startup/ramp-down buffer
        min_flow_mbps: skip flows below this threshold (for tractable ns-3 sims)

    Returns:
        list of Flow objects
    """
    n = matrix.shape[0]
    flows = []

    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] >= min_flow_mbps:
                flows.append(
                    Flow(
                        src=i,
                        dst=j,
                        rate_mbps=matrix[i, j],
                        start=ramp_time,
                        stop=sim_time - ramp_time,
                    )
                )

    return sorted(flows, key=lambda f: (f.src, f.dst))


def save_flows_json(flows: list[Flow], out: Path) -> Path:
    """Write flows to JSON for ns-3 consumption."""
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "flows": [asdict(f) for f in flows],
        "summary": {
            "n_flows": len(flows),
            "total_demand_mbps": sum(f.rate_mbps for f in flows),
        },
    }
    with out.open("w") as f:
        json.dump(data, f, indent=2)
    return out


def summary(matrix: np.ndarray) -> str:
    """Print matrix statistics."""
    mask = matrix > 0
    nz = mask.sum()
    return (
        f"Demand matrix: {matrix.shape[0]}x{matrix.shape[0]}\n"
        f"  non-zero flows: {nz}\n"
        f"  total demand: {matrix.sum():.1f} Mbps\n"
        f"  per-flow min/mean/max: "
        f"{matrix[mask].min():.1f} / {matrix[mask].mean():.1f} / {matrix[mask].max():.1f}"
    )


if __name__ == "__main__":
    import sys

    topo = sys.argv[1] if len(sys.argv) > 1 else "abilene"
    load_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    min_flow = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    print(f"Generating gravity model for {topo} with load={load_factor}, min_flow={min_flow} Mbps")
    mat = generate_matrix(topo, load_factor=load_factor, weight_dist="lognorm")
    print(summary(mat))

    flows = matrix_to_flows(mat, topo, sim_time=60.0, min_flow_mbps=min_flow)
    suffix = f"_α{load_factor}" + (f"_min{int(min_flow)}" if min_flow > 0 else "")
    out = Path(__file__).resolve().parent.parent / "results" / f"traffic_{topo}{suffix}.json"
    save_flows_json(flows, out)
    print(f"\nWrote {len(flows)} flows to {out}")
