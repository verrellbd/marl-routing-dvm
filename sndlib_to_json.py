#!/usr/bin/env python3
"""
Convert an SNDlib native-format file (topologies/<name>.sndlib) into our topology
JSON schema + a REAL demand matrix — replacing the reconstructed topologies and the
synthetic gravity traffic with cited, ground-truth data.

SNDlib native format:
  NODES:   <name> ( <lon> <lat> )
  LINKS:   <id> ( <src> <dst> ) <preinst_cap> <preinst_cost> <route_cost> <setup_cost> ( {<mod_cap> <mod_cost>}* )
  DEMANDS: <id> ( <src> <dst> ) <routing_unit> <value> <max_path_len>

Capacity per link = pre-installed capacity if > 0, else the (largest) installable
module capacity. Delay is computed from great-circle distance (SNDlib has none).

Writes:
  topologies/<name>_sndlib.json        — topology (nodes + links, our schema)
  topologies/<name>_sndlib_demands.json — real directed demand matrix (Mbps)

Usage: python sndlib_to_json.py abilene geant germany50
"""
import json
import re
import sys
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

T = Path("topologies")


def section(text, name):
    """Return the lines inside `NAME ( ... )`."""
    m = re.search(rf"{name}\s*\((.*?)\n\)", text, re.S)
    return m.group(1).strip().splitlines() if m else []


def haversine(lo1, la1, lo2, la2):
    lo1, la1, lo2, la2 = map(radians, [lo1, la1, lo2, la2])
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def convert(name, cap_scale=1.0):
    text = (T / f"{name}.sndlib").read_text()
    unit = (re.search(r"unit\s*=\s*(\S+)", text) or [None, "UNSPECIFIED"])[1]

    # --- nodes ---
    nodes, nid = [], {}
    for ln in section(text, "NODES"):
        m = re.match(r"\s*(\S+)\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", ln)
        if not m:
            continue
        nm, lon, lat = m.group(1), float(m.group(2)), float(m.group(3))
        nid[nm] = len(nodes)
        nodes.append({"id": len(nodes), "name": nm, "city": nm,
                      "lat": lat, "lon": lon})

    # --- links: capacity = preinstalled if >0 else max module capacity ---
    links = []
    for ln in section(text, "LINKS"):
        m = re.match(r"\s*\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*\((.*)\)", ln)
        if not m:
            continue
        s, d, preinst = m.group(1), m.group(2), float(m.group(3))
        mods = [float(x) for x in m.group(4).split()]
        mod_caps = mods[0::2] if mods else [0.0]
        cap = preinst if preinst > 0 else (max(mod_caps) if mod_caps else 0.0)
        cap *= cap_scale
        delay = round(max(0.1, haversine(nodes[nid[s]]["lon"], nodes[nid[s]]["lat"],
                                         nodes[nid[d]]["lon"], nodes[nid[d]]["lat"]) * 0.005), 2)
        links.append({"src": nid[s], "dst": nid[d], "capacity": cap, "delay": delay})

    topo = {
        "name": f"{name} (SNDlib)",
        "description": (f"SNDlib native instance '{name}' (sndlib.zib.de) — REAL data. "
                       f"Capacity = pre-installed else installable module; unit {unit}"
                       f"{'' if cap_scale==1 else f' x{cap_scale}'}. Delays from "
                       f"great-circle distance. Nodes/links/demands are ground-truth."),
        "capacity_unit": "Mbps", "delay_unit": "ms", "nodes": nodes, "links": links,
    }
    (T / f"{name}_sndlib.json").write_text(json.dumps(topo, indent=2))

    # --- real demand matrix (directed) ---
    n = len(nodes)
    D = [[0.0] * n for _ in range(n)]
    ndem = 0
    for ln in section(text, "DEMANDS"):
        m = re.match(r"\s*\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\S+\s+([\d.]+)", ln)
        if not m:
            continue
        D[nid[m.group(1)]][nid[m.group(2)]] = float(m.group(3)) * cap_scale
        ndem += 1
    (T / f"{name}_sndlib_demands.json").write_text(json.dumps(
        {"topo": f"{name}_sndlib", "unit": "Mbps", "n_demands": ndem, "matrix": D}, indent=2))

    caps = {}
    for l in links:
        caps[l["capacity"]] = caps.get(l["capacity"], 0) + 1
    tot = sum(sum(r) for r in D)
    print(f"[{name}] {n} nodes, {len(links)} links, {ndem} demands | "
          f"caps {dict(sorted(caps.items()))} | total demand {tot:.0f} ({unit})")
    return name


if __name__ == "__main__":
    names = sys.argv[1:] or ["abilene", "geant", "germany50"]
    # germany50 capacities/demands are in unitless modules (~Gbit/s); x1000 -> Mbps so
    # absolute rates are sane for ns-3 (utilisation is a ratio, unaffected).
    scales = {"germany50": 1000.0}
    for nm in names:
        convert(nm, cap_scale=scales.get(nm, 1.0))
