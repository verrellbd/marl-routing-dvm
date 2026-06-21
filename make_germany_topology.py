#!/usr/bin/env python3
"""
Generate topologies/germany50.json — the German research backbone (Germany50).

PROVENANCE (be honest in the write-up):
- NODES: the real Germany50 city set (50 German cities), with accurate lat/lon.
- LINKS: a faithful reconstruction of the German backbone adjacency (geographically
  derived, approximating SNDlib germany50). Swap in the exact SNDlib edge list if you
  have the XML — only this block changes.
- CAPACITIES: assigned by a hub/edge/peripheral hierarchy (not from SNDlib modules), so
  the topology can exhibit BOTH a well-connected-win regime (like Abilene) and
  capacity-limited bottlenecks (like GÉANT). Documented, not claimed as ground truth.
- DELAYS: computed from great-circle distance (fiber ~5 µs/km), realistic.
"""
import json
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

CITIES = [  # id: (name, lat, lon)
    ("Aachen", 50.7753, 6.0839), ("Augsburg", 48.3705, 10.8978),
    ("Bayreuth", 49.9456, 11.5713), ("Berlin", 52.5200, 13.4050),
    ("Bielefeld", 52.0302, 8.5325), ("Braunschweig", 52.2689, 10.5268),
    ("Bremen", 53.0793, 8.8017), ("Bremerhaven", 53.5396, 8.5809),
    ("Chemnitz", 50.8278, 12.9214), ("Darmstadt", 49.8728, 8.6512),
    ("Dortmund", 51.5136, 7.4653), ("Dresden", 51.0504, 13.7373),
    ("Duesseldorf", 51.2277, 6.7735), ("Erfurt", 50.9848, 11.0299),
    ("Essen", 51.4556, 7.0116), ("Flensburg", 54.7937, 9.4460),
    ("Frankfurt", 50.1109, 8.6821), ("Freiburg", 47.9990, 7.8421),
    ("Fulda", 50.5558, 9.6808), ("Giessen", 50.5841, 8.6784),
    ("Greifswald", 54.0865, 13.3923), ("Hamburg", 53.5511, 9.9937),
    ("Hannover", 52.3759, 9.7320), ("Karlsruhe", 49.0069, 8.4037),
    ("Kassel", 51.3127, 9.4797), ("Kempten", 47.7267, 10.3168),
    ("Kiel", 54.3233, 10.1228), ("Koblenz", 50.3569, 7.5890),
    ("Koeln", 50.9375, 6.9603), ("Konstanz", 47.6603, 9.1758),
    ("Leipzig", 51.3397, 12.3731), ("Magdeburg", 52.1205, 11.6276),
    ("Mannheim", 49.4875, 8.4660), ("Muenchen", 48.1351, 11.5820),
    ("Muenster", 51.9607, 7.6261), ("Norden", 53.5972, 7.2061),
    ("Nuernberg", 49.4521, 11.0767), ("Oldenburg", 53.1435, 8.2146),
    ("Osnabrueck", 52.2799, 8.0472), ("Passau", 48.5667, 13.4319),
    ("Regensburg", 49.0134, 12.1016), ("Saarbruecken", 49.2402, 6.9969),
    ("Schwerin", 53.6355, 11.4012), ("Siegen", 50.8748, 8.0243),
    ("Stuttgart", 48.7758, 9.1829), ("Trier", 49.7596, 6.6441),
    ("Ulm", 48.4011, 9.9876), ("Wesel", 51.6586, 6.6176),
    ("Wilhelmshaven", 53.5293, 8.1075), ("Wuerzburg", 49.7913, 9.9534),
]
NAME = {n: i for i, (n, _, _) in enumerate(CITIES)}

# Backbone adjacency (undirected), by city name — geographic German backbone.
EDGES = [
    ("Flensburg", "Kiel"), ("Kiel", "Hamburg"), ("Kiel", "Schwerin"),
    ("Hamburg", "Bremen"), ("Hamburg", "Schwerin"), ("Hamburg", "Hannover"),
    ("Schwerin", "Berlin"), ("Schwerin", "Greifswald"), ("Greifswald", "Berlin"),
    ("Bremerhaven", "Bremen"), ("Bremerhaven", "Wilhelmshaven"),
    ("Wilhelmshaven", "Oldenburg"), ("Oldenburg", "Bremen"), ("Oldenburg", "Norden"),
    ("Norden", "Osnabrueck"), ("Bremen", "Hannover"), ("Bremen", "Osnabrueck"),
    ("Osnabrueck", "Muenster"), ("Osnabrueck", "Bielefeld"),
    ("Muenster", "Dortmund"), ("Muenster", "Wesel"), ("Bielefeld", "Hannover"),
    ("Bielefeld", "Dortmund"), ("Hannover", "Braunschweig"), ("Hannover", "Kassel"),
    ("Hannover", "Magdeburg"), ("Braunschweig", "Magdeburg"), ("Magdeburg", "Berlin"),
    ("Magdeburg", "Leipzig"), ("Berlin", "Leipzig"), ("Berlin", "Dresden"),
    ("Leipzig", "Dresden"), ("Leipzig", "Chemnitz"), ("Leipzig", "Erfurt"),
    ("Dresden", "Chemnitz"), ("Chemnitz", "Bayreuth"), ("Erfurt", "Kassel"),
    ("Erfurt", "Fulda"), ("Erfurt", "Nuernberg"), ("Kassel", "Fulda"),
    ("Kassel", "Giessen"), ("Kassel", "Bielefeld"), ("Wesel", "Essen"),
    ("Essen", "Dortmund"), ("Essen", "Duesseldorf"), ("Dortmund", "Siegen"),
    ("Duesseldorf", "Koeln"), ("Duesseldorf", "Aachen"), ("Koeln", "Aachen"),
    ("Koeln", "Koblenz"), ("Koeln", "Siegen"), ("Siegen", "Giessen"),
    ("Siegen", "Frankfurt"), ("Koblenz", "Frankfurt"), ("Koblenz", "Trier"),
    ("Trier", "Saarbruecken"), ("Giessen", "Frankfurt"), ("Giessen", "Fulda"),
    ("Frankfurt", "Darmstadt"), ("Frankfurt", "Fulda"), ("Frankfurt", "Wuerzburg"),
    ("Darmstadt", "Mannheim"), ("Mannheim", "Karlsruhe"), ("Mannheim", "Saarbruecken"),
    ("Karlsruhe", "Stuttgart"), ("Karlsruhe", "Freiburg"), ("Stuttgart", "Ulm"),
    ("Stuttgart", "Wuerzburg"), ("Wuerzburg", "Nuernberg"), ("Wuerzburg", "Fulda"),
    ("Nuernberg", "Bayreuth"), ("Nuernberg", "Regensburg"), ("Nuernberg", "Augsburg"),
    ("Regensburg", "Passau"), ("Regensburg", "Muenchen"), ("Muenchen", "Augsburg"),
    ("Muenchen", "Passau"), ("Muenchen", "Kempten"), ("Muenchen", "Ulm"),
    ("Augsburg", "Ulm"), ("Ulm", "Kempten"), ("Kempten", "Konstanz"),
    ("Freiburg", "Konstanz"), ("Bayreuth", "Regensburg"),
]

# capacity hierarchy (Mbps): big hubs get 40G cores, peripheral/leaf links 2.5G
HUBS = {"Frankfurt", "Hannover", "Koeln", "Berlin", "Leipzig", "Nuernberg",
        "Muenchen", "Stuttgart", "Mannheim", "Dortmund"}


def haversine(a, b):
    (_, la1, lo1), (_, la2, lo2) = CITIES[a], CITIES[b]
    la1, lo1, la2, lo2 = map(radians, [la1, lo1, la2, lo2])
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def degree_map(edges):
    deg = {i: 0 for i in range(len(CITIES))}
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    return deg


def main():
    eidx = sorted({tuple(sorted((NAME[u], NAME[v]))) for u, v in EDGES})
    deg = degree_map(eidx)
    links = []
    for u, v in eidx:
        un, vn = CITIES[u][0], CITIES[v][0]
        if un in HUBS and vn in HUBS:
            cap = 40000
        elif deg[u] <= 2 or deg[v] <= 2:      # peripheral / leaf link
            cap = 2500
        else:
            cap = 10000
        dist = haversine(u, v)
        delay = round(max(0.5, dist * 0.005), 2)   # fiber ~5 µs/km
        links.append({"src": u, "dst": v, "capacity": cap, "delay": delay})

    topo = {
        "name": "Germany50",
        "description": ("German research backbone (Germany50 city set). Nodes = real "
                        "50 German cities w/ accurate lat/lon. Links = geographically "
                        "reconstructed German backbone (~SNDlib germany50 structure). "
                        "Capacities = hub(40G)/edge(10G)/peripheral(2.5G) hierarchy "
                        "(modelled, not SNDlib modules). Delays from great-circle distance."),
        "capacity_unit": "Mbps", "delay_unit": "ms",
        "nodes": [{"id": i, "name": n, "city": n, "lat": la, "lon": lo}
                  for i, (n, la, lo) in enumerate(CITIES)],
        "links": links,
    }
    out = Path("topologies/germany50.json")
    out.write_text(json.dumps(topo, indent=2))
    caps = {}
    for l in links:
        caps[l["capacity"]] = caps.get(l["capacity"], 0) + 1
    print(f"[saved] {out}: {len(CITIES)} nodes, {len(links)} links")
    print(f"  capacity mix (Mbps): {dict(sorted(caps.items()))}")
    leaves = [CITIES[i][0] for i, d in deg.items() if d == 1]
    print(f"  degree-1 leaves: {leaves}")


if __name__ == "__main__":
    main()
