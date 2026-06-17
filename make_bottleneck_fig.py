#!/usr/bin/env python3
"""Illustration of the GÉANT capacity-limited bottleneck (matrix 1005):
3,893 Mbps of demand for leaf node 22 must funnel through its single 2.5 Gbps
uplink (18->22) -> 156% utilisation that NO routing can relieve."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

# ---- backbone cloud (rest of GÉANT) ----
cloud = FancyBboxPatch((0.3, 2.2), 2.7, 3.6, boxstyle="round,pad=0.1",
                       fc="#eef1f6", ec="#9aa6bd", lw=1.5)
ax.add_patch(cloud)
ax.text(1.65, 5.4, "Rest of GÉANT\nbackbone", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#3a4a6b")
ax.text(1.65, 3.0, "22 flows totalling\n3,893 Mbps are\ndestined for node 22",
        ha="center", va="center", fontsize=8.5, color="#3a4a6b")

# ---- node 18 (hub) ----
hub = Circle((6.0, 4.0), 0.55, fc="#4c72b0", ec="black", lw=1.5, zorder=5)
ax.add_patch(hub)
ax.text(6.0, 4.0, "18", ha="center", va="center", color="white",
        fontweight="bold", fontsize=13, zorder=6)
ax.text(6.0, 3.15, "hub", ha="center", va="center", fontsize=8, color="#444")

# ---- three FAT 10G links feeding the hub from the backbone ----
for y in (5.2, 4.0, 2.8):
    a = FancyArrowPatch((3.1, y), (5.55, 4.0), arrowstyle="-|>",
                        mutation_scale=18, lw=6, color="#9ec4e8",
                        connectionstyle="arc3,rad=0.0", zorder=2)
    ax.add_patch(a)
ax.text(4.3, 5.55, "10 Gbps links (plenty of room)", fontsize=8.5,
        color="#3070b0", ha="center", style="italic")

# ---- node 22 (leaf) ----
leaf = Circle((10.6, 4.0), 0.55, fc="#c44e52", ec="black", lw=1.5, zorder=5)
ax.add_patch(leaf)
ax.text(10.6, 4.0, "22", ha="center", va="center", color="white",
        fontweight="bold", fontsize=13, zorder=6)
ax.text(10.6, 3.15, "leaf node\n(one wire only)", ha="center", va="center",
        fontsize=8, color="#444")

# ---- the single THIN 2.5G bottleneck link 18 -> 22 ----
ax.add_patch(FancyArrowPatch((6.6, 4.0), (10.0, 4.0), arrowstyle="-|>",
             mutation_scale=18, lw=4, color="#c44e52", zorder=4))
ax.text(8.3, 4.65, "ONLY link in/out of node 22", ha="center",
        fontsize=9.5, fontweight="bold", color="#c44e52")
ax.text(8.3, 4.32, "18 → 22  capacity = 2,500 Mbps (2.5 Gbps)", ha="center",
        fontsize=9, color="#c44e52")

# overflow callout
ax.annotate("3,893 Mbps wants through\na 2,500 Mbps pipe\n→ 156% utilisation\n→ ~1,393 Mbps dropped",
            xy=(8.3, 4.0), xytext=(8.3, 1.5), ha="center", fontsize=9.5,
            color="#7a1f24", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdeaea", ec="#c44e52"),
            arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.5))

# ---- title + takeaway ----
ax.text(6.0, 7.5, "Why GÉANT is capacity-limited — node 22 (matrix 1005)",
        ha="center", fontsize=14, fontweight="bold")
ax.text(6.0, 6.9,
        "Traffic can arrive at hub 18 freely over fat 10G links, but every packet for "
        "node 22 must cross its single 2.5G uplink.",
        ha="center", fontsize=9.5, color="#333")
ax.text(6.0, 0.55,
        "No routing can help: there is no second path into node 22. "
        "Greedy with up to k=8 paths is also stuck at 156%. Only a faster link (hardware) fixes it.",
        ha="center", fontsize=9.5, style="italic", color="#7a1f24")

fig.tight_layout()
fig.savefig("results/fig_geant_bottleneck.png", dpi=150)
print("[saved] results/fig_geant_bottleneck.png")
