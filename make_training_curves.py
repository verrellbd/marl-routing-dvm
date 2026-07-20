#!/usr/bin/env python3
"""MARL training curves from the monaco retrain console logs (logs/RUN_*.log).
Top row: mean episode return (the training reward) vs environment steps.
Bottom row: held-out mean Δ max-util vs OSPF (pt, higher = better) vs steps.
Per network, seeds 0/1/2 as thin lines + mean as the bold line. No re-training —
parses the existing logs. NOTE: SA-GNN (SB3, verbose=0) logged no rewards, so this
figure covers the MARL runs only.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
SEEDS = [0, 1, 2]
NETS = [("abilene_sndlib", "RUN_abilene_sndlib_marl_real_seed{s}.log",
         "Abilene (12n)"),
        ("geant_sndlib", "RUN_geant_sndlib_marl_real_seed{s}.log",
         "GÉANT (22n)"),
        ("germany50_sndlib", "RUN_germany50_sndlib_marl_opt3b_seed{s}.log",
         "Germany50 (50n)")]
MA_C = "#4c72b0"  # MARL blue, matching fig_real3way_*
SEED_C = ["#9ab4d4", "#7d9fc7", "#6089ba"]

HDR = re.compile(r"(\d+)\s+steps")
UPD = re.compile(r"upd\s+(\d+)/(\d+)\s+ep_ret~(-?[\d.]+).*mean Δ ([+-][\d.]+)pt")


def parse(path):
    """-> (steps[], ep_ret[], delta[]) or None."""
    if not os.path.exists(path):
        return None
    total_steps = None
    upd, ret, dlt = [], [], []
    tot_upd = 1
    for ln in open(path, errors="replace"):
        if total_steps is None and "[train-marl]" in ln:
            m = HDR.search(ln)
            if m:
                total_steps = int(m.group(1))
        m = UPD.search(ln)
        if m:
            upd.append(int(m.group(1)))
            tot_upd = int(m.group(2))
            ret.append(float(m.group(3)))
            dlt.append(float(m.group(4)))
    if not upd:
        return None
    steps = np.array(upd) / tot_upd * (total_steps or tot_upd)
    return steps, np.array(ret), np.array(dlt)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex="col")
    for col, (net, pat, title) in enumerate(NETS):
        runs = [parse(f"logs/{pat.format(s=s)}") for s in SEEDS]
        runs = [r for r in runs if r is not None]
        if not runs:
            for row in (0, 1):
                axes[row][col].text(.5, .5, "no log", ha="center",
                                    transform=axes[row][col].transAxes)
            continue
        # align on the shortest run for the mean line
        n = min(len(r[0]) for r in runs)
        xs = runs[0][0][:n] / 1e3
        for row, idx, ylab in [(0, 1, "Mean episode return"),
                               (1, 2, "Held-out Δ vs OSPF (pt)")]:
            ax = axes[row][col]
            for si, r in enumerate(runs):
                ax.plot(r[0] / 1e3, r[idx], lw=0.9, alpha=0.55,
                        color=SEED_C[si], label=f"seed {si}" if col == 0 else None)
            mean = np.mean([r[idx][:n] for r in runs], axis=0)
            ax.plot(xs, mean, lw=2.2, color=MA_C,
                    label="mean (3 seeds)" if col == 0 else None)
            if row == 1:
                ax.axhline(0, color="#c44e52", lw=1, ls="--")
                if col == 0:
                    ax.text(xs[-1], 0, " OSPF parity", color="#c44e52",
                            fontsize=8, va="bottom", ha="right")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylab)
        axes[0][col].set_title(title, fontweight="bold")
        axes[1][col].set_xlabel("Environment steps (×10³)")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle("MARL (MAPPO) training curves — reward and held-out gain vs OSPF\n"
                 "seeds 0/1/2, logged every 5 PPO updates",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = f"{R}/fig_marl_training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
