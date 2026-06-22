# Checkpoint — 2026-06-22

## TL;DR
Switched the ENTIRE study from synthetic to **real, cited data** (user-provided SNDlib
files only). Real topologies + real capacities + real **dynamic** measured traffic for
all three networks, with temporal train/test splits. Both methods (single-agent GNN and
MARL) **beat OSPF on all three topologies on real unseen traffic**. Two findings flipped
vs the synthetic results — see below.

## What we did today
1. **Converted SNDlib native files** (`sndlib_to_json.py`): topologies/{abilene,geant,
   germany50}.sndlib → <name>_sndlib.json (+ <name>_sndlib_demands.json). Capacity =
   pre-installed else module; delay = haversine(file coords) × 0.005 ms/km (fiber speed);
   germany50 ×1000 so all are Mbps. REAL specs: abilene 12n/15L (14×9920 + 1×2480),
   geant 22n/36L (40G), germany50 50n/88L (40G).
2. **Real dynamic traffic loader** (`marl_routing/real_traffic.py`): reads the three
   measured demand-matrix archives (Abilene-Zhang 48096×5min/6mo; GÉANT-Uhlig 11460×
   15min/4mo; Germany50-DFN 288×5min/1day) straight from the .tgz, with a TEMPORAL split
   (train=first 70% of timeline, test=last 30% → generalization to unseen real traffic).
   Skips empty snapshots. Raw demands are dimensioning-scale ("3Tbps"), so magnitude is
   scaled by a load factor (real STRUCTURE kept). Per-topo congesting scales: abilene 8/12/16,
   geant 3/5/7, germany50 35/50/65.
3. **`--traffic real`** wired into train_gnn_qos.py + train_marl.py.
4. Trained all 6 models (3 topos × {GNN, MARL}) on real dynamic traffic.

## Real-data results (max link-util %, REAL unseen held-out traffic)
| topo | OSPF (ex.) | greedy | single-agent GNN | MARL |
|------|-----------|--------|------------------|------|
| abilene | 127.8 | 129.0 | 71.6 | 78.4 |
| geant | 188.4 | 140.5 | 132.1 | 137.1 |
| germany50 | 177.0 | 135.8 | 139.0 | 148.2 |
Both methods beat OSPF on all three (MARL 5/5 on each); single-agent usually a touch
better than MARL (centralized edge). Models: results/{abilene,geant,germany50}_sndlib_
{qos,marl}_real_seed0/.

## Two findings that FLIPPED vs synthetic data (important, honest)
1. **GÉANT is NOT capacity-limited** with real data. The old "GÉANT ties OSPF" result was
   an artifact of my *reconstructed* mixed-2.5G/10G topology. Real GÉANT (uniform 40G,
   22 nodes) is winnable — both methods cut overload substantially.
2. **Germany50 MARL miscoordination disappeared.** On synthetic gravity it had
   catastrophic outliers (96→141%); on real measured traffic MARL wins cleanly 5/5. Real
   traffic structure is less adversarial than synthetic log-normal hot-spots.

## Data provenance (for the methodology write-up)
- Topologies/capacities: REAL, from user's SNDlib files. (Abilene caps real OC-192/48;
  GÉANT/Germany50 from installable 40G modules.)
- Traffic: REAL measured dynamic matrices (SNDlib/TOTEM/DFN). Only the overall MAGNITUDE
  is scaled (standard TE practice) — spatial+temporal structure is untouched.
- Delays: computed from the file's own coordinates (haversine × fiber speed). The one
  value SNDlib doesn't provide.

## Remaining (next session)
1. **ns-3 packet-level validation on real data**: wire `--traffic real` into
   evaluate_ns3.py + export_marl_routing.py (use real TEST matrices), run the 3-way
   (loss/delay) on all three real topologies + figures. germany50 needs --max-flows.
2. Multi-seed (1,2) for error bars on real data.
3. Update RESULTS_SUMMARY.md to the real-data numbers (current §2b/§3 still cite the
   synthetic/reconstructed runs).

## Tools added (reusable)
sndlib_to_json.py · marl_routing/real_traffic.py · --traffic real on both trainers.
