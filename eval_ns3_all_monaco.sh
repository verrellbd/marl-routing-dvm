#!/usr/bin/env bash
# Re-run the ns-3 packet-level QoS evaluation on the FRESH (monaco-retrained) models.
# For each (network, seed): export SA-GNN paths and MARL paths, then run ns-3 (OSPF +
# method) over the 5 real test matrices, writing summary.json (loss / delay / throughput).
#
# Usage:
#   ./eval_ns3_all_monaco.sh --preflight   # test ONE ns-3 run first (do this once!)
#   ./eval_ns3_all_monaco.sh               # seed 0 only (consistency refresh, 6 dirs)
#   SEEDS="0 1 2" ./eval_ns3_all_monaco.sh # full multi-seed QoS (18 dirs, slow)
#   DRYRUN=1 ./eval_ns3_all_monaco.sh      # print commands only
#
# IMPORTANT: ns-3 is a compiled binary. If it was built on malmo with CPU-specific flags
# it may fail on monaco's older cores ("illegal instruction"). Run --preflight FIRST; if it
# crashes, rebuild ns-3 on monaco (cd ns-3-dev && ~/thesis/configure_ns3.sh && ./ns3 build -j8).

set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
SEEDS="${SEEDS:-0}"
MAXJOBS="${MAXJOBS:-10}"        # concurrent ns-3 runs (each ~1 core; 40-core box -> 10 safe)
DRYRUN="${DRYRUN:-0}"
PREFLIGHT=0; [ "${1:-}" = "--preflight" ] && PREFLIGHT=1

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ns3ai-venv/bin/activate ]; then source ns3ai-venv/bin/activate; fi

# network -> "load:max_flows:marl_tag:marl_extra_args"
declare -A NET=(
  ["abilene_sndlib"]="12:0:_real:"
  ["geant_sndlib"]="5:0:_real:"
  ["germany50_sndlib"]="35:200:_opt3b:--max-stretch 4"
)

run(){ if [ "$DRYRUN" = "1" ]; then echo "[dry] $*"; else echo "[run] $*"; eval "$@"; fi; }

export_paths(){   # export_paths <net> <seed>
  local net="$1" s="$2"; IFS=':' read -r load mf mtag mextra <<< "${NET[$net]}"
  run "python evaluate_ns3.py --topo $net --traffic real --load $load \
       --model results/${net}_qos_real_seed${s}/gnn_generalist_qos \
       --n-overload 3 --n-feasible 2 --max-flows $mf --export-only \
       --tag _realsa_fresh_${net}_s${s}"
  run "python export_marl_routing.py --topo $net --traffic real --load $load \
       --model results/${net}_marl${mtag}_seed${s}/mappo_actor_critic.pt $mextra \
       --n-overload 3 --n-feasible 2 --max-flows $mf \
       --tag _realmarl_fresh_${net}_s${s}"
}

ns3_run(){        # ns3_run <dir> <net>
  local dir="$1" net="$2"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  if [ "$DRYRUN" = "1" ]; then echo "[dry] ns3 $dir"; return; fi
  ( python run_ns3_phase2.py --dir "$(pwd)/results/$dir" --topo "$net" --ratescale 20 \
      > "logs/NS3_${dir}.log" 2>&1 ) &
}

# ---- preflight: one export + one ns-3 run, then stop ----
if [ "$PREFLIGHT" = "1" ]; then
  echo "=== PREFLIGHT: abilene seed 0 SA-GNN, one ns-3 eval ==="
  export_paths abilene_sndlib 0
  python run_ns3_phase2.py --dir "$(pwd)/results/ns3_eval_realsa_fresh_abilene_sndlib_s0" \
     --topo abilene_sndlib --ratescale 20 || { echo "!!! ns-3 FAILED — rebuild on monaco"; exit 1; }
  echo "=== PREFLIGHT OK — ns-3 runs on monaco. Now run without --preflight. ==="
  exit 0
fi

# ---- phase 1: all exports (fast) ----
echo "=== phase 1: exporting paths (seeds: $SEEDS) ==="
for net in "${!NET[@]}"; do for s in $SEEDS; do export_paths "$net" "$s"; done; done

# ---- phase 2: all ns-3 runs in parallel ----
echo "=== phase 2: ns-3 runs (up to $MAXJOBS at once) ==="
for net in "${!NET[@]}"; do for s in $SEEDS; do
  ns3_run "ns3_eval_realsa_fresh_${net}_s${s}"   "$net"
  ns3_run "ns3_eval_realmarl_fresh_${net}_s${s}" "$net"
done; done
wait
echo "=== DONE. summaries: results/ns3_eval_*_fresh_*/summary.json ; logs: logs/NS3_*.log ==="
