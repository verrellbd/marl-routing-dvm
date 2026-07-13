#!/usr/bin/env bash
# Parallel training launcher — trains the full model set (3 networks x 2 methods x 3 seeds
# = 18 models) concurrently on a many-core CPU box (e.g. monaco: 40 cores / 512 GB).
#
# Each job is capped to a few CPU threads so many run at once without fighting.
# Usage:
#   ./train_all_monaco.sh            # run everything
#   MAXJOBS=10 ./train_all_monaco.sh # cap concurrency (e.g. on a 20-core box)
#   DRYRUN=1 ./train_all_monaco.sh   # print the commands without running
#
# Reproduces the settled configs: abilene/geant MARL uncapped (lambda 0.5); germany50
# MARL hop-capped (max-stretch 4, lambda 0.1, tag _opt3b); GNN lambda per-network.

set -u
cd "$(dirname "$0")"

# --- CPU thread caps so parallel jobs don't oversubscribe ---
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
THREADS=2                      # torch threads per job (GNN --threads)
MAXJOBS="${MAXJOBS:-18}"       # max concurrent jobs (18 fits 40 cores at 2 threads each)
SEEDS="${SEEDS:-0 1 2}"
DRYRUN="${DRYRUN:-0}"

mkdir -p logs

# activate venv if not already active (NFS-shared home -> same venv on monaco)
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ns3ai-venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source ns3ai-venv/bin/activate
fi

# config: "network:loads:delay_penalty:extra_marl_args"
CONFIGS=(
  "abilene_sndlib:8,12,16:0.5:"
  "geant_sndlib:3,5,7:0.5:"
  "germany50_sndlib:35,50,65:0.1:--max-stretch 4"
)

launch() {   # launch() "<command...>" "<logfile>"
  local cmd="$1" log="$2"
  if [ "$DRYRUN" = "1" ]; then echo "[dry] $cmd  > $log"; return; fi
  # throttle: wait until fewer than MAXJOBS background jobs are running
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  echo "[launch] $log"
  bash -c "$cmd" > "$log" 2>&1 &
}

echo "=== training $(( ${#CONFIGS[@]} * 2 * $(echo $SEEDS | wc -w) )) models, up to $MAXJOBS at once, $THREADS threads each ==="
for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r topo loads dp extra <<< "$cfg"
  for s in $SEEDS; do
    # --- MARL (decentralized). germany50 uses hop-cap + tag _opt3b; others _real ---
    if [ -n "$extra" ]; then mtag="_opt3b"; else mtag="_real"; fi
    launch "python train_marl.py --topo $topo --traffic real --loads $loads \
            --seed $s --delay-penalty $dp $extra --tag $mtag" \
           "logs/RUN_${topo}_marl${mtag}_seed${s}.log"
    # --- GNN (centralized single-agent) ---
    launch "python train_gnn_qos.py --topo $topo --traffic real --loads $loads \
            --seed $s --delay-penalty $dp --threads $THREADS --tag _real" \
           "logs/RUN_${topo}_qos_real_seed${s}.log"
  done
done

echo "=== all jobs launched; waiting for completion ==="
wait
echo "=== DONE. models in results/<net>_{marl,qos}_*_seed*/; logs in logs/RUN_*.log ==="
