#!/usr/bin/env bash
# Extend the reported arms from 3 seeds to 10 (adds seeds 3..9).
#
# WHY: the feasible-regime packet loss reported at 3 seeds is a mean over a bimodal
# set -- on Abilene, seeds 0/1 match OSPF (0.16/0.34%) while seed 2 fails (2.42%).
# Three points cannot distinguish "one unlucky run" from "this happens a third of the
# time", which is exactly the caveat the Limitations paragraph currently concedes.
#
# SETTINGS ARE COPIED, NOT RECHOSEN. Every flag below is taken verbatim from the runs
# that produced seeds 0-2, verified against logs/train_marlh32cm_s0.log and
# logs/train_singleRM_s0.log. The seed is the
# only thing that differs, otherwise the ten runs are not one population.
#
# ARMS. h=64 is dropped: it is not reported anywhere in the paper. The single agent
# uses --reward-form marl (tag _singleH64gRM), which is the arm in final_ns3_grid.json
# -- NOT _singleH64gcap, which is the superseded whole-reward ablation.
#
# COMMIT TO REPORTING ALL TEN before looking at them. Running extra seeds and keeping
# the flattering subset would invalidate the very variance claim this is meant to fix.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_DIR="${NS3_DIR:-$REPO/ns-3-dev}"
cd "$REPO" || exit 1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SEEDS=${SEEDS:-"3 4 5 6 7 8 9"}
JOBS=${JOBS:-14}
mkdir -p logs
JOBFILE=logs/train_seeds10_jobs.txt
: > $JOBFILE

for s in $SEEDS; do
  # single agent -- SB3 defaults untouched (it is the baseline), marl reward form
  [ -f "results/single_singleH64gRM_seed$s/policy.zip" ] || \
  echo "nice -n 19 python train_single_tier2.py --seed $s --timesteps 1500000 --traffic tmgen --k-paths 3 --hidden 64 --rounds 3 --n-envs 8 --n-steps 256 --ent-coef 0.01 --metric weighted --reward-form marl --tag _singleH64gRM > logs/train_singleRM_s$s.log 2>&1 && echo OK single_s$s || echo FAIL single_s$s" >> $JOBFILE

  # MARL h=32 -- moved onto the baseline's PPO settings
  [ -f "results/marlgnn_tier2m15cm_seed$s/policy.pt" ] || \
  echo "nice -n 19 python train_marl_gnn_tier2.py --seed $s --updates 732 --rollout 2048 --gamma 0.995 --n-epochs 10 --minibatch 256 --hidden 32 --rounds 3 --traffic tmgen --metric weighted --tag _tier2m15cm > logs/train_marlh32cm_s$s.log 2>&1 && echo OK marlh32_s$s || echo FAIL marlh32_s$s" >> $JOBFILE
done

echo "[seeds10] $(wc -l < $JOBFILE) runs to do, JOBS=$JOBS"
# the single agent is the long pole (~3h vs ~1.3h); start those first so they overlap
grep single $JOBFILE  > logs/.seeds10_ordered
grep -v single $JOBFILE >> logs/.seeds10_ordered 2>/dev/null
xargs -a logs/.seeds10_ordered -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo "ALL_DONE" > logs/train_seeds10.done
