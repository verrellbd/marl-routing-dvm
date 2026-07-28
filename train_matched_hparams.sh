#!/usr/bin/env bash
# Matched-hyperparameter re-run, capacity-aware (--metric weighted).
#
# WHY THIS SHAPE: the single agent is the BASELINE. Trimming its PPO settings to match our
# MAPPO (fewer epochs = less optimisation) would read as handicapping the baseline to win.
# SB3's defaults are the community standard; GNNMAPPO is the custom code. So the BASELINE
# KEEPS ITS CONFIGURATION UNCHANGED and OUR method moves to meet it.
#
# The four mismatches being closed (audited 2026-07-28), MARL -> single:
#   gamma       0.99  -> 0.995
#   n_epochs    6     -> 10
#   minibatch   512   -> 256
#   buffer      4096  -> 2048  (=> 732 updates instead of 366; 2048*732 = 1,499,136 steps)
# Already identical and untouched: lr 3e-4, gae_lambda 0.95, clip 0.2, vf_coef 0.5,
# max_grad_norm 0.5, ent_coef 0.01.
#
# The ENVIRONMENT side was audited and needs no change: both methods already use the same
# 17 training topologies, the same TMgen call (n_patterns=3, load_scales 0.6-1.5, same seed),
# the same max_flows=500 filter, the same test matrices (real_matrices n_per_scale=6,
# split=test), delay_penalty 0.5, normalize_reward True, eval seed A.seed+1.
#
# STILL NOT MATCHABLE, and reported as a limitation rather than papered over: one
# single-agent step routes a WHOLE FLOW, one MARL step routes ONE HOP. 1.5M steps is
# therefore ~6x more flows routed for the single agent. Matched on env steps, unmatched on
# work. Report a second budget axis (flows routed) and show the conclusion is invariant.
#
# Tags: _cm = capacity-aware + matched hyperparameters. The earlier _cap MARL runs
# (n_epochs 6, gamma 0.99) stay on disk as the "MARL at its own defaults" ablation.
cd ~/thesis || exit 1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
JOBS=${JOBS:-9}
mkdir -p logs
JOBFILE=logs/train_matched_jobs.txt
: > $JOBFILE

for s in 0 1 2; do
  # ---- single agent: SB3 DEFAULTS, DELIBERATELY UNCHANGED (this is the baseline) ----
  echo "nice -n 19 python train_single_tier2.py --seed $s --timesteps 1500000 --traffic tmgen --k-paths 3 --hidden 64 --rounds 3 --n-envs 8 --n-steps 256 --ent-coef 0.01 --metric weighted --tag _singleH64gcap > logs/train_singlecap_s$s.log 2>&1 && echo OK single_s$s || echo FAIL single_s$s" >> $JOBFILE
  # ---- MARL: moved onto the baseline's PPO settings ----
  echo "nice -n 19 python train_marl_gnn_tier2.py --seed $s --updates 732 --rollout 2048 --gamma 0.995 --n-epochs 10 --minibatch 256 --hidden 32 --rounds 3 --traffic tmgen --metric weighted --tag _tier2m15cm > logs/train_marlh32cm_s$s.log 2>&1 && echo OK marlh32_s$s || echo FAIL marlh32_s$s" >> $JOBFILE
  echo "nice -n 19 python train_marl_gnn_tier2.py --seed $s --updates 732 --rollout 2048 --gamma 0.995 --n-epochs 10 --minibatch 256 --hidden 64 --rounds 3 --traffic tmgen --metric weighted --tag _tier2m15h64cm > logs/train_marlh64cm_s$s.log 2>&1 && echo OK marlh64_s$s || echo FAIL marlh64_s$s" >> $JOBFILE
done

echo "[matched] $(wc -l < $JOBFILE) runs, JOBS=$JOBS"
# single agent is the long pole (~6h vs ~2h); start those first so they overlap
grep single $JOBFILE  > logs/.matched_ordered
grep -v single $JOBFILE >> logs/.matched_ordered
xargs -a logs/.matched_ordered -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo "ALL_DONE" > logs/train_matched.done
