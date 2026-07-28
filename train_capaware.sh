#!/usr/bin/env bash
# CAPACITY-AWARE re-run of the matched-1.5M-budget arms (3 seeds x 3 arms).
#
# WHY: our learned methods chose among HOP-shortest candidate paths with HOP-based detour
# limits, so on a heterogeneous-capacity topology the slow link sits in their candidate set
# and they cannot route around it. On abilene_sndlib (14x9.92G + 1x2.48G) that lost us the
# comparison outright: weighted OSPF 57.25% vs MARL h64 70.64%. See CLAUDE.md.
#
# THE FIX: --metric weighted everywhere -> OSPF cost refBW/linkBW governs the k-shortest
# candidates, the dist_to progress test, the stretch limits, and the OSPF/ECMP references.
# Verified before launch:
#   abilene  slow link in k=3 candidates 112/396 -> 50/396; random-policy max util 80.4 -> 40.2
#   geant    byte-identical under both metrics (uniform 40G) — nothing else can regress
#   germany50 max util identical (69.9), OSPF ref 28.6 -> 28.2 (equal-cost tie-breaking only)
#
# EVERYTHING ELSE IS UNCHANGED from the runs this replaces (recovered from the saved
# policies, not guessed): single = k3/hidden64/rounds3/8 envs/n_steps256/ent 0.01/1.5M steps;
# MARL = rounds3/rollout4096/366 updates (=1,499,136 steps), hidden 32 and 64.
#
# New tags (…cap) so the capacity-blind results stay on disk for the before/after table.
cd ~/thesis || exit 1

# One torch thread per job: 9 uncapped procs on a 128-core box took geneva to load 275.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
JOBS=${JOBS:-9}
UPDATES=366          # x4096 rollout = 1.5M env steps, matched to the single-agent budget
STEPS=1500000

JOBFILE=logs/train_capaware_jobs.txt
mkdir -p logs
: > $JOBFILE

for s in 0 1 2; do
  echo "nice -n 19 python train_marl_gnn_tier2.py --seed $s --updates $UPDATES --rollout 4096 --hidden 32 --rounds 3 --traffic tmgen --metric weighted --tag _tier2m15cap > logs/train_marlh32cap_s$s.log 2>&1 && echo OK marlh32_s$s || echo FAIL marlh32_s$s" >> $JOBFILE
  echo "nice -n 19 python train_marl_gnn_tier2.py --seed $s --updates $UPDATES --rollout 4096 --hidden 64 --rounds 3 --traffic tmgen --metric weighted --tag _tier2m15h64cap > logs/train_marlh64cap_s$s.log 2>&1 && echo OK marlh64_s$s || echo FAIL marlh64_s$s" >> $JOBFILE
  echo "nice -n 19 python train_single_tier2.py --seed $s --timesteps $STEPS --traffic tmgen --k-paths 3 --hidden 64 --rounds 3 --n-envs 8 --n-steps 256 --ent-coef 0.01 --metric weighted --tag _singleH64gcap > logs/train_singlecap_s$s.log 2>&1 && echo OK single_s$s || echo FAIL single_s$s" >> $JOBFILE
done

echo "[capaware] $(wc -l < $JOBFILE) runs, JOBS=$JOBS"
# single-agent is the long pole (~3h vs ~40min); start those first so they overlap
grep single $JOBFILE  > logs/.cap_ordered
grep -v single $JOBFILE >> logs/.cap_ordered
xargs -a logs/.cap_ordered -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo "ALL_DONE" > logs/train_capaware.done
