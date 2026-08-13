#!/usr/bin/env bash
# Packet-level evaluation of the seeds added by train_seeds10.sh.
#
# Arms: singleRM | marlh32 | ecmp.  h=64 is dropped (not reported).
#
# OSPF IS NOT RE-SIMULATED. It is deterministic given the matrix -- the routing depends
# only on topology and cost metric, and the matrices are chosen deterministically, so the
# ospf_path in every seed's routing file is identical. Verified: ns3_ospf_{0,8,11}.json
# agree to four decimals across seeds 0/1/2. Re-running it for seven more seeds would add
# 126 simulations, ~42 of them Germany50 at ~17 min each, for bit-identical output. The
# existing seed 0-2 OSPF runs remain the reference for every paired comparison.
#
# Loads, flow caps and stratification are copied verbatim from run_ns3_final.sh and
# run_ns3_rewardform.sh so the new seeds land on exactly the same matrices as 0-2.
# Abilene uses 16/22/28 (a correctly weighted OSPF has no overload regime below ~16);
# Germany50's feasible regime needs loads <=30 and lives in separate _g50feas_ dirs.
#
# Idempotent: existing export dirs and existing state files are skipped, so a killed run
# resumes where it stopped.
#
# --no-build IS LOAD-BEARING. Plain `./ns3 run` checks the build first, and contrib/ospf
# (the ns3-ospf side quest) leaves the tree perpetually out of date, so every job spawns
# its own ninja build. At JOBS=16 that is sixteen concurrent LTO link steps, which
# exhausted memory on a shared machine and killed the whole run -- 378 queued sims, zero
# state files, and `xargs: cannot fork`. The scratch binary has not changed since
# 2026-07-28 and is the same one that produced the reported grid, so skipping the build
# check is correct as well as necessary. Rebuild deliberately with `./ns3 build` if the
# scenario source is ever edited.
cd ~/thesis || exit 1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NS3_TIMEOUT=${NS3_TIMEOUT:-3600}
SEEDS=${SEEDS:-"3 4 5 6 7 8 9"}
JOBS=${JOBS:-16}
NPER=6
MAXFLOWS=200
RATESCALE=20
SIMTIME=8
mkdir -p logs

# ---------- 1) exports ----------
for s in $SEEDS; do
  for spec in "abilene_sndlib:16,22,28:abilene:" \
              "geant_sndlib:3,5,7:geant:" \
              "germany50_sndlib:35,50,65:germany50:" \
              "germany50_sndlib:15,20,25,30:g50feas:--n-overload 0 --n-feasible 3"; do
    IFS=':' read -r T L SHORT STRAT <<< "$spec"

    o="ns3f_singleRM_${SHORT}_s$s"
    if [ ! -d "results/$o" ]; then
      python export_topoagn_routes.py --topo "$T" \
        --model "results/single_singleH64gRM_seed$s/policy.zip" \
        --loads "$L" --n-per-scale $NPER --max-flows $MAXFLOWS --k-paths 3 \
        --metric weighted $STRAT --out "$o" > "logs/ns3f_$o.log" 2>&1 || echo "EXPORT FAIL $o"
    fi

    o="ns3f_marlh32_${SHORT}_s$s"
    if [ ! -d "results/$o" ]; then
      python export_topoagn_marl_routes.py --topo "$T" \
        --model "results/marlgnn_tier2m15cm_seed$s/policy.pt" \
        --hidden 32 --rounds 3 --loads "$L" --n-per-scale $NPER \
        --max-flows $MAXFLOWS --metric weighted $STRAT --out "$o" > "logs/ns3f_$o.log" 2>&1 || echo "EXPORT FAIL $o"
    fi

    o="ns3f_ecmp_${SHORT}_s$s"
    if [ ! -d "results/$o" ]; then
      python export_ecmp_routes.py --topo "$T" --loads "$L" --n-per-scale $NPER \
        --max-flows $MAXFLOWS --seed $s --metric weighted $STRAT --out "$o" \
        > "logs/ns3f_$o.log" 2>&1 || echo "EXPORT FAIL $o"
    fi
  done
done
echo "[seeds10] exports done"

# ---------- 2) sims (learned/ecmp path only; see the OSPF note above) ----------
JOBFILE=logs/ns3_seeds10_jobs.txt
: > $JOBFILE
for s in $SEEDS; do
  for arm in singleRM marlh32 ecmp; do
    for short in abilene geant germany50 g50feas; do
      d="results/ns3f_${arm}_${short}_s$s"
      [ -d "$d" ] || continue
      case $short in
        abilene) T=abilene_sndlib ;;
        geant)   T=geant_sndlib ;;
        *)       T=germany50_sndlib ;;
      esac
      if [ "$arm" = ecmp ]; then pref=ns3_ecmp; else pref=ns3_gnn; fi
      for rf in $d/routing_seed*.json; do
        [ -e "$rf" ] || continue
        i=$(basename "$rf" .json); i=${i#routing_seed}
        st="$HOME/thesis/results/ns3f_${arm}_${short}_s$s/${pref}_$i.json"
        [ -f "$st" ] && continue
        echo "cd ~/thesis/ns-3-dev && ./ns3 run --no-build \"scratch/abilene-validate/abilene-validate --topo=$HOME/thesis/topologies/$T.json --routing_file=$HOME/thesis/$rf --routing=gnn --state=$st --simTime=$SIMTIME --rateScale=$RATESCALE\" > /dev/null 2>&1 || echo SIMFAIL ${arm}_${short}_s$s $i" >> $JOBFILE
      done
    done
  done
done

echo "[seeds10] $(wc -l < $JOBFILE) sims, JOBS=$JOBS"
# Germany50 is the slow one (~17 min/sim) -> front-load so it overlaps the fast topologies
grep -E "germany50" $JOBFILE  > logs/.seeds10_sim_ordered
grep -vE "germany50" $JOBFILE >> logs/.seeds10_sim_ordered 2>/dev/null
xargs -a logs/.seeds10_sim_ordered -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo "ALL_DONE" > logs/ns3_seeds10.done
