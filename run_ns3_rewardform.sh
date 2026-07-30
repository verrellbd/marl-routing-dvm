#!/usr/bin/env bash
# Packet-level evaluation of the single agent trained under MARL's EXACT reward form
# (results/single_singleH64gRM_seed*, --reward-form marl), so the two reward forms can be
# compared where it counts rather than only on the analytical surrogate.
#
# Analytically the marl form has better means in 3 of 4 cells but larger seed spreads
# (one Germany50 seed collapses to OSPF level), so the analytical comparison is
# inconclusive -- hence this run.
#
# Only the LEARNED side is simulated. Matrix stratification depends on OSPF utilisation
# alone, so the retained matrices are identical to results/ns3f_single_*; the OSPF rows are
# reused from there rather than re-simulated.
cd ~/thesis || exit 1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NS3_TIMEOUT=${NS3_TIMEOUT:-3600}
JOBS=${JOBS:-12}
NPER=6; MAXFLOWS=200; RATESCALE=20; SIMTIME=8

for s in 0 1 2; do
  for spec in "abilene_sndlib:16,22,28:abilene:" \
              "geant_sndlib:3,5,7:geant:" \
              "germany50_sndlib:35,50,65:germany50:" \
              "germany50_sndlib:15,20,25,30:g50feas:--n-overload 0 --n-feasible 3"; do
    IFS=':' read -r T L SHORT STRAT <<< "$spec"
    o="ns3f_singleRM_${SHORT}_s$s"
    [ -d "results/$o" ] && continue
    python export_topoagn_routes.py --topo "$T" \
      --model "results/single_singleH64gRM_seed$s/policy.zip" \
      --loads "$L" --n-per-scale $NPER --max-flows $MAXFLOWS --k-paths 3 \
      --metric weighted $STRAT --out "$o" > "logs/ns3f_$o.log" 2>&1 \
      || echo "EXPORT FAIL $o"
  done
done
echo "[rewardform] exports done"

JOBFILE=logs/ns3_rewardform_jobs.txt
: > $JOBFILE
for d in results/ns3f_singleRM_*; do
  [ -d "$d" ] || continue
  o=$(basename "$d"); short=$(echo "$o" | cut -d_ -f3)
  case $short in
    abilene) T=abilene_sndlib ;;
    geant)   T=geant_sndlib ;;
    *)       T=germany50_sndlib ;;
  esac
  for rf in $d/routing_seed*.json; do
    [ -e "$rf" ] || continue
    i=$(basename "$rf" .json); i=${i#routing_seed}
    st="$HOME/thesis/results/$o/ns3_gnn_$i.json"
    [ -f "$st" ] && continue
    echo "cd ~/thesis/ns-3-dev && ./ns3 run \"scratch/abilene-validate/abilene-validate --topo=$HOME/thesis/topologies/$T.json --routing_file=$HOME/thesis/$rf --routing=gnn --state=$st --simTime=$SIMTIME --rateScale=$RATESCALE\" > /dev/null 2>&1 || echo SIMFAIL $o $i" >> $JOBFILE
  done
done
echo "[rewardform] $(wc -l < $JOBFILE) sims, JOBS=$JOBS"
grep -E "germany50|g50feas" $JOBFILE  > logs/.rf_ordered
grep -vE "germany50|g50feas" $JOBFILE >> logs/.rf_ordered
xargs -a logs/.rf_ordered -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo ALL_DONE > logs/ns3_rewardform.done
