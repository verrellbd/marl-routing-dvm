#!/usr/bin/env bash
# ns-3 packet-level ECMP baseline for the matched-budget comparison.
# Only the ECMP ("gnn" key) half is simulated — OSPF is already measured by the main batch.
cd ~/thesis || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NS3_TIMEOUT=${NS3_TIMEOUT:-3600}
JOBS=${JOBS:-9}
RATESCALE=20
SIMTIME=8

# 1) export routing JSONs (cheap, sequential)
for s in 0 1 2; do
  for t in abilene_sndlib geant_sndlib germany50_sndlib; do
    case $t in
      abilene_sndlib)   L="8,12,16" ;;
      geant_sndlib)     L="3,5,7" ;;
      germany50_sndlib) L="35,50,65" ;;
    esac
    short=${t%_sndlib}
    o="ns3m_ecmp_${short}_s$s"
    [ -f "results/$o/.exported" ] && continue
    python export_ecmp_routes.py --topo $t --loads $L --n-per-scale 6 --max-flows 200 \
      --seed $s --out "$o" > "logs/ns3_$o.log" 2>&1 && touch "results/$o/.exported" \
      || echo "EXPORT FAIL $o"
  done
done
echo "[ecmp] exports done"

# 2) one ns-3 sim per (dir, matrix), ECMP routing only
JOBFILE=logs/ns3_ecmp_jobs.txt
: > $JOBFILE
for s in 0 1 2; do
  for t in abilene_sndlib geant_sndlib germany50_sndlib; do
    short=${t%_sndlib}; o="ns3m_ecmp_${short}_s$s"
    for rf in results/$o/routing_seed*.json; do
      [ -e "$rf" ] || continue
      i=$(basename "$rf" .json); i=${i#routing_seed}
      st="$HOME/thesis/results/$o/ns3_ecmp_$i.json"
      [ -f "$st" ] && continue
      echo "cd ~/thesis/ns-3-dev && ./ns3 run \"scratch/abilene-validate/abilene-validate --topo=$HOME/thesis/topologies/$t.json --routing_file=$HOME/thesis/$rf --routing=gnn --state=$st --simTime=$SIMTIME --rateScale=$RATESCALE\" > /dev/null 2>&1 || echo SIMFAIL $o $i" >> $JOBFILE
    done
  done
done
echo "[ecmp] $(wc -l < $JOBFILE) sims, JOBS=$JOBS"
grep germany50 $JOBFILE > logs/.ecmp_ordered
grep -v germany50 $JOBFILE >> logs/.ecmp_ordered
xargs -a logs/.ecmp_ordered -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo "ALL_DONE" > logs/ns3_ecmp.done
