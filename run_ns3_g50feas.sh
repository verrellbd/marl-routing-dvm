#!/usr/bin/env bash
# Fill the missing germany50 FEASIBLE regime.
#
# The trainers' TEST_LOADS for germany50 (35/50/65) are ALL above capacity (OSPF 100-222%),
# so germany50 only ever produced overload matrices -> no feasible row, unlike abilene and
# geant. Here we re-export at loads 15/20/25/30 (OSPF ~43-103%) and take the 3 highest
# matrices with OSPF < 100%, exactly as the stratifier does for the other two topologies.
#
# Written to *separate* dirs (…_g50feas_…) because matrix indices from a different load
# list would otherwise collide with the existing overload files.
cd ~/thesis || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NS3_TIMEOUT=${NS3_TIMEOUT:-3600}
JOBS=${JOBS:-16}
T=germany50_sndlib
L="15,20,25,30"
NPER=6
MAXFLOWS=200
STRAT="--n-overload 0 --n-feasible 3"

JOBFILE=logs/ns3_g50feas_jobs.txt
: > $JOBFILE

# ---- 1) exports (sequential; cheap relative to the sims) ----
for s in 0 1 2; do
  o="ns3m_single_g50feas_s$s"
  [ -d "results/$o" ] || python export_topoagn_routes.py --topo $T \
      --model results/single_singleH64g_seed$s/policy.zip --loads $L --n-per-scale $NPER \
      --max-flows $MAXFLOWS --k-paths 3 $STRAT --out "$o" > "logs/ns3_$o.log" 2>&1
  o="ns3m_marlh32_g50feas_s$s"
  [ -d "results/$o" ] || python export_topoagn_marl_routes.py --topo $T \
      --model results/marlgnn_tier2m15_seed$s/policy.pt --hidden 32 --rounds 3 --loads $L \
      --n-per-scale $NPER --max-flows $MAXFLOWS $STRAT --out "$o" > "logs/ns3_$o.log" 2>&1
  o="ns3m_marlh64_g50feas_s$s"
  [ -d "results/$o" ] || python export_topoagn_marl_routes.py --topo $T \
      --model results/marlgnn_tier2m15h64_seed$s/policy.pt --hidden 64 --rounds 3 --loads $L \
      --n-per-scale $NPER --max-flows $MAXFLOWS $STRAT --out "$o" > "logs/ns3_$o.log" 2>&1
  o="ns3m_ecmp_g50feas_s$s"
  [ -d "results/$o" ] || python export_ecmp_routes.py --topo $T --loads $L \
      --n-per-scale $NPER --max-flows $MAXFLOWS --seed $s $STRAT --out "$o" \
      > "logs/ns3_$o.log" 2>&1
done
echo "[g50feas] exports done"

# ---- 2) sims: OSPF + learned for the model arms; ECMP-only for the ECMP dirs ----
for s in 0 1 2; do
  for arm in single marlh32 marlh64 ecmp; do
    o="ns3m_${arm}_g50feas_s$s"
    for rf in results/$o/routing_seed*.json; do
      [ -e "$rf" ] || continue
      i=$(basename "$rf" .json); i=${i#routing_seed}
      if [ "$arm" = ecmp ]; then modes="gnn:ns3_ecmp"; else modes="ospf:ns3_ospf gnn:ns3_gnn"; fi
      for m in $modes; do
        mode=${m%%:*}; pref=${m##*:}
        st="$HOME/thesis/results/$o/${pref}_$i.json"
        [ -f "$st" ] && continue
        echo "cd ~/thesis/ns-3-dev && ./ns3 run \"scratch/abilene-validate/abilene-validate --topo=$HOME/thesis/topologies/$T.json --routing_file=$HOME/thesis/$rf --routing=$mode --state=$st --simTime=8 --rateScale=20\" > /dev/null 2>&1 || echo SIMFAIL $o $i $mode" >> $JOBFILE
      done
    done
  done
done
echo "[g50feas] $(wc -l < $JOBFILE) sims, JOBS=$JOBS"
xargs -a $JOBFILE -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo "ALL_DONE" > logs/ns3_g50feas.done
