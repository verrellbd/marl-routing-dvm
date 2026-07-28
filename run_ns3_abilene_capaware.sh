#!/usr/bin/env bash
# Abilene with CAPACITY-AWARE LEARNED PATHS (the _abilenew_ re-baseline fixed only the
# OSPF/ECMP side: back then --metric reached ospf_path + stratification but NOT the env,
# so the learned candidate paths were still hop-shortest and still depended on the 2.48G
# link). Since the envs now take metric=, re-exporting the SAME saved policies with today's
# code changes the learned paths themselves — 39/132 candidate sets differ on abilene.
# No retraining needed: the policies are topology-agnostic and consume whatever candidates
# the env hands them. Same loads/seeds/flows as _abilenew_ so the two are directly comparable.
cd ~/thesis || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NS3_TIMEOUT=${NS3_TIMEOUT:-3600}
JOBS=${JOBS:-16}
T=abilene_sndlib; L="16,22,28"; M=200; NPER=6
JOBFILE=logs/ns3_abilenecap_jobs.txt; : > $JOBFILE
for s in 0 1 2; do
  o="ns3m_single_abilenecap_s$s";  [ -d results/$o ] || python export_topoagn_routes.py --topo $T \
     --model results/single_singleH64g_seed$s/policy.zip --loads $L --n-per-scale $NPER \
     --max-flows $M --k-paths 3 --metric weighted --out $o > logs/ns3_$o.log 2>&1
  o="ns3m_marlh32_abilenecap_s$s"; [ -d results/$o ] || python export_topoagn_marl_routes.py --topo $T \
     --model results/marlgnn_tier2m15_seed$s/policy.pt --hidden 32 --rounds 3 --loads $L \
     --n-per-scale $NPER --max-flows $M --metric weighted --out $o > logs/ns3_$o.log 2>&1
  o="ns3m_marlh64_abilenecap_s$s"; [ -d results/$o ] || python export_topoagn_marl_routes.py --topo $T \
     --model results/marlgnn_tier2m15h64_seed$s/policy.pt --hidden 64 --rounds 3 --loads $L \
     --n-per-scale $NPER --max-flows $M --metric weighted --out $o > logs/ns3_$o.log 2>&1
  o="ns3m_ecmp_abilenecap_s$s";    [ -d results/$o ] || python export_ecmp_routes.py --topo $T \
     --loads $L --n-per-scale $NPER --max-flows $M --seed $s --metric weighted \
     --out $o > logs/ns3_$o.log 2>&1
done
echo "[abilenecap] exports done"
for s in 0 1 2; do
  for arm in single marlh32 marlh64 ecmp; do
    o="ns3m_${arm}_abilenecap_s$s"
    for rf in results/$o/routing_seed*.json; do
      [ -e "$rf" ] || continue
      i=$(basename "$rf" .json); i=${i#routing_seed}
      if [ "$arm" = ecmp ]; then modes="gnn:ns3_ecmp"; else modes="ospf:ns3_ospf gnn:ns3_gnn"; fi
      for m in $modes; do
        mode=${m%%:*}; pref=${m##*:}; st="$HOME/thesis/results/$o/${pref}_$i.json"
        [ -f "$st" ] && continue
        echo "cd ~/thesis/ns-3-dev && ./ns3 run \"scratch/abilene-validate/abilene-validate --topo=$HOME/thesis/topologies/$T.json --routing_file=$HOME/thesis/$rf --routing=$mode --state=$st --simTime=8 --rateScale=20\" > /dev/null 2>&1 || echo SIMFAIL $o $i $mode" >> $JOBFILE
      done
    done
  done
done
echo "[abilenecap] $(wc -l < $JOBFILE) sims, JOBS=$JOBS"
xargs -a $JOBFILE -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo ALL_DONE > logs/ns3_abilenecap.done
