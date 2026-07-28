#!/usr/bin/env bash
# ns-3 NATIVE ECMP (Ipv4GlobalRouting + RandomEcmpRouting) as a second ECMP variant.
# Differs from our flow-level ECMP: ns-3 randomises among equal-cost next hops per lookup
# (per packet), so it models packet-level spraying rather than 5-tuple flow hashing.
# Seed variance comes from --RngRun (the routing JSONs' own hash choice is ignored here).
cd ~/thesis || exit 1
export NS3_TIMEOUT=${NS3_TIMEOUT:-3600}
JOBS=${JOBS:-16}
JOBFILE=logs/ns3_ecmpnat_jobs.txt
: > $JOBFILE
for s in 0 1 2; do
  for base in abilene geant germany50 g50feas; do
    case $base in
      abilene)  T=abilene_sndlib ;;
      geant)    T=geant_sndlib ;;
      *)        T=germany50_sndlib ;;
    esac
    src="results/ns3m_ecmp_${base}_s$s"; out="results/ns3m_ecmpnat_${base}_s$s"
    [ -d "$src" ] || continue
    mkdir -p "$out"; cp -n "$src"/routing_seed*.json "$out"/ 2>/dev/null
    for rf in "$out"/routing_seed*.json; do
      i=$(basename "$rf" .json); i=${i#routing_seed}
      st="$HOME/thesis/$out/ns3_gnn_$i.json"
      [ -f "$st" ] && continue
      echo "cd ~/thesis/ns-3-dev && ./ns3 run \"scratch/abilene-validate/abilene-validate --topo=$HOME/thesis/topologies/$T.json --routing_file=$HOME/thesis/$rf --routing=ecmp --state=$st --simTime=8 --rateScale=20 --RngRun=$((s+1))\" > /dev/null 2>&1 || echo SIMFAIL $out $i" >> $JOBFILE
    done
  done
done
echo "[ecmpnat] $(wc -l < $JOBFILE) sims, JOBS=$JOBS"
grep -E "germany50|g50feas" $JOBFILE > logs/.ecmpnat_ordered
grep -vE "germany50|g50feas" $JOBFILE >> logs/.ecmpnat_ordered
xargs -a logs/.ecmpnat_ordered -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo "ALL_DONE" > logs/ns3_ecmpnat.done
