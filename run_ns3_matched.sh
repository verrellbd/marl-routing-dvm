#!/usr/bin/env bash
# Packet-level (ns-3) validation of the matched-1.5M-budget arms, 3 seeds each,
# on the 3 held-out backbones with REAL measured traffic.
#
#   arms: single (SB3, hidden 64) | MARL h32 | MARL h64
#   per (arm, topo, seed): export routes -> run_ns3_phase2 (OSPF + learned, 6 matrices)
#   metrics from the per-sim JSONs: loss_pct, mean_delay_ms, throughput_mbps,
#   max_offered_util_pct, link_utils
#
# Loads/stratification match the analytical grid (TEST_LOADS in the trainers), so the
# ns-3 numbers correspond to the same matrices reported analytically.
cd ~/thesis || exit 1

MAXFLOWS=200
NPER=6
JOBS=${JOBS:-12}
export NS3_TIMEOUT=${NS3_TIMEOUT:-3600}
# CRITICAL: without these, each concurrent export spawns torch threads per core (128!)
# -> load explodes and imports die with MemoryError/segfault. One thread per job.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1

JOBFILE=logs/ns3_matched_jobs.txt
: > $JOBFILE

for s in 0 1 2; do
  for t in abilene_sndlib geant_sndlib germany50_sndlib; do
    case $t in
      abilene_sndlib)   L="8,12,16" ;;
      geant_sndlib)     L="3,5,7" ;;
      germany50_sndlib) L="35,50,65" ;;
    esac
    short=${t%_sndlib}
    # single-agent (SB3 policy.zip)
    o="ns3m_single_${short}_s$s"
    echo "python export_topoagn_routes.py --topo $t --model results/single_singleH64g_seed$s/policy.zip --loads $L --n-per-scale $NPER --max-flows $MAXFLOWS --k-paths 3 --out $o > logs/ns3_$o.log 2>&1 && python run_ns3_phase2.py --dir results/$o --topo $t --ratescale 20 >> logs/ns3_$o.log 2>&1 && echo OK $o || echo FAIL $o" >> $JOBFILE
    # MARL h32
    o="ns3m_marlh32_${short}_s$s"
    echo "python export_topoagn_marl_routes.py --topo $t --model results/marlgnn_tier2m15_seed$s/policy.pt --hidden 32 --rounds 3 --loads $L --n-per-scale $NPER --max-flows $MAXFLOWS --out $o > logs/ns3_$o.log 2>&1 && python run_ns3_phase2.py --dir results/$o --topo $t --ratescale 20 >> logs/ns3_$o.log 2>&1 && echo OK $o || echo FAIL $o" >> $JOBFILE
    # MARL h64
    o="ns3m_marlh64_${short}_s$s"
    echo "python export_topoagn_marl_routes.py --topo $t --model results/marlgnn_tier2m15h64_seed$s/policy.pt --hidden 64 --rounds 3 --loads $L --n-per-scale $NPER --max-flows $MAXFLOWS --out $o > logs/ns3_$o.log 2>&1 && python run_ns3_phase2.py --dir results/$o --topo $t --ratescale 20 >> logs/ns3_$o.log 2>&1 && echo OK $o || echo FAIL $o" >> $JOBFILE
  done
done

echo "[ns3-matched] $(wc -l < $JOBFILE) jobs, JOBS=$JOBS, NS3_TIMEOUT=$NS3_TIMEOUT"
# germany50 (slowest) first so it overlaps the fast ones instead of trailing at the end
grep germany50 $JOBFILE  > logs/.ns3_ordered
grep -v germany50 $JOBFILE >> logs/.ns3_ordered
xargs -a logs/.ns3_ordered -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo "ALL_DONE" > logs/ns3_matched.done
