#!/usr/bin/env bash
# FINAL packet-level grid: capacity-aware (--metric weighted) + matched hyperparameters.
#
# Supersedes run_ns3_matched.sh / run_ns3_ecmp.sh / run_ns3_g50feas.sh, which used
# capacity-blind candidate paths (--metric reached only ospf_path + stratification, never
# the env) and the pre-matching MARL settings.
#
# Arms (all at matched 1.5M steps, all --metric weighted):
#   single   results/single_singleH64gcap_seed*   (SB3 defaults, DELIBERATELY unchanged --
#                                                  it is the baseline)
#   marlh32  results/marlgnn_tier2m15cm_seed*     (moved onto SB3's PPO settings)
#   marlh64  results/marlgnn_tier2m15h64cm_seed*
#   ecmp     equal-COST split (export_ecmp_routes.py --metric weighted)
#   ospf     re-simulated inside every non-ECMP dir
#
# LOADS. abilene uses 16/22/28, NOT the trainers' 8/12/16: under a correctly weighted OSPF
# abilene has no overload regime at all below ~16 (0/18 matrices >=100%), so the lower loads
# would yield a feasible-only row. geant/germany50 keep the trainers' TEST_LOADS so the
# packet-level numbers correspond to the analytical grid. germany50's feasible regime needs
# loads <=30 and lives in separate _g50feas_ dirs (its TEST_LOADS are all overload).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_DIR="${NS3_DIR:-$REPO/ns-3-dev}"
cd "$REPO" || exit 1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NS3_TIMEOUT=${NS3_TIMEOUT:-3600}
JOBS=${JOBS:-16}
NPER=6
MAXFLOWS=200
RATESCALE=20
SIMTIME=8

model_of() {  # arm seed -> "<model path> <extra export args>"
  case $1 in
    single)  echo "results/single_singleH64gcap_seed$2/policy.zip" ;;
    marlh32) echo "results/marlgnn_tier2m15cm_seed$2/policy.pt" ;;
    marlh64) echo "results/marlgnn_tier2m15h64cm_seed$2/policy.pt" ;;
  esac
}

# ---------- 1) exports (cheap relative to the sims; sequential keeps logs readable) ----------
for s in 0 1 2; do
  for spec in "abilene_sndlib:16,22,28:abilene::" \
              "geant_sndlib:3,5,7:geant::" \
              "germany50_sndlib:35,50,65:germany50::" \
              "germany50_sndlib:15,20,25,30:g50feas:--n-overload 0 --n-feasible 3:"; do
    IFS=':' read -r T L SHORT STRAT _ <<< "$spec"
    for arm in single marlh32 marlh64 ecmp; do
      o="ns3f_${arm}_${SHORT}_s$s"
      [ -d "results/$o" ] && continue
      case $arm in
        single)  python export_topoagn_routes.py --topo "$T" --model "$(model_of single $s)" \
                   --loads "$L" --n-per-scale $NPER --max-flows $MAXFLOWS --k-paths 3 \
                   --metric weighted $STRAT --out "$o" > "logs/ns3f_$o.log" 2>&1 ;;
        marlh32) python export_topoagn_marl_routes.py --topo "$T" --model "$(model_of marlh32 $s)" \
                   --hidden 32 --rounds 3 --loads "$L" --n-per-scale $NPER \
                   --max-flows $MAXFLOWS --metric weighted $STRAT --out "$o" > "logs/ns3f_$o.log" 2>&1 ;;
        marlh64) python export_topoagn_marl_routes.py --topo "$T" --model "$(model_of marlh64 $s)" \
                   --hidden 64 --rounds 3 --loads "$L" --n-per-scale $NPER \
                   --max-flows $MAXFLOWS --metric weighted $STRAT --out "$o" > "logs/ns3f_$o.log" 2>&1 ;;
        ecmp)    python export_ecmp_routes.py --topo "$T" --loads "$L" --n-per-scale $NPER \
                   --max-flows $MAXFLOWS --seed $s --metric weighted $STRAT --out "$o" \
                   > "logs/ns3f_$o.log" 2>&1 ;;
      esac || echo "EXPORT FAIL $o"
    done
  done
done
echo "[final] exports done"

# ---------- 2) sims: OSPF + learned per model dir; ECMP-only in the ECMP dirs ----------
JOBFILE=logs/ns3_final_jobs.txt
: > $JOBFILE
for d in results/ns3f_*; do
  [ -d "$d" ] || continue
  o=$(basename "$d")
  arm=$(echo "$o" | cut -d_ -f2); short=$(echo "$o" | cut -d_ -f3)
  case $short in
    abilene)   T=abilene_sndlib ;;
    geant)     T=geant_sndlib ;;
    germany50|g50feas) T=germany50_sndlib ;;
  esac
  for rf in $d/routing_seed*.json; do
    [ -e "$rf" ] || continue
    i=$(basename "$rf" .json); i=${i#routing_seed}
    if [ "$arm" = ecmp ]; then modes="gnn:ns3_ecmp"; else modes="ospf:ns3_ospf gnn:ns3_gnn"; fi
    for m in $modes; do
      mode=${m%%:*}; pref=${m##*:}
      st="$REPO/results/$o/${pref}_$i.json"
      [ -f "$st" ] && continue
      echo "cd $NS3_DIR && ./ns3 run \"scratch/abilene-validate/abilene-validate --topo=$REPO/topologies/$T.json --routing_file=$REPO/$rf --routing=$mode --state=$st --simTime=$SIMTIME --rateScale=$RATESCALE\" > /dev/null 2>&1 || echo SIMFAIL $o $i $mode" >> $JOBFILE
    done
  done
done

echo "[final] $(wc -l < $JOBFILE) sims, JOBS=$JOBS"
# germany50 is the slow one (~17min/sim) -> front-load it so it overlaps the fast topologies
grep -E "germany50|g50feas" $JOBFILE  > logs/.final_ordered
grep -vE "germany50|g50feas" $JOBFILE >> logs/.final_ordered
xargs -a logs/.final_ordered -P "$JOBS" -d '\n' -I CMD bash -c CMD
echo "ALL_DONE" > logs/ns3_final.done
