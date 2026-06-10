#!/bin/bash
# Baseline comparison: OSPF vs ECMP with min 30 Mbps flows
# This generates ~100 flows per load factor to maximize network competition
# and hopefully show differences between OSPF and ECMP

set -e
cd /home/uceedv1/thesis

echo "=========================================="
echo "Step 1: Generate min 30 Mbps traffic flows"
echo "=========================================="
python -m marl_routing.traffic abilene 0.3 30
python -m marl_routing.traffic abilene 1.0 30
python -m marl_routing.traffic abilene 1.5 30

echo ""
echo "=========================================="
echo "Step 2: Run OSPF baselines"
echo "=========================================="
cd ./ns-3-dev

echo -e "\n[OSPF α=0.3]"
time ./ns3 run "scratch/abilene/abilene --topo=/home/uceedv1/thesis/topologies/abilene.json --traffic=/home/uceedv1/thesis/results/traffic_abilene_α0.3_min30.json --simTime=30" 2>&1 | tee ~/ospf_0.3_min30.log

echo -e "\n[OSPF α=1.0]"
time ./ns3 run "scratch/abilene/abilene --topo=/home/uceedv1/thesis/topologies/abilene.json --traffic=/home/uceedv1/thesis/results/traffic_abilene_α1.0_min30.json --simTime=30" 2>&1 | tee ~/ospf_1.0_min30.log

echo -e "\n[OSPF α=1.5]"
time ./ns3 run "scratch/abilene/abilene --topo=/home/uceedv1/thesis/topologies/abilene.json --traffic=/home/uceedv1/thesis/results/traffic_abilene_α1.5_min30.json --simTime=30" 2>&1 | tee ~/ospf_1.5_min30.log

echo ""
echo "=========================================="
echo "Step 3: Run ECMP baselines"
echo "=========================================="

echo -e "\n[ECMP α=0.3]"
time ./ns3 run "scratch/abilene/abilene --topo=/home/uceedv1/thesis/topologies/abilene.json --traffic=/home/uceedv1/thesis/results/traffic_abilene_α0.3_min30.json --simTime=30 --ecmp=true" 2>&1 | tee ~/ecmp_0.3_min30.log

echo -e "\n[ECMP α=1.0]"
time ./ns3 run "scratch/abilene/abilene --topo=/home/uceedv1/thesis/topologies/abilene.json --traffic=/home/uceedv1/thesis/results/traffic_abilene_α1.0_min30.json --simTime=30 --ecmp=true" 2>&1 | tee ~/ecmp_1.0_min30.log

echo -e "\n[ECMP α=1.5]"
time ./ns3 run "scratch/abilene/abilene --topo=/home/uceedv1/thesis/topologies/abilene.json --traffic=/home/uceedv1/thesis/results/traffic_abilene_α1.5_min30.json --simTime=30 --ecmp=true" 2>&1 | tee ~/ecmp_1.5_min30.log

echo ""
echo "=========================================="
echo "Step 4: Summary of results (min 30 Mbps)"
echo "=========================================="
echo ""
for f in ~/ospf_*_min30.log ~/ecmp_*_min30.log; do
  name=$(basename $f .log)
  echo "=== $name ==="
  grep "Loaded.*flows" $f | head -1
  grep "Max link utilization" $f
  echo ""
done

echo "=========================================="
echo "✅ All tests completed!"
echo "=========================================="
echo "Results saved to:"
echo "  ~/ospf_0.3_min30.log, ~/ospf_1.0_min30.log, ~/ospf_1.5_min30.log"
echo "  ~/ecmp_0.3_min30.log, ~/ecmp_1.0_min30.log, ~/ecmp_1.5_min30.log"
