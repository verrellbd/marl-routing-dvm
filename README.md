# Improving Quality of Service in Network Routing with Graph Neural Network-Based Multi-Agent Reinforcement Learning

One routing policy, trained on seventeen backbone topologies, evaluated **zero-shot** on
three it has never seen — with real measured operator traffic and packet-level simulation
in ns-3.

Each router is an agent. It observes only the load on its own links and the destination of
the demand passing through it, then chooses the next hop. Training is centralized (a critic
reads global state); execution is not (the critic is discarded, and each node acts on local
information alone). The policy carries no node identity and no coordinates, so one parameter
set serves a network of any size.

## What is compared

| Arm | Description |
|-----|-------------|
| **OSPF** | Shortest path under the standard cost metric `refBW / capacity` |
| **ECMP** | True equal-**cost** multipath, one path per flow by header hash |
| **Single-agent GNN** | Centralized controller choosing one of `k=3` candidate paths per demand (PPO, 100,740 parameters) |
| **MARL** | Per-node agents forwarding hop by hop (MAPPO + GNN, 17,570 parameters) |

Both learned arms are trained on an identical budget — 1.5 × 10⁶ environment steps, 732
policy updates, 10 seeds — with every optimizer hyperparameter held equal. The centralized
baseline keeps Stable-Baselines3 defaults so it is not handicapped.

## Headline results

Packet-level, from 558 ns-3 simulations. Mean over ten seeds.

**Overload regime** (OSPF drives a link to or past capacity):

| Topology | Metric | OSPF | ECMP | Single-agent | MARL |
|---|---|---|---|---|---|
| GÉANT | loss | 14.13 % | 7.52 % | 11.23 % | **3.36 %** |
| Germany50 | loss | 27.29 % | 20.68 % | 14.94 % | **10.63 %** |
| Abilene | loss | 14.35 % | **14.31 %** | 17.81 % | 14.90 % |

**Abilene is a negative result and is reported as one.** It is the only evaluation network
with heterogeneous link capacity (one 2.48 Gb/s link among fourteen at 9.92 Gb/s), all
seventeen training topologies have uniform capacity, and the node features encode congestion
as a percentage rather than an absolute rate — so the policy has no training signal for
capacity heterogeneity. A correctly costed OSPF simply routes around the slow link.

ECMP is a strong baseline, not a straw man: it uses equal-**cost** paths and beats OSPF in
every overload cell.

## Layout

```
marl_routing/          the library
  topo_agnostic_marl_env.py   per-node hop-by-hop environment (the reward lives here)
  marl_gnn.py                 GNN actor-critic + MAPPO
  graph_routing_env.py        single-agent path-selection environment
  ospf_metric.py              OSPF cost, capacity-aware shortest paths
  real_traffic.py             measured SNDlib matrices
  tmgen_traffic.py            modulated-gravity synthetic matrices
  topology.py, visualize.py   loaders and rendering

train_marl_gnn_tier2.py    train the decentralized policy
train_single_tier2.py      train the centralized baseline
train_seeds10.sh           both arms, ten seeds

export_topoagn_marl_routes.py / export_topoagn_routes.py / export_ecmp_routes.py
                           roll a policy out and write per-flow routes for ns-3
run_ns3_final.sh           export + simulate (seeds 0-2)
eval_seeds10.sh            export + simulate, seeds selectable via $SEEDS

rebuild_ns3_grid.py        -> results/final_ns3_grid.json   (packet-level table)
fill_offered_grid.py       -> results/offered_grid.json      (analytical table)
decompose_reward.py        -> results/reward_decomp.json     (return decomposition)
eval_greedy.py             -> greedy best-response reference
select_width.py            -> hidden-width selection, on training topologies only
validate_surrogate.py      -> results/surrogate_validation.json
significance_test.py       -> Wilcoxon signed-rank, paired on the traffic matrix
make_figures.py, make_heatmap.py

ns3_scenarios/             ns-3 scenario sources (see its README for installation)
topologies/                SNDlib topologies and demand matrices
results/                   policies, per-simulation output, aggregated grids
logs/                      training logs behind the convergence figure
```

## Installing

Requires Python 3.9 and a working ns-3 build. The learned policies train and evaluate on
**CPU only**; no GPU is needed.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # exact pinned versions
```

Key versions: PyTorch 2.6.0, PyTorch Geometric 2.6.1, Stable-Baselines3 2.7.1, NetworkX,
ns-3.42.

For the packet-level half you also need ns-3 and ns3-ai:

```bash
# ns3-ai, pinned to the commit this project was built against
git clone https://github.com/hust-diangroup/ns3-ai.git <ns-3-dev>/contrib/ai
cd <ns-3-dev>/contrib/ai && git checkout b8c9858

# two of its examples do not compile against ns-3.42; this disables them
git apply <repo>/ns3ai_local_changes.patch

pip install -e <ns-3-dev>/contrib/ai/python_utils/
pip install -e <ns-3-dev>/contrib/ai/model/gym-interface/py/
./configure_ns3.sh                        # custom flags for pybind11, protobuf, venv python
cd <ns-3-dev> && ./ns3 build -j8
```

`PYTHONPATH` must include `<ns-3-dev>/contrib/ai/model/gym-interface/py`. A bare
`./ns3 configure` will not work — use `configure_ns3.sh`.

The scripts locate the repository from their own path, so a clone works anywhere. Two
environment variables override the defaults: `NS3_DIR` (default `<repo>/ns-3-dev`) and
`CONDA_PREFIX_DIR` (default `/opt/anaconda3`, where `configure_ns3.sh` looks for protobuf).

Copy the sources in `ns3_scenarios/` into `<ns-3-dev>/scratch/` before building.

### Upstream versions

- **ns-3**: official ns-3.42 release, `gitlab.com/nsnam/ns-3-dev`, commit `ab4cce021`, unmodified.
- **ns3-ai**: `github.com/hust-diangroup/ns3-ai`, commit `b8c9858` (main branch, *not* the
  v1.2.0 tag). One local change — `examples/CMakeLists.txt` disables the `rate-control` and
  `multi-bss` examples, which have drifted from the ns-3.42 API. Saved as
  `ns3ai_local_changes.patch`.

## Reproducing

```bash
# 1. train both arms, ten seeds each (CPU; cap threads when running in parallel)
#    SEEDS defaults to 3-9, so name all ten explicitly for a full reproduction
OMP_NUM_THREADS=1 SEEDS="0 1 2 3 4 5 6 7 8 9" ./train_seeds10.sh

# 2. roll the policies out and simulate the grid in ns-3
SEEDS="0 1 2 3 4 5 6 7 8 9" ./eval_seeds10.sh

# 3. aggregate
python rebuild_ns3_grid.py
python fill_offered_grid.py
python decompose_reward.py
python validate_surrogate.py

# 4. figures
python make_figures.py --out figures
python make_heatmap.py --out figures
```

Steps 3 and 4 run in seconds from the committed `results/`, so the tables and figures can
be regenerated without repeating the training or the simulations.

## Notes on the evaluation

- **Training uses an analytical surrogate**; every reported loss, delay and utilization
  figure is ns-3 packet-level. ns-3 is far too slow to sit inside a training loop.
- **Analytical values above 100 % are *offered load*, not utilization.** ns-3 utilization
  saturates at 100 % and turns the excess into loss. `validate_surrogate.py` quantifies the
  agreement between the two: where nothing is dropped the surrogate reproduces the measured
  offered load (ρ = 0.89), and under overload it orders measured packet loss correctly
  (ρ = 0.77).
- **Learned routes are source-specific** and are installed as per-flow static routes.
  Deploying them would require MPLS or segment routing; plain destination-based IP
  forwarding cannot express them. OSPF and ECMP are restricted to destination-based
  forwarding, so part of the measured difference is attributable to the forwarding paradigm.
- Each simulation carries **one static matrix of constant-rate UDP traffic**. Adaptation
  within an episode and interaction with transport-layer congestion control are untested.

## Data

Topologies and measured traffic come from [SNDlib](http://sndlib.zib.de). Synthetic training
matrices are generated with the modulated-gravity model from
[TMgen](https://github.com/progwriter/TMgen). Raw upstream archives are not redistributed
here; `sndlib_to_json.py` converts SNDlib native files into the JSON format this code reads.

## Licence

Code in this repository is released under the MIT Licence. Third-party datasets remain
under their original terms.
