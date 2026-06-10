# ns-3 scenarios

Custom ns-3 scenarios for this project. The `ns-3-dev/` tree (3.7 GB) is git-ignored,
so these source files are mirrored here for version control.

To build/run, copy each into the ns-3 tree (also needs json.hpp = nlohmann/json):

    cp -r ns3_scenarios/abilene-gym       ns-3-dev/scratch/
    cp -r ns3_scenarios/abilene-validate  ns-3-dev/scratch/
    cd ns-3-dev && ./ns3 build abilene-validate -j8

- **abilene-gym** — interactive gym bridge (OSPF + link-weight control via action/state JSON).
- **abilene-validate** — installs exact per-flow paths (static host-routes), measures
  utilization + packet loss; validates the trained GNN routing vs OSPF.
