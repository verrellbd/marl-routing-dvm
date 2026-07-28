// Per-flow routing validation for the GNN routing agent.
//
// Reads a routing JSON (each flow with an explicit node path) and installs the
// EXACT path per flow via static host-routes to a unique per-flow destination
// address. Uses one-way OnOff UDP (no echo) so utilization and PACKET LOSS are
// measured cleanly. Run once with --routing=ospf and once with --routing=gnn to
// compare the deployed baseline against the learned policy at high fidelity.
//
// Output (state JSON): max offered link utilization, total tx/rx packets, and
// overall packet-loss fraction.

#include "json.hpp"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

using json = nlohmann::json;
using namespace ns3;

struct LinkSpec
{
    uint32_t src, dst;
    double capacityMbps, delayMs;
    Ptr<PointToPointNetDevice> dev0, dev1;
    Ipv4Address addr0, addr1;
};

int
main(int argc, char* argv[])
{
    std::string topoPath = "topologies/abilene.json";
    std::string routingPath = "results/generalization/ns3_routing.json";
    std::string statePath = "/tmp/validate-state.json";
    std::string routing = "ospf";  // "ospf" | "gnn"
    double simTime = 20.0;
    double rateScale = 1.0;  // divide BOTH rates and capacities (preserves util & loss)

    CommandLine cmd;
    cmd.AddValue("topo", "Topology JSON", topoPath);
    cmd.AddValue("routing_file", "Routing JSON (per-flow paths)", routingPath);
    cmd.AddValue("routing", "Which path to install: ospf | gnn | ecmp", routing);
    cmd.AddValue("state", "Output state JSON", statePath);
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("rateScale", "Divide rates & capacities by this (speedup, util-preserving)", rateScale);
    cmd.Parse(argc, argv);

    // "ecmp" = ns-3's NATIVE Ipv4GlobalRouting with RandomEcmpRouting, i.e. the routing
    // layer picks among equal-cost next hops itself (per lookup, so per packet) instead of
    // us installing an explicit path. Must be set before the stack is installed, because
    // Ipv4GlobalRouting reads the attribute when it is constructed.
    const bool nativeEcmp = (routing == "ecmp");
    if (nativeEcmp)
    {
        Config::SetDefault("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(true));
        std::cout << "Validation run: routing=ecmp (ns-3 native RandomEcmpRouting)\n";
    }
    const std::string pathKey = (routing == "gnn") ? "gnn_path" : "ospf_path";
    if (!nativeEcmp)
        std::cout << "Validation run: routing=" << routing << " (key=" << pathKey << ")\n";

    // --- topology ---
    std::ifstream f(topoPath);
    NS_ABORT_MSG_IF(!f.is_open(), "Cannot open topology: " << topoPath);
    json j; f >> j;
    uint32_t nNodes = j["nodes"].size();
    NodeContainer nodes; nodes.Create(nNodes);
    InternetStackHelper internet; internet.Install(nodes);

    std::vector<LinkSpec> links;
    for (const auto& l : j["links"])
    {
        LinkSpec link;
        link.src = l["src"].get<uint32_t>();
        link.dst = l["dst"].get<uint32_t>();
        link.capacityMbps = l["capacity"].get<double>();
        link.delayMs = l["delay"].get<double>();

        link.capacityMbps /= rateScale;  // downscale capacity (util ratio preserved)
        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate",
            DataRateValue(DataRate(uint64_t(link.capacityMbps * 1e6))));
        // NanoSeconds (not MilliSeconds(uint64_t ...)) so sub-millisecond link delays
        // are NOT truncated to 0 — critical on geographically small topologies like
        // germany50 whose links are 0.13-1.26 ms.
        p2p.SetChannelAttribute("Delay", TimeValue(NanoSeconds(uint64_t(link.delayMs * 1e6))));
        NetDeviceContainer devs = p2p.Install(nodes.Get(link.src), nodes.Get(link.dst));
        link.dev0 = DynamicCast<PointToPointNetDevice>(devs.Get(0));
        link.dev1 = DynamicCast<PointToPointNetDevice>(devs.Get(1));
        links.push_back(link);
    }

    // --- IP addresses (one /30 subnet per link) ---
    Ipv4AddressHelper ipv4;
    for (size_t i = 0; i < links.size(); ++i)
    {
        std::ostringstream subnet;
        subnet << "10." << ((i >> 8) & 0xFF) << "." << (i & 0xFF) << ".0";
        ipv4.SetBase(subnet.str().c_str(), "255.255.255.252");
        NetDeviceContainer devs; devs.Add(links[i].dev0); devs.Add(links[i].dev1);
        Ipv4InterfaceContainer ifaces = ipv4.Assign(devs);
        links[i].addr0 = ifaces.GetAddress(0);
        links[i].addr1 = ifaces.GetAddress(1);
    }

    // --- directed adjacency: (a,b) -> (output iface on a, next-hop ip = b) ---
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, Ipv4Address>> dir;
    std::map<uint32_t, uint32_t> nodeIface;  // any local iface per node (for /32 alias)
    for (auto& lk : links)
    {
        Ptr<Ipv4> ia = nodes.Get(lk.src)->GetObject<Ipv4>();
        Ptr<Ipv4> ib = nodes.Get(lk.dst)->GetObject<Ipv4>();
        uint32_t ifA = ia->GetInterfaceForDevice(lk.dev0);
        uint32_t ifB = ib->GetInterfaceForDevice(lk.dev1);
        dir[{lk.src, lk.dst}] = {ifA, lk.addr1};
        dir[{lk.dst, lk.src}] = {ifB, lk.addr0};
        nodeIface[lk.src] = ifA;
        nodeIface[lk.dst] = ifB;
    }

    // --- read routing JSON ---
    std::ifstream fr(routingPath);
    NS_ABORT_MSG_IF(!fr.is_open(), "Cannot open routing: " << routingPath);
    json jr; fr >> jr;
    std::cout << "Loaded " << jr["flows"].size() << " flows from " << routingPath << "\n";

    Ipv4StaticRoutingHelper srh;
    uint16_t basePort = 1000;
    uint32_t flowIdx = 0;

    for (const auto& fl : jr["flows"])
    {
        uint32_t src = fl["src"].get<uint32_t>();
        uint32_t dst = fl["dst"].get<uint32_t>();
        double rate = fl["rate_mbps"].get<double>();
        double start = fl["start"].get<double>();
        double stop = fl["stop"].get<double>();
        std::vector<uint32_t> path = fl[pathKey].get<std::vector<uint32_t>>();
        if (path.size() < 2) { flowIdx++; continue; }

        Ipv4Address flowAddr;
        Ptr<Ipv4> idst = nodes.Get(dst)->GetObject<Ipv4>();
        if (nativeEcmp)
        {
            // Global routing only advertises an interface's PRIMARY address, so the
            // per-flow /32 aliases used below would never receive a route. Address the
            // sink at the destination's real interface address instead; flows stay
            // distinguishable to FlowMonitor by their unique port.
            flowAddr = idst->GetAddress(nodeIface[dst], 0).GetLocal();
        }
        else
        {
            // unique /32 destination address for this flow (integer-based, safe for
            // >255 flows: 10.200.0.1, 10.200.0.2, ... 10.200.1.0, ...).
            uint32_t flowAddrInt = ((10u << 24) | (200u << 16)) + flowIdx + 1;
            flowAddr = Ipv4Address(flowAddrInt);
            idst->AddAddress(nodeIface[dst],
                             Ipv4InterfaceAddress(flowAddr, Ipv4Mask("255.255.255.255")));

            // install static host-routes along the path
            for (size_t h = 0; h + 1 < path.size(); ++h)
            {
                auto key = std::make_pair(path[h], path[h + 1]);
                NS_ABORT_MSG_IF(dir.find(key) == dir.end(),
                                "No link " << path[h] << "->" << path[h + 1]);
                uint32_t oif = dir[key].first;
                Ipv4Address nh = dir[key].second;
                Ptr<Ipv4StaticRouting> sr =
                    srh.GetStaticRouting(nodes.Get(path[h])->GetObject<Ipv4>());
                sr->AddHostRouteTo(flowAddr, nh, oif);
            }
        }

        uint16_t port = basePort + (flowIdx % 60000);
        // sink
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(flowAddr, port));
        ApplicationContainer sapp = sink.Install(nodes.Get(dst));
        sapp.Start(Seconds(0.0)); sapp.Stop(Seconds(simTime + 1.0));
        // source (one-way constant-rate UDP)
        OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(flowAddr, port));
        onoff.SetConstantRate(DataRate(uint64_t(rate / rateScale * 1e6)), 1400);
        ApplicationContainer capp = onoff.Install(nodes.Get(src));
        capp.Start(Seconds(start)); capp.Stop(Seconds(stop));
        flowIdx++;
    }

    if (nativeEcmp)
    {
        // Build shortest-path routes for every destination; with RandomEcmpRouting set,
        // Ipv4GlobalRouting spreads traffic over equal-cost next hops itself.
        Ipv4GlobalRoutingHelper::PopulateRoutingTables();
        std::cout << "Populated global routing tables (ECMP enabled)\n";
    }

    FlowMonitorHelper fmHelper;
    Ptr<FlowMonitor> fm = fmHelper.InstallAll();

    std::cout << "Running " << simTime << "s...\n";
    Simulator::Stop(Seconds(simTime + 1.0));
    Simulator::Run();

    // --- per-link offered utilization (bytes enqueued / capacity / window) ---
    double window = simTime;  // approx; flows active [start, stop]
    double maxUtil = 0.0;
    std::vector<double> linkUtils;
    for (auto& lk : links)
    {
        uint64_t b0 = lk.dev0->GetQueue()->GetTotalReceivedBytes();
        uint64_t b1 = lk.dev1->GetQueue()->GetTotalReceivedBytes();
        double u0 = b0 * 8.0 / (lk.capacityMbps * 1e6 * window) * 100.0;
        double u1 = b1 * 8.0 / (lk.capacityMbps * 1e6 * window) * 100.0;
        linkUtils.push_back(u0); linkUtils.push_back(u1);
        maxUtil = std::max({maxUtil, u0, u1});
    }

    // --- QoS metrics via FlowMonitor: loss, delay, delivered throughput ---
    fm->CheckForLostPackets();
    auto stats = fm->GetFlowStats();
    uint64_t txP = 0, rxP = 0, rxBytes = 0;
    double delaySumS = 0.0;
    for (auto& kv : stats)
    {
        txP += kv.second.txPackets;
        rxP += kv.second.rxPackets;
        rxBytes += kv.second.rxBytes;
        delaySumS += kv.second.delaySum.GetSeconds();
    }
    double lossPct = (txP > 0) ? 100.0 * (double)(txP - rxP) / txP : 0.0;
    double meanDelayMs = (rxP > 0) ? 1000.0 * delaySumS / rxP : 0.0;
    double throughputMbps = rxBytes * 8.0 / (window * 1e6);

    json out;
    out["routing"] = routing;
    out["max_offered_util_pct"] = maxUtil;
    out["tx_packets"] = txP;
    out["rx_packets"] = rxP;
    out["lost_packets"] = (txP >= rxP) ? (txP - rxP) : 0;
    out["loss_pct"] = lossPct;
    out["mean_delay_ms"] = meanDelayMs;
    out["throughput_mbps"] = throughputMbps;
    out["link_utils"] = linkUtils;
    std::ofstream so(statePath);
    so << out.dump(2) << std::endl;

    std::cout << "\n=== " << routing << " ===\n"
              << "  max offered util: " << std::fixed << std::setprecision(1) << maxUtil << "%\n"
              << "  loss=" << std::setprecision(2) << lossPct << "%"
              << "  mean delay=" << meanDelayMs << " ms"
              << "  throughput=" << std::setprecision(0) << throughputMbps << " Mbps\n"
              << "  -> " << statePath << "\n";

    Simulator::Destroy();
    return 0;
}
