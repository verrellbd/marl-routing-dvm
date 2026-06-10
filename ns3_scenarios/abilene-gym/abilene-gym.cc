// Abilene routing with OSPF + link weight control via gym interface.
// Agent outputs link weight multipliers in action.json.
// OSPF recalculates paths based on new weights, runs simulation, writes utilization to state.json.

#include "json.hpp"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using json = nlohmann::json;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE("AbilenGym");

struct LinkSpec
{
    uint32_t src;
    uint32_t dst;
    double capacityMbps;
    double delayMs;
    Ptr<PointToPointNetDevice> dev0;
    Ptr<PointToPointNetDevice> dev1;
    Ipv4Address addr0;
    Ipv4Address addr1;
    uint32_t metric;  // OSPF metric (cost)
};

struct FlowSpec
{
    uint32_t src;
    uint32_t dst;
    double rateMbps;
    double startTime;
    double stopTime;
};

// Global state
std::vector<LinkSpec> g_links;
std::vector<FlowSpec> g_flows;
std::vector<std::string> g_node_names;
NodeContainer g_nodes;
uint32_t g_n_nodes = 0;
std::string g_action_file;
std::string g_state_file;
double g_sim_time = 60.0;
double g_step_time = 1.0;
uint32_t g_step_count = 0;
uint32_t g_episode_count = 0;
Ptr<FlowMonitor> g_flow_monitor = nullptr;

// Forward declarations
void apply_link_weights();
void write_state_json();
void step_callback();


void
write_state_json()
{
    // Measure link utilization
    double max_util = 0.0;
    std::vector<double> link_utils;

    for (const auto& link : g_links)
    {
        Ptr<Queue<Packet>> q0 = link.dev0->GetQueue();
        Ptr<Queue<Packet>> q1 = link.dev1->GetQueue();

        uint64_t bytes0 = q0->GetTotalReceivedBytes();
        uint64_t bytes1 = q1->GetTotalReceivedBytes();

        // Utilization = (bytes * 8 bits/byte) / (capacity_bps * time_s) * 100%
        double util0 = (bytes0 * 8.0) / (link.capacityMbps * 1e6 * g_step_time) * 100.0;
        double util1 = (bytes1 * 8.0) / (link.capacityMbps * 1e6 * g_step_time) * 100.0;

        link_utils.push_back(util0);
        max_util = std::max(max_util, util0);
        max_util = std::max(max_util, util1);
    }

    // Measure average flow path length (hop count) using FlowMonitor
    // This is a proxy for latency: longer paths = more hops = higher delay
    double avg_path_length = 0.0;
    uint32_t flow_count = 0;
    if (g_flow_monitor)
    {
        g_flow_monitor->CheckForLostPackets();
        std::map<FlowId, FlowMonitor::FlowStats> stats = g_flow_monitor->GetFlowStats();

        double total_path_length = 0.0;
        for (auto it = stats.begin(); it != stats.end(); ++it)
        {
            if (it->second.rxPackets > 0)
            {
                // Hops = number of times packet traversed a link
                // Average hops = total bytes transmitted / (bytes per packet * hops per packet)
                // Simpler: use the packet count to estimate average path length
                // Path length ≈ total_tx_bytes / total_rx_bytes (ratio of transmissions to receptions)
                double tx_bytes = it->second.txBytes;
                double rx_bytes = it->second.rxBytes;
                if (rx_bytes > 0 && tx_bytes > 0)
                {
                    double path_hops = (double)tx_bytes / rx_bytes;
                    total_path_length += path_hops;
                    flow_count++;
                }
            }
        }
        if (flow_count > 0)
        {
            avg_path_length = total_path_length / flow_count;
        }
    }

    // Write state JSON
    json state;
    state["episode"] = g_episode_count;
    state["step"] = g_step_count;
    state["max_link_utilization_pct"] = max_util;
    state["link_utilizations"] = link_utils;
    state["avg_path_length"] = avg_path_length;  // Proxy for latency
    state["flow_count"] = flow_count;

    std::ofstream state_out(g_state_file);
    state_out << state.dump(2) << std::endl;
    state_out.close();

    std::cout << "  State written: max_util=" << std::fixed << std::setprecision(2)
              << max_util << "%, avg_path_length=" << avg_path_length << " hops, flows=" << flow_count << "\n";
}

void
step_callback()
{
    g_step_count++;

    // Measure utilization and write state
    write_state_json();

    // Schedule next step if not done
    if (Simulator::Now().GetSeconds() < g_sim_time)
    {
        Simulator::Schedule(Seconds(g_step_time), &step_callback);
    }
    else
    {
        std::cout << "\n✅ Gym simulation complete!\n";
        Simulator::Stop();
    }
}

int
main(int argc, char* argv[])
{
    std::string topoPath = "topologies/abilene.json";
    std::string trafficPath = "results/traffic_abilene_α0.3_min30.json";
    std::string actionPath = "/tmp/ns3-gym-action.json";
    std::string statePath = "/tmp/ns3-gym-state.json";
    double simTime = 60.0;
    double stepTime = 1.0;

    CommandLine cmd;
    cmd.AddValue("topo", "Path to topology JSON", topoPath);
    cmd.AddValue("traffic", "Path to traffic JSON", trafficPath);
    cmd.AddValue("action", "Path to action JSON (input)", actionPath);
    cmd.AddValue("state", "Path to state JSON (output)", statePath);
    cmd.AddValue("simTime", "Simulation time (s)", simTime);
    cmd.AddValue("stepTime", "Time per gym step (s)", stepTime);
    cmd.Parse(argc, argv);

    g_action_file = actionPath;
    g_state_file = statePath;
    g_sim_time = simTime;
    g_step_time = stepTime;
    g_episode_count = 1;

    // --- Parse topology JSON ---
    std::ifstream f(topoPath);
    NS_ABORT_MSG_IF(!f.is_open(), "Cannot open topology: " << topoPath);
    json j;
    f >> j;

    g_n_nodes = j["nodes"].size();
    for (const auto& node : j["nodes"])
    {
        g_node_names.push_back(node["name"].get<std::string>());
    }

    std::cout << "Loaded " << j["name"].get<std::string>() << ": " << g_n_nodes << " nodes\n";

    // --- Read action.json to get link weights BEFORE creating links ---
    std::vector<double> link_weights;
    std::cout << "Reading action.json for link weights...\n";
    if (std::ifstream(actionPath).good())
    {
        std::ifstream action_in(actionPath);
        json action_data;
        try
        {
            action_in >> action_data;
            if (action_data.contains("link_weights"))
            {
                link_weights = action_data["link_weights"].get<std::vector<double>>();
                std::cout << "  Loaded " << link_weights.size() << " weights from action.json\n";
                if (link_weights.size() >= 3)
                {
                    std::cout << "  [Sample] weights[0-2] = " << link_weights[0] << ", "
                              << link_weights[1] << ", " << link_weights[2] << "\n";
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Warning: failed to parse action.json: " << e.what() << "\n";
        }
    }
    else
    {
        std::cout << "  (action.json not found, using uniform weights=1.0)\n";
    }

    // Build ns-3 topology ---
    g_nodes.Create(g_n_nodes);
    InternetStackHelper internet;
    internet.Install(g_nodes);

    // --- Build links with weight-adjusted capacity ---
    // Note: Topology has 15 undirected links, gym has 30 directed link weights
    // We apply weight[i] to forward direction (i) and weight[n+i] to reverse (i)
    std::vector<LinkSpec> links;
    size_t n_undirected = j["links"].size();
    for (size_t link_idx = 0; link_idx < n_undirected; ++link_idx)
    {
        const auto& l = j["links"][link_idx];
        LinkSpec link;
        link.src = l["src"].get<uint32_t>();
        link.dst = l["dst"].get<uint32_t>();
        link.capacityMbps = l["capacity"].get<double>();
        link.delayMs = l["delay"].get<double>();
        link.metric = 100;  // Base metric

        // For bidirectional links, average forward and reverse weights
        // weights[0..14] = direction A→B
        // weights[15..29] = direction B→A
        double weight_fwd = 1.0, weight_rev = 1.0;
        if (link_idx < link_weights.size())
        {
            weight_fwd = link_weights[link_idx];
        }
        if (link_idx + n_undirected < link_weights.size())
        {
            weight_rev = link_weights[link_idx + n_undirected];
        }
        // Average the two directions for the bidirectional link
        double weight = (weight_fwd + weight_rev) / 2.0;
        weight = std::max(0.1, std::min(10.0, weight));  // Clamp to [0.1, 10.0]

        // Effective capacity = original / weight
        // This inverts the weight so lower weights = higher capacity (preferred)
        double effective_capacity = link.capacityMbps / weight;

        if (link_idx < 3)
        {
            std::cerr << "  Link " << link_idx << " (" << link.src << "->" << link.dst
                      << "): weight=" << weight
                      << " orig_cap=" << link.capacityMbps << " Mbps"
                      << " → eff_cap=" << effective_capacity << " Mbps\n";
        }

        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate(uint64_t(effective_capacity * 1e6))));
        p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(uint64_t(link.delayMs))));

        NetDeviceContainer devs =
            p2p.Install(g_nodes.Get(link.src), g_nodes.Get(link.dst));
        link.dev0 = DynamicCast<PointToPointNetDevice>(devs.Get(0));
        link.dev1 = DynamicCast<PointToPointNetDevice>(devs.Get(1));

        links.push_back(link);
    }

    // --- Assign IP addresses ---
    Ipv4AddressHelper ipv4;
    for (size_t i = 0; i < links.size(); ++i)
    {
        std::ostringstream subnet;
        subnet << "10." << ((i >> 8) & 0xFF) << "." << (i & 0xFF) << ".0";
        ipv4.SetBase(subnet.str().c_str(), "255.255.255.252");

        NetDeviceContainer devs;
        devs.Add(links[i].dev0);
        devs.Add(links[i].dev1);
        Ipv4InterfaceContainer ifaces = ipv4.Assign(devs);

        links[i].addr0 = ifaces.GetAddress(0);
        links[i].addr1 = ifaces.GetAddress(1);
    }
    g_links = links;  // Copy links to global AFTER IP address assignment
    std::cout << "Link weights applied to topology. OSPF will route based on adjusted capacities.\n";

    // --- Enable OSPF (GlobalRouting) ---
    Ipv4GlobalRoutingHelper globalRouting;
    for (uint32_t i = 0; i < g_n_nodes; ++i)
    {
        Ptr<Ipv4> ipv4 = g_nodes.Get(i)->GetObject<Ipv4>();
        Ptr<Ipv4RoutingProtocol> existing = ipv4->GetRoutingProtocol();
        Ptr<Ipv4ListRouting> listRouting = DynamicCast<Ipv4ListRouting>(existing);

        if (!listRouting)
        {
            listRouting = CreateObject<Ipv4ListRouting>();
            ipv4->SetRoutingProtocol(listRouting);
        }

        // Add GlobalRouting (OSPF)
        Ptr<Ipv4GlobalRouting> globalRtg = CreateObject<Ipv4GlobalRouting>();
        listRouting->AddRoutingProtocol(globalRtg, 0);  // Priority 0 = highest
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    std::cout << "Using GlobalRouting (OSPF). Agent controls via link weights.\n";

    // Debug: Print some route entries to verify weights affected routing
    // (Print after PopulateRoutingTables but before simulation starts)
    if (true)  // Always print on init
    {
        std::cerr << "\n[DEBUG] Sample routing table on node 0:\n";
        Ptr<Ipv4> ipv4 = g_nodes.Get(0)->GetObject<Ipv4>();
        Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol();
        Ptr<Ipv4ListRouting> listRouting = DynamicCast<Ipv4ListRouting>(routing);

        if (listRouting)
        {
            for (uint32_t j = 0; j < listRouting->GetNRoutingProtocols(); ++j)
            {
                int16_t priority;
                Ptr<Ipv4RoutingProtocol> proto = listRouting->GetRoutingProtocol(j, priority);
                Ptr<Ipv4GlobalRouting> globalRtg = DynamicCast<Ipv4GlobalRouting>(proto);

                if (globalRtg)
                {
                    std::ostringstream ss;
                    Ptr<OutputStreamWrapper> osw = Create<OutputStreamWrapper>(&ss);
                    globalRtg->PrintRoutingTable(osw);
                    std::string table = ss.str();
                    // Print first 500 chars
                    if (table.size() > 500)
                        table = table.substr(0, 500);
                    std::cerr << table << "...\n";
                    break;
                }
            }
        }
    }

    // --- Parse traffic flows ---
    std::ifstream ft(trafficPath);
    NS_ABORT_MSG_IF(!ft.is_open(), "Cannot open traffic: " << trafficPath);
    json jt;
    ft >> jt;

    std::vector<FlowSpec> flowSpecs;
    for (const auto& f : jt["flows"])
    {
        flowSpecs.push_back({f["src"].get<uint32_t>(),
                             f["dst"].get<uint32_t>(),
                             f["rate_mbps"].get<double>(),
                             f["start"].get<double>(),
                             f["stop"].get<double>()});
    }
    g_flows = flowSpecs;

    std::cout << "Loaded " << flowSpecs.size() << " flows from " << trafficPath << "\n";

    // --- Create one sink per destination node ---
    std::set<uint32_t> dst_nodes;
    for (const auto& fs : flowSpecs)
    {
        dst_nodes.insert(fs.dst);
    }

    // Install one server per destination node
    for (uint32_t dst : dst_nodes)
    {
        UdpEchoServerHelper echoServer(9);
        ApplicationContainer serverApps = echoServer.Install(g_nodes.Get(dst));
        serverApps.Start(Seconds(0.0));
        serverApps.Stop(Seconds(g_sim_time + 1.0));
    }

    // --- Install OnOff client applications ---
    for (size_t k = 0; k < flowSpecs.size(); ++k)
    {
        const auto& fs = flowSpecs[k];

        // Find dst's first interface address
        Ipv4Address dstAddr;
        for (size_t i = 0; i < links.size(); ++i)
        {
            if (links[i].src == fs.dst)
            {
                dstAddr = links[i].addr0;
                break;
            }
            if (links[i].dst == fs.dst)
            {
                dstAddr = links[i].addr1;
                break;
            }
        }

        if (!dstAddr.IsInitialized())
        {
            std::cerr << "Warning: could not find address for node " << fs.dst << "\n";
            continue;
        }

        // OnOff client on src
        UdpEchoClientHelper echoClient(dstAddr, 9);
        echoClient.SetAttribute("MaxPackets", UintegerValue((uint32_t)(fs.rateMbps * 1e6 * (fs.stopTime - fs.startTime) / 8 / 1400)));
        echoClient.SetAttribute("Interval", TimeValue(Seconds(1400.0 * 8 / (fs.rateMbps * 1e6))));
        echoClient.SetAttribute("PacketSize", UintegerValue(1400));

        ApplicationContainer clientApps = echoClient.Install(g_nodes.Get(fs.src));
        clientApps.Start(Seconds(fs.startTime));
        clientApps.Stop(Seconds(fs.stopTime));
    }

    // --- Install FlowMonitor ---
    FlowMonitorHelper flowmon;
    g_flow_monitor = flowmon.InstallAll();

    // --- Run simulation ---
    std::cout << "Starting simulation...\n";
    Simulator::Schedule(Seconds(g_step_time), &step_callback);
    Simulator::Stop(Seconds(g_sim_time));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}
