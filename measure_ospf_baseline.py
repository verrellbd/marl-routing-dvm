#!/usr/bin/env python3
"""
Measure OSPF baseline performance.
Pure routing with no RL agent - just standard OSPF.
This is the benchmark we compare GNN routing against.
"""
import json
from pathlib import Path

from marl_routing.routing_env import RoutingEnv


def measure_ospf_baseline(topo_name="abilene", n_episodes=3):
    """Measure OSPF performance with uniform link weights.

    Args:
        topo_name: Topology to test
        n_episodes: Number of episodes to average

    Returns:
        dict with baseline metrics
    """
    print("=" * 70)
    print(f"OSPF Baseline Measurement ({topo_name})")
    print("=" * 70)
    print(f"Running {n_episodes} episodes with uniform link weights (no agent)")

    env = RoutingEnv(topo_name=topo_name)

    results = {
        "topology": topo_name,
        "n_episodes": n_episodes,
        "episodes": [],
    }

    for episode in range(n_episodes):
        print(f"\n[Episode {episode+1}/{n_episodes}]")
        obs, info = env.reset()

        max_utils = []
        episode_reward = 0.0

        for step in range(60):
            # Use uniform weights (all 1.0) - no optimization
            action = env.action_space.sample() * 0 + 1.0  # All weights = 1.0
            obs, reward, terminated, truncated, info = env.step(action)

            max_util = info["max_util"]
            max_utils.append(max_util)
            episode_reward += reward

            if step % 10 == 0:
                print(f"  Step {step:2d}: max_util={max_util:6.2f}%")

            if terminated:
                break

        episode_data = {
            "episode": episode + 1,
            "max_util_final": max_utils[-1] if max_utils else 0.0,
            "max_util_mean": sum(max_utils) / len(max_utils) if max_utils else 0.0,
            "max_util_peak": max(max_utils) if max_utils else 0.0,
            "episode_reward": episode_reward,
            "steps": len(max_utils),
        }
        results["episodes"].append(episode_data)

        print(f"  Final max_util: {episode_data['max_util_final']:.2f}%")
        print(f"  Peak max_util: {episode_data['max_util_peak']:.2f}%")
        print(f"  Episode reward: {episode_reward:.2f}")

    # Compute aggregate statistics
    final_utils = [ep["max_util_final"] for ep in results["episodes"]]
    mean_utils = [ep["max_util_mean"] for ep in results["episodes"]]
    peak_utils = [ep["max_util_peak"] for ep in results["episodes"]]

    results["aggregate"] = {
        "final_util_mean": sum(final_utils) / len(final_utils),
        "final_util_std": (sum((x - sum(final_utils)/len(final_utils))**2 for x in final_utils) / len(final_utils))**0.5,
        "mean_util_mean": sum(mean_utils) / len(mean_utils),
        "peak_util_mean": sum(peak_utils) / len(peak_utils),
    }

    print("\n" + "=" * 70)
    print("OSPF BASELINE RESULTS")
    print("=" * 70)
    print(f"Final max utilization (mean): {results['aggregate']['final_util_mean']:.2f}%")
    print(f"Final max utilization (std):  {results['aggregate']['final_util_std']:.2f}%")
    print(f"Mean utilization during run:  {results['aggregate']['mean_util_mean']:.2f}%")
    print(f"Peak utilization observed:    {results['aggregate']['peak_util_mean']:.2f}%")
    print("=" * 70)

    env.close()
    return results


def main():
    # Measure OSPF baseline on Abilene
    ospf_results = measure_ospf_baseline("abilene", n_episodes=3)

    # Save results
    results_file = Path(__file__).parent / "results" / "ospf_baseline.json"
    with open(results_file, "w") as f:
        json.dump(ospf_results, f, indent=2)
    print(f"\n✅ Baseline saved to {results_file}")

    print("\n" + "=" * 70)
    print("WHAT THIS BASELINE MEANS")
    print("=" * 70)
    print("This is what GNN routing agent must beat (or match, or lose to).")
    print(f"GNN agent will be trained to minimize max utilization.")
    print(f"Success: GNN < {ospf_results['aggregate']['final_util_mean']:.2f}%")
    print(f"Equivalent: GNN ≈ {ospf_results['aggregate']['final_util_mean']:.2f}%")
    print(f"Failure: GNN > {ospf_results['aggregate']['final_util_mean']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
