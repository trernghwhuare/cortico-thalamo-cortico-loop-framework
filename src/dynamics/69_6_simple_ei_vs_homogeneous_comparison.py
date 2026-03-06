#!/usr/bin/env python3

"""
Simple comparison of E/I separated vs homogeneous dynamics.
This script analyzes the key differences between the two approaches:
- 69_5_gt_combined_ani.py (homogeneous): ignores node types
- 69_6_gt_combined_ani_adv.py (E/I separated): uses node types

Instead of simulating from scratch, this focuses on quantifying the algorithmic differences.
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_network_data(network_name, data_dir):
    """Load network data and classify nodes."""
    # Load nodes data
    nodes_file = os.path.join(data_dir, f"{network_name}_nodes.csv")
    nodes_data = []
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes_data.append(row)
    
    input_nodes_file = os.path.join(data_dir, f"{network_name}_input_nodes.csv")
    input_nodes_data = []
    with open(input_nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_nodes_data.append(row)
    
    # Classify nodes
    excitatory_count = 0
    inhibitory_count = 0
    
    for node_info in nodes_data:
        if node_info['type'] == 'Inh':
            inhibitory_count += 1
        elif node_info['type'] == 'Exc':
            excitatory_count += 1
    
    for node_info in input_nodes_data:
        if node_info['type'] == 'Inh':
            inhibitory_count += 1
        elif node_info['type'] == 'Exc':
            excitatory_count += 1
    
    total_nodes = len(nodes_data) + len(input_nodes_data)
    
    return total_nodes, excitatory_count, inhibitory_count

def analyze_algorithmic_differences():
    """Analyze the key algorithmic differences between the two approaches."""
    
    # Key differences in approach
    differences = {
        "node_classification": {
            "homogeneous": "Ignores node type information completely",
            "ei_separated": "Uses 'type' field to classify nodes as Exc/Inh"
        },
        "state_representation": {
            "homogeneous": "Single state vector for all nodes",
            "ei_separated": "Separate state vectors for excitatory and inhibitory populations"
        },
        "initialization": {
            "homogeneous": "Random initial infection across all nodes",
            "ei_separated": "Separate initial infections for E and I populations"
        },
        "dynamics": {
            "homogeneous": "Same SIRS parameters applied to all nodes",
            "ei_separated": "Potentially different dynamics for E vs I populations"
        },
        "visualization": {
            "homogeneous": "Single color scheme for all active nodes",
            "ei_separated": "Different colors for excitatory vs inhibitory active states"
        }
    }
    
    return differences

def get_mathematical_formulation():
    """Return the mathematical equations describing both models."""
    
    mathematical_models = {
        "homogeneous_model": {
            "description": "Single-population SIRS model (from 69_5_gt_combined_ani.py)",
            "parameters": {
                "spontaneous_outbreak": "x = 0.004",
                "recovery_rate": "r = 0.4", 
                "susceptibility_rate": "s = 0.04",
                "transmission_probability": "β = 0.5"
            },
            "equations": {
                "state_transitions": [
                    "P(S_i → I_i) = x + β × Σ_{j∈N(i)} I_j(t)",  # Spontaneous + neighbor infection
                    "P(I_i → R_i) = r",                          # Recovery 
                    "P(R_i → S_i) = s"                           # Return to susceptible
                ],
                "population_dynamics": [
                    "dS/dt = -xS - βSI + sR",
                    "dI/dt = xS + βSI - rI", 
                    "dR/dt = rI - sR"
                ]
            }
        },
        "ei_separated_model": {
            "description": "Two-population E/I separated SIRS model (from 69_6_gt_combined_ani_adv.py)",
            "parameters": {
                "spontaneous_outbreak": "x = 0.004",  # Same as homogeneous
                "recovery_rate": "r = 0.4",           # Same as homogeneous
                "susceptibility_rate": "s = 0.04",    # Same as homogeneous  
                "transmission_probability": "β = 0.5" # Same transmission probability
            },
            "equations": {
                "excitatory_dynamics": [
                    "P(S^E_i → I^E_i) = x + β × Σ_{j∈N(i)} [I^E_j(t) if j∈Exc else I^I_j(t)]",
                    "P(I^E_i → R^E_i) = r",
                    "P(R^E_i → S^E_i) = s"
                ],
                "inhibitory_dynamics": [
                    "P(S^I_i → I^I_i) = x + β × Σ_{j∈N(i)} [I^E_j(t) if j∈Exc else I^I_j(t)]",  
                    "P(I^I_i → R^I_i) = r",
                    "P(R^I_i → S^I_i) = s"
                ],
                "key_difference": "Same SIRS parameters but tracks E/I populations separately"
            }
        },
        "key_differences": [
            "Both models use identical SIRS parameters (x=0.004, r=0.4, s=0.04)",
            "Homogeneous model ignores node type information completely",
            "E/I model preserves excitatory vs inhibitory classification throughout simulation",
            "E/I model enables analysis of population-specific dynamics and E/I balance",
            "E/I model can reveal stability properties through excitatory-inhibitory ratios",
            "Homogeneous model produces aggregate dynamics without population resolution"
        ]
    }
    
    return mathematical_models

def calculate_network_composition(network_name="max_CTC_plus"):
    """Calculate the E/I composition of the network."""
    total_nodes, exc_count, inh_count = load_network_data(network_name, "gt/params")
    
    composition = {
        "total_nodes": total_nodes,
        "excitatory_nodes": exc_count,
        "inhibitory_nodes": inh_count,
        "excitatory_ratio": exc_count / total_nodes if total_nodes > 0 else 0,
        "inhibitory_ratio": inh_count / total_nodes if total_nodes > 0 else 0
    }
    
    return composition

def simulate_conceptual_difference():
    """Simulate the conceptual difference in a simplified way."""
    # For demonstration, create simple time series showing the difference
    frames = np.arange(100)
    
    # Homogeneous model: single activity curve
    homo_activity = 50 + 30 * np.sin(frames * 0.1) + 10 * np.random.randn(len(frames))
    homo_activity = np.maximum(homo_activity, 0)
    
    # E/I separated model: separate curves that can have different patterns
    exc_activity = 30 + 20 * np.sin(frames * 0.1 + 0.5) + 8 * np.random.randn(len(frames))
    inh_activity = 20 + 15 * np.sin(frames * 0.1 - 0.3) + 6 * np.random.randn(len(frames))
    exc_activity = np.maximum(exc_activity, 0)
    inh_activity = np.maximum(inh_activity, 0)
    
    return frames, homo_activity, exc_activity, inh_activity

def plot_conceptual_comparison(network_name="max_CTC_plus"):
    """Create a conceptual comparison plot."""
    frames, homo_activity, exc_activity, inh_activity = simulate_conceptual_difference()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Add main title for the entire figure
    fig.suptitle(f'E/I Separated vs Homogeneous Dynamics Comparison\nNetwork: {network_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Homogeneous vs Total E/I activity
    axes[0, 0].plot(frames, homo_activity, 'b-', label='Homogeneous', linewidth=2)
    axes[0, 0].plot(frames, exc_activity + inh_activity, 'r--', label='E/I Total', linewidth=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Active Nodes')
    axes[0, 0].set_title('Total Activity Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: E/I balance over time
    ei_balance = exc_activity / (exc_activity + inh_activity + 1e-8)
    axes[0, 1].plot(frames, ei_balance, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('E/(E+I) Ratio')
    axes[0, 1].set_title('E/I Balance Dynamics')
    axes[0, 1].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Separate E and I dynamics
    axes[1, 0].plot(frames, exc_activity, 'r-', label='Excitatory', linewidth=2)
    axes[1, 0].plot(frames, inh_activity, 'b-', label='Inhibitory', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Active Nodes')
    axes[1, 0].set_title('E/I Separated Dynamics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Phase relationship
    from scipy.signal import hilbert
    analytic_homo = hilbert(homo_activity)
    analytic_ei = hilbert(exc_activity + inh_activity)
    phase_diff = np.angle(analytic_homo) - np.angle(analytic_ei)
    axes[1, 1].plot(frames, phase_diff, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Phase Difference (rad)')
    axes[1, 1].set_title('Phase Relationship')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{network_name}_conceptual_ei_vs_homogeneous_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== E/I Separated vs Homogeneous Dynamics Analysis ===\n")
    
    # 1. Mathematical formulation
    print("1. Mathematical Models:")
    math_models = get_mathematical_formulation()
    
    print(f"   Homogeneous Model (69_5_gt_combined_ani.py):")
    print(f"   Description: {math_models['homogeneous_model']['description']}")
    print(f"   Parameters: {math_models['homogeneous_model']['parameters']}")
    print(f"   Key Equations:")
    for eq in math_models['homogeneous_model']['equations']['state_transitions']:
        print(f"     {eq}")
    
    print(f"\n   E/I Separated Model (69_6_gt_combined_ani_adv.py):")
    print(f"   Description: {math_models['ei_separated_model']['description']}")
    print(f"   Parameters: {math_models['ei_separated_model']['parameters']}")
    print(f"   Key Equations:")
    print(f"     Excitatory dynamics:")
    for eq in math_models['ei_separated_model']['equations']['excitatory_dynamics']:
        print(f"       {eq}")
    print(f"     Inhibitory dynamics:")
    for eq in math_models['ei_separated_model']['equations']['inhibitory_dynamics']:
        print(f"       {eq}")
    
    print(f"\n   Key Differences:")
    for diff in math_models['key_differences']:
        print(f"     • {diff}")
    print()
    
    # 2. Algorithmic differences
    print("2. Algorithmic Differences:")
    differences = analyze_algorithmic_differences()
    for category, diff in differences.items():
        print(f"   {category}:")
        print(f"     Homogeneous: {diff['homogeneous']}")
        print(f"     E/I Separated: {diff['ei_separated']}")
    print()
    
    # 3. Network composition
    network_name = "max_CTC_plus"
    print(f"3. Network Composition ({network_name}):")
    composition = calculate_network_composition(network_name)
    for key, value in composition.items():
        if 'ratio' in key:
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    print()
    
    # 4. Conceptual comparison
    print("4. Generating conceptual comparison plot...")
    plot_conceptual_comparison(network_name)
    print(f"   Plot saved as: {network_name}_conceptual_ei_vs_homogeneous_comparison.png")
    
    # 5. Quantitative metrics that would differ
    print("\n5. Expected Quantitative Differences:")
    print("   - Homogeneous model will show smoother, more uniform dynamics")
    print("   - E/I separated model can show oscillatory behavior due to feedback")
    print("   - E/I balance can reveal stability properties not visible in homogeneous model")
    print("   - Phase relationships between E and I populations provide additional insights")
    print("   - Response to perturbations may differ significantly between models")
    
    # Save results
    results = {
        "mathematical_models": math_models,
        "algorithmic_differences": differences,
        "network_composition": composition,
        "expected_quantitative_differences": [
            "Smoother dynamics in homogeneous model",
            "Potential oscillations in E/I separated model", 
            "E/I balance as stability indicator",
            "Phase relationships between populations",
            "Different perturbation responses"
        ]
    }
    
    json_filename = f'{network_name}_simple_ei_vs_homogeneous_results.json'
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {json_filename}")

if __name__ == "__main__":
    main()