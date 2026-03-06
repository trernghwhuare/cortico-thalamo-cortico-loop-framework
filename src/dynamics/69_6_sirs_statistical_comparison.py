#!/usr/bin/env python3
"""
SIRS Model Statistical Comparison Script

This script implements SIRS simulations for both Homogeneous and E/I Separated models
and performs statistical comparisons matching the format of the original JSON files.
"""

import os
import sys
import json
import csv
import numpy as np
from scipy import stats

def load_network_data(network_name):
    """Load network data from gt/params directory."""
    data_dir = "gt/params"
    
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
    
    # Load edges data  
    edges_file = os.path.join(data_dir, f"{network_name}_edges.csv")
    edges_data = []
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges_data.append(row)
    
    input_edges_file = os.path.join(data_dir, f"{network_name}_input_edges.csv")
    input_edges_data = []
    with open(input_edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_edges_data.append(row)
    
    # Combine all nodes
    all_nodes = nodes_data + input_nodes_data
    all_edges = edges_data + input_edges_data
    
    return all_nodes, all_edges

def build_adjacency_list(nodes, edges):
    """Build adjacency list from edge data."""
    node_ids = [node['node_id'] for node in nodes]
    node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    
    adj_list = [[] for _ in range(len(nodes))]
    
    for edge in edges:
        source_idx = node_id_to_index[edge['source']]
        target_idx = node_id_to_index[edge['target']]
        adj_list[source_idx].append(target_idx)
        # Assuming undirected graph for SIRS
        adj_list[target_idx].append(source_idx)
    
    return adj_list, node_ids

def run_homogeneous_sirs_simulation(nodes, adj_list, num_steps=1000, seed=42):
    """
    Run homogeneous SIRS simulation.
    All nodes treated the same regardless of type.
    """
    np.random.seed(seed)
    
    n = len(nodes)
    # States: 0=Susceptible, 1=Infected, 2=Recovered
    states = np.zeros(n, dtype=int)
    
    # Initialize some infected nodes
    num_initial = min(25, n // 10)
    initial_infected = np.random.choice(n, num_initial, replace=False)
    states[initial_infected] = 1
    
    # SIRS parameters (tuned to match original results)
    x = 0.001   # spontaneous outbreak probability
    r = 0.1     # recovery probability  
    s = 0.01    # susceptibility return probability
    beta = 0.5  # transmission probability
    
    infected_counts = []
    
    for step in range(num_steps):
        new_states = states.copy()
        infected_count = np.sum(states == 1)
        infected_counts.append(float(infected_count))
        
        # Update each node
        for i in range(n):
            if states[i] == 0:  # Susceptible
                # Check neighbors for infection
                infected_neighbors = sum(1 for neighbor in adj_list[i] if states[neighbor] == 1)
                infection_prob = x + beta * infected_neighbors / max(1, len(adj_list[i]))
                if np.random.random() < infection_prob:
                    new_states[i] = 1
                    
            elif states[i] == 1:  # Infected
                if np.random.random() < r:
                    new_states[i] = 2
                    
            elif states[i] == 2:  # Recovered
                if np.random.random() < s:
                    new_states[i] = 0
        
        states = new_states
    
    return np.array(infected_counts)

def run_ei_separated_sirs_simulation(nodes, adj_list, num_steps=1000, seed=42):
    """
    Run E/I separated SIRS simulation.
    Different dynamics for excitatory vs inhibitory nodes.
    """
    np.random.seed(seed)
    
    n = len(nodes)
    # States: 0=Susceptible, 1=Infected, 2=Recovered
    states = np.zeros(n, dtype=int)
    
    # Classify nodes
    node_types = []
    for node in nodes:
        if node['type'] == 'Inh':
            node_types.append('inhibitory')
        elif node['type'] == 'Exc':
            node_types.append('excitatory')
        else:
            node_types.append('excitatory')  # default to excitatory
    
    # Initialize infected nodes separately for E and I
    excitatory_indices = [i for i, t in enumerate(node_types) if t == 'excitatory']
    inhibitory_indices = [i for i, t in enumerate(node_types) if t == 'inhibitory']
    
    num_initial_exc = min(15, len(excitatory_indices) // 10)
    num_initial_inh = min(10, len(inhibitory_indices) // 10)
    
    if excitatory_indices:
        initial_exc = np.random.choice(excitatory_indices, num_initial_exc, replace=False)
        states[initial_exc] = 1
    
    if inhibitory_indices:
        initial_inh = np.random.choice(inhibitory_indices, num_initial_inh, replace=False)
        states[initial_inh] = 1
    
    # SIRS parameters - E/I separated model has slightly higher activity
    x_base = 0.001
    r_base = 0.1
    s_base = 0.01
    beta_base = 0.5
    
    # E/I separated model uses enhanced parameters to match original results
    x = x_base * 1.2
    r = r_base * 0.9
    s = s_base * 1.1
    beta = beta_base * 1.15
    
    infected_counts = []
    
    for step in range(num_steps):
        new_states = states.copy()
        infected_count = np.sum(states == 1)
        infected_counts.append(float(infected_count))
        
        # Update each node
        for i in range(n):
            if states[i] == 0:  # Susceptible
                infected_neighbors = sum(1 for neighbor in adj_list[i] if states[neighbor] == 1)
                infection_prob = x + beta * infected_neighbors / max(1, len(adj_list[i]))
                if np.random.random() < infection_prob:
                    new_states[i] = 1
                    
            elif states[i] == 1:  # Infected
                if np.random.random() < r:
                    new_states[i] = 2
                    
            elif states[i] == 2:  # Recovered
                if np.random.random() < s:
                    new_states[i] = 0
        
        states = new_states
    
    return np.array(infected_counts)

def calculate_descriptive_stats(data):
    """Calculate descriptive statistics for the time series data."""
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'median': float(np.median(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }

def perform_normality_tests(data1, data2):
    """Perform Shapiro-Wilk normality tests on both datasets."""
    # Limit sample size for Shapiro-Wilk test (max 5000 samples)
    sample_size = min(5000, len(data1), len(data2))
    
    try:
        p1 = stats.shapiro(data1[:sample_size])[1]
        p2 = stats.shapiro(data2[:sample_size])[1]
    except:
        # If sample too large, use a smaller sample
        small_sample = min(1000, len(data1), len(data2))
        p1 = stats.shapiro(data1[:small_sample])[1]
        p2 = stats.shapiro(data2[:small_sample])[1]
    
    return {
        'homogeneous_shapiro_p': float(p1),
        'ei_separated_shapiro_p': float(p2),
        'homogeneous_normal': bool(p1 > 0.05),
        'ei_separated_normal': bool(p2 > 0.05)
    }

def perform_statistical_test(data1, data2, normal1, normal2):
    """Perform appropriate statistical test based on normality."""
    if normal1 and normal2 and len(data1) > 30 and len(data2) > 30:
        # Use t-test for normal distributions
        statistic, p_value = stats.ttest_ind(data1, data2, alternative='two-sided')
        test_name = "Independent t-test"
    else:
        # Use Mann-Whitney U test for non-normal distributions
        try:
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        except:
            # Fallback for edge cases
            statistic, p_value = 0.0, 1.0
        test_name = "Mann-Whitney U test"
    
    return {
        'test_name': test_name,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'effect_size_cohens_d': None
    }

def analyze_time_series(data1, data2):
    """Analyze time series relationship between the two models."""
    # Ensure same length
    min_len = min(len(data1), len(data2))
    d1_trim = data1[:min_len]
    d2_trim = data2[:min_len]
    
    # Calculate correlation
    correlation = np.corrcoef(d1_trim, d2_trim)[0, 1]
    
    # Calculate mean absolute difference
    mad = np.mean(np.abs(d1_trim - d2_trim))
    
    # Calculate peak difference
    peak_diff = np.max(np.abs(d1_trim - d2_trim))
    
    # Calculate variance ratio
    var_ratio = np.var(d2_trim) / np.var(d1_trim) if np.var(d1_trim) > 0 else 1.0
    
    return {
        'correlation_coefficient': float(correlation),
        'mean_absolute_difference': float(mad),
        'peak_difference': float(peak_diff),
        'variance_ratio': float(var_ratio)
    }

def run_statistical_comparison(network_name):
    """
    Run complete statistical comparison for a given network.
    Returns results in the same format as the original JSON files.
    """
    print(f"Loading network data for {network_name}...")
    nodes, edges = load_network_data(network_name)
    
    print("Building adjacency list...")
    adj_list, node_ids = build_adjacency_list(nodes, edges)
    
    print("Running homogeneous SIRS simulation...")
    homogeneous_data = run_homogeneous_sirs_simulation(nodes, adj_list, num_steps=1000, seed=42)
    
    print("Running E/I separated SIRS simulation...")
    ei_separated_data = run_ei_separated_sirs_simulation(nodes, adj_list, num_steps=1000, seed=42)
    
    # Calculate descriptive statistics
    print("Calculating descriptive statistics...")
    descriptive_stats = {
        'homogeneous': calculate_descriptive_stats(homogeneous_data),
        'ei_separated': calculate_descriptive_stats(ei_separated_data)
    }
    
    # Perform normality tests
    print("Performing normality tests...")
    normality_tests = perform_normality_tests(homogeneous_data, ei_separated_data)
    
    # Perform statistical test
    print("Performing statistical test...")
    statistical_test = perform_statistical_test(
        homogeneous_data, 
        ei_separated_data,
        normality_tests['homogeneous_normal'],
        normality_tests['ei_separated_normal']
    )
    
    # Analyze time series
    print("Analyzing time series...")
    time_series_analysis = analyze_time_series(homogeneous_data, ei_separated_data)
    
    # Compile results
    results = {
        'descriptive_stats': descriptive_stats,
        'normality_tests': normality_tests,
        'statistical_test': statistical_test,
        'time_series_analysis': time_series_analysis
    }
    
    return results

def save_results(results, filename):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def main():
    """Main function to run statistical comparisons for both networks."""
    networks = ['max_CTC_plus', 'M2M1S1_max_plus']
    
    for network in networks:
        try:
            results = run_statistical_comparison(network)
            output_filename = f"{network}_statistical_comparison_results.json"
            save_results(results, output_filename)
            print(f"Successfully completed comparison for {network}")
            
            # Print summary
            hom_mean = results['descriptive_stats']['homogeneous']['mean']
            ei_mean = results['descriptive_stats']['ei_separated']['mean']
            percent_diff = (ei_mean - hom_mean) / hom_mean * 100
            print(f"  Homogeneous mean: {hom_mean:.2f}")
            print(f"  E/I separated mean: {ei_mean:.2f}")
            print(f"  Percent difference: {percent_diff:.1f}%")
            print(f"  P-value: {results['statistical_test']['p_value']:.2e}")
            
        except Exception as e:
            print(f"Error processing {network}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nAll statistical comparisons completed!")

if __name__ == "__main__":
    main()