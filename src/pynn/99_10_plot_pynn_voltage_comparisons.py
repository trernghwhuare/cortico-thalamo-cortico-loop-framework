#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot population membrane voltage comparison plots for PyNN simulation results across different stimulation conditions.

This script compares:
- Basic (control) vs Pain stimulation
- Basic (control) vs TMS stimulation  
- Basic (control) vs Visual stimulation

For both excitatory and inhibitory neuron populations across COBA and CUBA benchmarks.

Note: Voltage plots show:
- X-axis: Time (ms)
- Y-axis: Membrane Potential (mV)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
from scipy import stats


def find_data_directory(base_dir):
    """Find any subdirectory in the base directory that contains .pkl files."""
    if not os.path.exists(base_dir):
        return None
    
    # Look for any subdirectory that contains .pkl files
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains .pkl files
            pkl_files = [f for f in os.listdir(item_path) if f.endswith('.pkl')]
            if pkl_files:
                return item_path
    
    return None


def load_pynn_data(filepath):
    """Load PyNN simulation data from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def identify_network_and_type(filepath):
    """Extract network name, neuron type, and benchmark from filename."""
    filename = os.path.basename(filepath)
    if filename.endswith('.pkl'):
        filename = filename[:-4]
    
    parts = filename.split('_')
    if len(parts) < 5:
        return None, None, None
    
    # Network name is everything except the last 4 components
    network_parts_count = len(parts) - 4
    network_name = '_'.join(parts[:network_parts_count])
    benchmark = parts[-4]     # COBA or CUBA
    neuron_type_raw = parts[-3]  # exc or inh
    
    neuron_type = 'exc' if neuron_type_raw == 'exc' else 'inh'
    return network_name, neuron_type, benchmark


def extract_voltage_traces(data):
    """Extract voltage traces from Neo data structure."""
    voltage_traces = []
    time_points = None
    
    if hasattr(data, 'segments') and len(data.segments) > 0:
        segment = data.segments[0]
        if hasattr(segment, 'analogsignals') and len(segment.analogsignals) > 0:
            for signal in segment.analogsignals:
                if len(signal) > 0:
                    # Convert quantities to regular arrays
                    voltage_values = np.array(signal)  # Shape: (time_points, neurons)
                    if time_points is None:
                        time_points = np.array(signal.times)  # Time points in ms
                    
                    # Store voltage traces for each recorded neuron
                    for neuron_idx in range(voltage_values.shape[1]):
                        voltage_traces.append(voltage_values[:, neuron_idx])
    
    return voltage_traces, time_points


def calculate_voltage_statistics(voltage_traces):
    """Calculate mean and std of voltage traces across time points."""
    if not voltage_traces:
        return None, None, None
    
    # Stack all traces to get shape (n_traces, n_timepoints)
    traces_array = np.vstack(voltage_traces)
    
    # Calculate mean and std across traces for each time point
    mean_trace = np.mean(traces_array, axis=0)
    std_trace = np.std(traces_array, axis=0)
    
    # Calculate overall statistics
    overall_mean = np.mean(traces_array)
    overall_std = np.std(traces_array)
    
    return mean_trace, std_trace, overall_mean, overall_std


def calculate_voltage_comparison_stats(voltage_traces1, voltage_traces2):
    """Calculate statistical comparison between two sets of voltage traces."""
    if not voltage_traces1 or not voltage_traces2:
        return None, None, None, None, None, None, None
    
    # Get overall statistics
    traces1_array = np.vstack(voltage_traces1)
    traces2_array = np.vstack(voltage_traces2)
    
    overall_mean1 = np.mean(traces1_array)
    overall_std1 = np.std(traces1_array)
    overall_mean2 = np.mean(traces2_array)
    overall_std2 = np.std(traces2_array)
    
    # Flatten all values for statistical testing
    all_values1 = traces1_array.flatten()
    all_values2 = traces2_array.flatten()
    
    # Perform statistical test
    if len(all_values1) > 5000 or len(all_values2) > 5000:
        # For very large samples, use t-test
        _, p_value = stats.ttest_ind(all_values1, all_values2, equal_var=False)
    else:
        # For smaller samples, check normality first
        try:
            _, p_norm1 = stats.shapiro(all_values1[:5000])
            _, p_norm2 = stats.shapiro(all_values2[:5000])
            
            if p_norm1 > 0.05 and p_norm2 > 0.05:
                # Both normally distributed, use t-test
                _, p_value = stats.ttest_ind(all_values1, all_values2, equal_var=False)
            else:
                # Not normally distributed, use Mann-Whitney U test
                _, p_value = stats.mannwhitneyu(all_values1, all_values2, alternative='two-sided')
        except:
            # Fallback to Mann-Whitney U test if Shapiro fails
            _, p_value = stats.mannwhitneyu(all_values1, all_values2, alternative='two-sided')
    
    return (overall_mean1, overall_std1, overall_mean2, overall_std2, p_value,
            np.mean(traces1_array, axis=0), np.mean(traces2_array, axis=0))


def format_p_value(p_value):
    """Format p-value for display."""
    if p_value < 0.001:
        return "p < 0.001"
    elif p_value < 0.01:
        return f"p = {p_value:.3f}"
    elif p_value < 0.05:
        return f"p = {p_value:.2f}"
    else:
        return f" p = {p_value:.2f}"


def load_all_conditions_voltage_data():
    """Load voltage data from all four stimulation conditions."""
    base_dirs = {
        'basic': 'PyNN_results',
        'pain': 'PyNN_pain_results', 
        'TMS': 'PyNN_TMS_results',
        'visual': 'PyNN_visual_results'
    }
    
    # Data structure: network -> benchmark -> neuron_type -> condition -> voltage_traces
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    time_data = {}  # Store time points for each network/benchmark/neuron_type
    
    for condition, base_dir in base_dirs.items():
        data_dir = find_data_directory(base_dir)
        if data_dir is None:
            print(f"Warning: No data directory found in {base_dir}, skipping {condition}")
            continue
            
        print(f"Loading voltage data from {data_dir} for {condition} condition")
            
        pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        
        for pkl_file in pkl_files:
            filepath = os.path.join(data_dir, pkl_file)
            network_name, neuron_type, benchmark = identify_network_and_type(filepath)
            
            if network_name is None or benchmark is None:
                continue
                
            data = load_pynn_data(filepath)
            voltage_traces, time_points = extract_voltage_traces(data)
            
            if voltage_traces:  # Only store if we have data
                all_data[network_name][benchmark][neuron_type][condition].extend(voltage_traces)
                
                # Store time points (should be same across conditions)
                key = f"{network_name}_{benchmark}_{neuron_type}"
                if key not in time_data and time_points is not None:
                    time_data[key] = time_points
    
    return all_data, time_data


def plot_voltage_comparison(time_points, voltage_traces1, voltage_traces2, 
                          condition1, condition2, title_suffix,
                          network_name, benchmark, neuron_type,
                          output_dir, is_combined_plot=False, ax=None):
    """Plot voltage comparison between two conditions."""
    if not voltage_traces1 or not voltage_traces2:
        return False
    
    # Calculate statistics
    stats_result = calculate_voltage_comparison_stats(voltage_traces1, voltage_traces2)
    if stats_result[0] is None:
        return False
    
    (mean1, std1, mean2, std2, p_value, mean_trace1, mean_trace2) = stats_result
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    # Plot mean traces with shaded std regions
    ax.plot(time_points, mean_trace1, color='#2E86AB', linewidth=2.5, label=f'{condition1.capitalize()}')
    ax.fill_between(time_points, mean_trace1 - std1, mean_trace1 + std1, 
                   color='#2E86AB', alpha=0.3)
    
    ax.plot(time_points, mean_trace2, color='#A23B72', linewidth=2.5, label=f'{condition2.capitalize()}')
    ax.fill_between(time_points, mean_trace2 - std2, mean_trace2 + std2, 
                   color='#A23B72', alpha=0.3)
    
    # Add p-value text
    p_text = format_p_value(p_value)
    significance_text = f"{p_text}"
    if p_value < 0.05:
        significance_text += " *"
    
    ax.text(0.95, 0.95, significance_text, transform=ax.transAxes, 
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
           fontsize=10, fontweight='bold' if p_value < 0.05 else 'normal')
    
    # Set labels and title
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    if is_combined_plot:
        ax.set_title(f'{network_name} - {benchmark} - {neuron_type.upper()}\n{title_suffix}')
    else:
        ax.set_title(f'{network_name} - {benchmark} - {neuron_type.upper()} Neurons\n{title_suffix}')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save individual plot if not in combined mode
    if not is_combined_plot:
        filename = f"{network_name}_{benchmark}_{neuron_type}_{condition1}_vs_{condition2}_voltage.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved individual voltage plot: {filepath}")
        return True
    else:
        return True


def plot_all_voltage_comparisons_in_one_figure(all_data, time_data, output_dir='voltage_comparison_plots'):
    """Create a single figure with all voltage comparison plots arranged in rows and columns."""
    os.makedirs(output_dir, exist_ok=True)
    
    comparisons = [
        ('basic', 'pain', 'Basic vs Pain'),
        ('basic', 'TMS', 'Basic vs TMS'),
        ('basic', 'visual', 'Basic vs Visual')
    ]
    
    # Get all available networks, benchmarks, and neuron types
    networks = list(all_data.keys())
    benchmarks = []
    neuron_types = ['exc', 'inh']
    
    # Collect all unique benchmarks
    for network in networks:
        benchmarks.extend(list(all_data[network].keys()))
    benchmarks = list(set(benchmarks))
    
    # Filter to only include combinations that have complete data
    valid_combinations = []
    for network in networks:
        for benchmark in benchmarks:
            for neuron_type in neuron_types:
                if benchmark in all_data[network] and neuron_type in all_data[network][benchmark]:
                    condition_data = all_data[network][benchmark][neuron_type]
                    valid_comparisons = []
                    for cond1, cond2, title_suffix in comparisons:
                        if cond1 in condition_data and cond2 in condition_data:
                            if len(condition_data[cond1]) > 0 and len(condition_data[cond2]) > 0:
                                valid_comparisons.append((cond1, cond2, title_suffix))
                    
                    if valid_comparisons:
                        valid_combinations.append((network, benchmark, neuron_type, valid_comparisons))
    
    # Calculate subplot dimensions
    total_plots = sum(len(comparisons_list) for _, _, _, comparisons_list in valid_combinations)
    if total_plots == 0:
        print("No voltage plots to create!")
        return
    
    # Use dynamic grid size based on number of plots
    cols = min(3, total_plots)  # Max 3 columns
    rows = (total_plots + cols - 1) // cols  # Ceiling division
    
    # Create the main figure
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if total_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    for network, benchmark, neuron_type, valid_comparisons in valid_combinations:
        condition_data = all_data[network][benchmark][neuron_type]
        time_key = f"{network}_{benchmark}_{neuron_type}"
        time_points = time_data.get(time_key)
        
        if time_points is None:
            continue
            
        for cond1, cond2, title_suffix in valid_comparisons:
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            voltage_traces1 = condition_data[cond1]
            voltage_traces2 = condition_data[cond2]
            
            success = plot_voltage_comparison(
                time_points, voltage_traces1, voltage_traces2,
                cond1, cond2, title_suffix,
                network, benchmark, neuron_type,
                output_dir, is_combined_plot=True, ax=ax
            )
            
            if success:
                plot_idx += 1
    
    # Hide any unused subplots
    for remaining_idx in range(plot_idx, len(axes)):
        axes[remaining_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the combined plot
    filepath = os.path.join(output_dir, 'all_voltage_comparisons_combined.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined voltage plot: {filepath}")
    
    # Also save individual plots
    for network, benchmark, neuron_type, valid_comparisons in valid_combinations:
        condition_data = all_data[network][benchmark][neuron_type]
        time_key = f"{network}_{benchmark}_{neuron_type}"
        time_points = time_data.get(time_key)
        
        if time_points is None:
            continue
            
        for cond1, cond2, title_suffix in valid_comparisons:
            voltage_traces1 = condition_data[cond1]
            voltage_traces2 = condition_data[cond2]
            
            plot_voltage_comparison(
                time_points, voltage_traces1, voltage_traces2,
                cond1, cond2, title_suffix,
                network, benchmark, neuron_type,
                output_dir, is_combined_plot=False
            )


def main():
    """Main function to run the voltage comparison plotting."""
    print("Loading PyNN simulation voltage data from all conditions...")
    all_data, time_data = load_all_conditions_voltage_data()
    
    # Check if we have any data
    if not all_data:
        print("No voltage data found in any of the condition directories!")
        return
    
    print("Creating voltage comparison plots with statistical analysis...")
    plot_all_voltage_comparisons_in_one_figure(all_data, time_data)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot voltage comparisons for PyNN simulation results")
    parser.add_argument("--output-dir", default="voltage_comparison_plots", 
                       help="Output directory for plots (default: voltage_comparison_plots)")
    args = parser.parse_args()
    
    main()