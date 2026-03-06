#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot KDE comparison plots for PyNN simulation results across different stimulation conditions.

This script compares:
- Basic (control) vs Pain stimulation
- Basic (control) vs TMS stimulation  
- Basic (control) vs Visual stimulation

For both excitatory and inhibitory neuron populations.

Note: Standard KDE plot orientation with:
- X-axis: Firing Rate (Hz) ranging from -70 to 70
- Y-axis: Density
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


def extract_firing_rates(data, duration_ms=1000):
    """Extract firing rates from Neo data structure."""
    firing_rates = []
    
    if hasattr(data, 'segments') and len(data.segments) > 0:
        segment = data.segments[0]
        if hasattr(segment, 'spiketrains') and len(segment.spiketrains) > 0:
            for spiketrain in segment.spiketrains:
                num_spikes = len(spiketrain)
                firing_rate = (num_spikes / duration_ms) * 1000  # Convert to Hz
                firing_rates.append(firing_rate)
    
    return firing_rates


def calculate_statistical_comparison(data1, data2):
    """
    Calculate statistical comparison between two datasets.
    Returns mean1, std1, mean2, std2, p_value
    """
    mean1, std1 = np.mean(data1), np.std(data1)
    mean2, std2 = np.mean(data2), np.std(data2)
    
    # Check if data is normally distributed (Shapiro-Wilk test)
    # For large samples, we'll use a simpler approach
    if len(data1) > 5000 or len(data2) > 5000:
        # For very large samples, use t-test (Central Limit Theorem applies)
        _, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    else:
        # For smaller samples, check normality first
        try:
            _, p_norm1 = stats.shapiro(data1[:5000])  # Limit to 5000 for performance
            _, p_norm2 = stats.shapiro(data2[:5000])
            
            if p_norm1 > 0.05 and p_norm2 > 0.05:
                # Both normally distributed, use t-test
                _, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            else:
                # Not normally distributed, use Mann-Whitney U test
                _, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        except:
            # Fallback to Mann-Whitney U test if Shapiro fails
            _, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    
    return mean1, std1, mean2, std2, p_value


def format_p_value(p_value):
    """Format p-value for display."""
    if p_value < 0.001:
        return "p < 0.001"
    elif p_value < 0.01:
        return f"p = {p_value:.3f}"
    elif p_value < 0.05:
        return f"p = {p_value:.2f}"
    else:
        return f"p = {p_value:.2f}"


def load_all_conditions_data():
    """Load data from all four stimulation conditions."""
    base_dirs = {
        'Basic': 'PyNN_results',
        'pain': 'PyNN_pain_results', 
        'TMS': 'PyNN_TMS_results',
        'visual': 'PyNN_visual_results'
    }
    
    # Update data structure to include benchmark (COBA/CUBA) level
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for condition, base_dir in base_dirs.items():
        # Find any data directory instead of hardcoding dates
        data_dir = find_data_directory(base_dir)
        if data_dir is None:
            print(f"Warning: No data directory found in {base_dir}, skipping {condition}")
            continue
            
        print(f"Loading data from {data_dir} for {condition} condition")
            
        pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        
        for pkl_file in pkl_files:
            filepath = os.path.join(data_dir, pkl_file)
            network_name, neuron_type, benchmark = identify_network_and_type(filepath)
            
            if network_name is None or benchmark is None:
                continue
                
            data = load_pynn_data(filepath)
            firing_rates = extract_firing_rates(data)
            
            if firing_rates:  # Only store if we have data
                all_data[network_name][benchmark][neuron_type][condition].extend(firing_rates)
    
    return all_data


def plot_all_kde_comparisons_in_one_figure(all_data, output_dir='kde_comparison_plots'):
    """Create a single figure with all KDE comparison plots arranged in rows and columns."""
    os.makedirs(output_dir, exist_ok=True)
    
    comparisons = [
        ('Basic', 'pain', 'Basic vs Pain'),
        ('Basic', 'TMS', 'Basic vs TMS'),
        ('Basic', 'visual', 'Basic vs Visual')
    ]
    
    # Get all available networks, benchmarks, and neuron types
    networks = list(all_data.keys())
    benchmarks = []  # Will collect unique benchmarks (COBA, CUBA)
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
    
    # Calculate subplot dimensions for 4×6 grid layout
    total_plots = sum(len(comparisons_list) for _, _, _, comparisons_list in valid_combinations)
    cols = 6  # Fixed number of columns for 4×6 grid
    rows = 4  # Fixed number of rows for 4×6 grid
    
    if total_plots == 0:
        print("No plots to create!")
        return
    
    # Create the main figure with 4×6 layout
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16))  # 4 rows × 6 columns
    
    plot_idx = 0
    for network, benchmark, neuron_type, valid_comparisons in valid_combinations:
        condition_data = all_data[network][benchmark][neuron_type]
        
        # Create plots for each comparison type
        for col_idx, (cond1, cond2, title_suffix) in enumerate(valid_comparisons):
            if plot_idx >= rows * cols:
                print(f"Warning: More plots than available subplot slots ({rows}×{cols}). Skipping remaining plots.")
                break
                
            row_idx = plot_idx // cols
            actual_col_idx = plot_idx % cols
            ax = axes[row_idx, actual_col_idx]
            
            data1 = condition_data[cond1]
            data2 = condition_data[cond2]
            
            # Calculate statistics including p-value
            mean1, std1, mean2, std2, p_value = calculate_statistical_comparison(data1, data2)
            
            # Plot STANDARD KDE curves (firing rates on x-axis, density on y-axis) with attractive colors
            sns.kdeplot(data=data1, label=cond1.capitalize(), alpha=0.8, linewidth=2.5, ax=ax, color='#2E86AB', fill=True, levels=10)  # Deep blue
            sns.kdeplot(data=data2, label=cond2.capitalize(), alpha=0.8, linewidth=2.5, ax=ax, color='#A23B72', fill=True, levels=10)  # Rich magenta
            
            ax.axvline(mean1, color='#2E86AB', linestyle='--', alpha=0.9, linewidth=2,
                      label=f'{cond1.capitalize()} mean±std: {mean1:.2f}±{std1:.2f}')
            ax.axvline(mean2, color='#A23B72', linestyle='--', alpha=0.9, linewidth=2,
                      label=f'{cond2.capitalize()} mean±std: {mean2:.2f}±{std2:.2f}')
            
            # Add arrows pointing to mean values inside the plot area with offset to prevent overlap
            if mean1 <= mean2:
                # Basic condition (left), Comparison condition (right)
                ax.annotate(f'{mean1:.1f}', xy=(mean1, 0.02), xytext=(mean1 - 3, 0.08),
                           arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5),
                           ha='center', va='bottom', color='#2E86AB', fontweight='bold')
                ax.annotate(f'{mean2:.1f}', xy=(mean2, 0.02), xytext=(mean2 + 3, 0.08),
                           arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
                           ha='center', va='bottom', color='#A23B72', fontweight='bold')
            else:
                # Comparison condition (left), Basic condition (right)
                ax.annotate(f'{mean1:.1f}', xy=(mean1, 0.02), xytext=(mean1 + 3, 0.08),
                           arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5),
                           ha='center', va='bottom', color='#2E86AB', fontweight='bold')
                ax.annotate(f'{mean2:.1f}', xy=(mean2, 0.02), xytext=(mean2 - 3, 0.08),
                           arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
                           ha='center', va='bottom', color='#A23B72', fontweight='bold')
            
            # Add p-value text to the plot
            p_text = format_p_value(p_value)
            significance_text = f"{p_text}"
            if p_value < 0.05:
                significance_text += " *"  # Add asterisk for significant results
            
            # Position p-value text in upper right corner
            ax.text(0.95, 0.95, significance_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   fontsize=10, fontweight='bold' if p_value < 0.05 else 'normal')
            
            # Set axis limits based on benchmark type
            if benchmark == 'CUBA':
                ax.set_xlim(-5, 15)
                ax.set_ylim(0, 1.5)
            else:
                # COBA plots: x-axis = firing rate (-15 to 40 Hz), y-axis = density (0 to 0.05)
                ax.set_xlim(-5, 25)
                ax.set_ylim(0, 1.0)
            
            ax.set_xlabel('Firing Rate (Hz)')
            ax.set_ylabel('Density')
            ax.set_title(f'{network} - {benchmark} - {neuron_type.upper()}\n{title_suffix}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Hide any unused subplots
    for remaining_idx in range(plot_idx, rows * cols):
        row_idx = remaining_idx // cols
        col_idx = remaining_idx % cols
        axes[row_idx, col_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the combined plot
    filepath = os.path.join(output_dir, 'all_kde_comparisons_combined.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined plot: {filepath}")
    
    # Also save individual plots for reference (keeping original functionality)
    for network, benchmark, neuron_type, valid_comparisons in valid_combinations:
        condition_data = all_data[network][benchmark][neuron_type]
        
        for cond1, cond2, title_suffix in valid_comparisons:
            data1 = condition_data[cond1]
            data2 = condition_data[cond2]
            
            # Calculate statistics including p-value
            mean1, std1, mean2, std2, p_value = calculate_statistical_comparison(data1, data2)
            
            plt.figure(figsize=(10, 6))
            # Plot STANDARD KDE curves (firing rates on x-axis, density on y-axis) with attractive colors
            sns.kdeplot(data=data1, label=cond1.capitalize(), alpha=0.8, linewidth=2.5, color='#2E86AB', fill=True, levels=10)  # Deep blue
            sns.kdeplot(data=data2, label=cond2.capitalize(), alpha=0.8, linewidth=2.5, color='#A23B72', fill=True, levels=10)  # Rich magenta
            
            plt.axvline(mean1, color='#2E86AB', linestyle='--', alpha=0.9, linewidth=2,
                       label=f'{cond1.capitalize()} mean: {mean1:.2f}±{std1:.2f}')
            plt.axvline(mean2, color='#A23B72', linestyle='--', alpha=0.9, linewidth=2,
                       label=f'{cond2.capitalize()} mean: {mean2:.2f}±{std2:.2f}')
            
            # Add arrows pointing to mean values inside the plot area with offset to prevent overlap
            if mean1 <= mean2:
                # Basic condition (left), Comparison condition (right)
                plt.annotate(f'{mean1:.1f}', xy=(mean1, 0.02), xytext=(mean1 - 3, 0.08),
                            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5),
                            ha='center', va='bottom', color='#2E86AB', fontweight='bold')
                plt.annotate(f'{mean2:.1f}', xy=(mean2, 0.02), xytext=(mean2 + 3, 0.08),
                            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
                            ha='center', va='bottom', color='#A23B72', fontweight='bold')
            else:
                # Comparison condition (left), Basic condition (right)
                plt.annotate(f'{mean1:.1f}', xy=(mean1, 0.02), xytext=(mean1 + 3, 0.08),
                            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5),
                            ha='center', va='bottom', color='#2E86AB', fontweight='bold')
                plt.annotate(f'{mean2:.1f}', xy=(mean2, 0.02), xytext=(mean2 - 3, 0.08),
                            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5),
                            ha='center', va='bottom', color='#A23B72', fontweight='bold')
            
            # Add p-value text to the individual plot
            p_text = format_p_value(p_value)
            significance_text = f"{p_text}"
            if p_value < 0.05:
                significance_text += " *"
            
            plt.text(0.95, 0.95, significance_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    fontsize=10, fontweight='bold' if p_value < 0.05 else 'normal')
            
            # Set axis limits based on benchmark type
            if benchmark == 'CUBA':
                plt.xlim(-5, 15)
                plt.ylim(0, 1.5)
            else:
                # COBA plots: x-axis = firing rate (-15 to 40 Hz), y-axis = density (0 to 0.05)
                plt.xlim(-5, 25)
                plt.ylim(0, 1.0)
            
            plt.xlabel('Firing Rate (Hz)')
            plt.ylabel('Density')
            plt.title(f'{network} - {benchmark} - {neuron_type.upper()} Neurons\n{title_suffix}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save individual plot
            filename = f"{network}_{benchmark}_{neuron_type}_{cond1}_vs_{cond2}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved individual plot: {filepath}")


def main():
    """Main function to run the KDE comparison plotting."""
    print("Loading PyNN simulation data from all conditions...")
    all_data = load_all_conditions_data()
    
    # Check if we have any data
    if not all_data:
        print("No data found in any of the condition directories!")
        return
    
    print("Creating KDE comparison plots with statistical analysis...")
    plot_all_kde_comparisons_in_one_figure(all_data)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot KDE comparisons for PyNN simulation results")
    parser.add_argument("--output-dir", default="kde_comparison_plots", 
                       help="Output directory for plots (default: kde_comparison_plots)")
    args = parser.parse_args()
    
    main()