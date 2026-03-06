#!/usr/bin/env python3
"""
Script to create an animated visualization of how different stimulations 
affect network firing rates over time.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import yaml
import os

# Import the refactored stimulation modules
try:
    import tms_stimulation
    import visual_stimulation  
    import pain_stimulation
    STIMULATION_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import stimulation modules: {e}")
    STIMULATION_MODULES_AVAILABLE = False

# Network IDs
network_ids = [
    'M2M1S1_max_plus',
    'max_CTC_plus',
    # 'M1a_max_plus',
    # 'M1_max_plus',
    # 'M2_max_plus',
    # 'M2aM1aS1a_max_plus',
    # 'S1bM1bM2b_max_plus',
    # 'TC2PT',
    # 'TC2CT',
    # 'TC2IT4_IT2CT',
    # 'TC2IT2PTCT',
]

# Simulation result files for multiple runs
simulation_result_files = [
    ('main', '/home/leo520/pynml/simulation_results.json'),
    ('MF_optimized', '/home/leo520/pynml/MF_optimized/actual_simulation_results.json'),
    ('MF_valid', '/home/leo520/pynml/MF_valid/actual_simulation_results.json'),
]

def load_simulation_params():
    """
    Load simulation parameters from YAML file.
    """
    try:
        with open('/home/leo520/pynml/yaml/simulation_params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Create default parameters if file doesn't exist
        return {
            'stimuli': {
                'visual_stimuli': {'frequency': 20.0, 'phase': 0.0},
                'TMS_monophasic': {'frequency': 5.0, 'phase': 0.0},
                'TMS_half_sine': {'frequency': 5.0, 'phase': 0.0},
                'TMS_biphasic': {'frequency': 5.0, 'phase': 0.0},
                'pain_stimuli': {'frequency': 10.0, 'phase': 0.0}
            }
        }

def load_simulation_results(filename='simulation_results.json'):
    """
    Load simulation results from JSON file.
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # If no saved results, generate default ones
        return generate_default_simulation_results()

def generate_default_simulation_results():
    """
    Generate default simulation results when no saved data is available.
    """
    results = {}
    for network_id in network_ids:
        results[network_id] = {
            'visual_stimuli': [100.0, 0.0],
            'TMS_monophasic': [70.0, 35.0],
            'TMS_half_sine': [72.0, 36.0],
            'TMS_biphasic': [68.0, 34.0],
            'pain_stimuli': [60.0, 60.0]
        }
    return results

def get_real_time_stimulation_data():
    """
    Get real-time stimulation data from the refactored modules.
    This provides more accurate and dynamic data than static JSON files.
    """
    if not STIMULATION_MODULES_AVAILABLE:
        return None
    
    # Get steady-state rates from each module
    tms_rates = tms_stimulation.get_tms_steady_state_rates()
    visual_rates = visual_stimulation.get_visual_steady_state_rates()
    pain_rates = pain_stimulation.get_pain_steady_state_rates()
    
    # Combine all rates
    all_rates = {}
    all_rates.update(tms_rates)
    all_rates.update(visual_rates)
    all_rates.update(pain_rates)
    
    # Create simulation results structure matching the expected format
    results = {}
    for network_id in network_ids:
        results[network_id] = {}
        for stimulus_type, rates in all_rates.items():
            results[network_id][stimulus_type] = rates
    
    return results

def create_individual_network_animation(network_id, simulation_results, params, run_name="main"):
    """
    Create an animation for a specific network showing its response to all stimulations.
    
    Args:
        network_id (str): The ID of the network to animate
        simulation_results (dict): The simulation results dictionary
        params (dict): Simulation parameters
        run_name (str): Name of the run for file naming
    """
    # Define stimulus types
    stimulus_types = [
        "visual_stimuli", 
        "TMS_monophasic", 
        "TMS_half_sine", 
        "TMS_biphasic", 
        "pain_stimuli"
    ]
    
    # Target E/I ratios
    target_ratios = {
        'visual_stimuli': '∞',
        'TMS_monophasic': '2.0',
        'TMS_half_sine': '2.0',
        'TMS_biphasic': '2.0',
        'pain_stimuli': '1.0'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'Dynamic Response of {network_id} to Different Stimulations ({run_name})', 
                 fontsize=16, fontweight='bold')
    
    # Time parameters for animation (in milliseconds)
    duration_ms = 3000  # milliseconds
    dt_ms = 20  # time step in milliseconds
    time_points_ms = np.arange(0, duration_ms, dt_ms)
    n_frames = len(time_points_ms)
    
    # Set up the axes - extended y-axis to show full range (0-550 Hz)
    ax.set_xlim(0, duration_ms)
    ax.set_ylim(0, 550)  # Extended to show full range of firing rates
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.grid(True, alpha=0.3)
    
    # Create lines for each stimulus type
    lines = {}
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, stimulus_type in enumerate(stimulus_types):
        line, = ax.plot([], [], color=colors[i], linewidth=0.5, 
                       label=f'{stimulus_type} (Target E/I: {target_ratios[stimulus_type]})')
        lines[stimulus_type] = line
    
    # Add excitatory/inhibitory distinction
    e_lines = {}
    i_lines = {}
    
    for i, stimulus_type in enumerate(stimulus_types):
        e_line, = ax.plot([], [], '--', color=colors[i], linewidth=0.5, alpha=0.7)
        i_line, = ax.plot([], [], '-.', color=colors[i], linewidth=0.5, alpha=0.7)
        e_lines[stimulus_type] = e_line
        i_lines[stimulus_type] = i_line
    
    ax.legend(loc='upper right')
    
    # Text annotations
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    current_stim_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)
    
    # Initialize function for animation
    def init():
        for line in lines.values():
            line.set_data([], [])
        for line in e_lines.values():
            line.set_data([], [])
        for line in i_lines.values():
            line.set_data([], [])
        time_text.set_text('')
        current_stim_text.set_text('')
        return list(lines.values()) + list(e_lines.values()) + list(i_lines.values()) + [time_text, current_stim_text]
    
    # Animation function
    def animate(frame):
        # Current time in milliseconds
        t_ms = time_points_ms[frame]
        # Convert to seconds for calculations
        t = t_ms / 1000.0
        
        # Determine which stimulus is "active" (for visualization purposes)
        stim_cycle_time = t % len(stimulus_types)
        active_stim_index = int(stim_cycle_time)
        if active_stim_index >= len(stimulus_types):
            active_stim_index = len(stimulus_types) - 1
        active_stimulus = stimulus_types[active_stim_index]
        
        # Update each stimulus line
        for stimulus_type in stimulus_types:
            # Get stimulus parameters
            stim_params = params['stimuli'][stimulus_type]
            frequency = stim_params.get('frequency', 10.0)
            phase = stim_params.get('phase', 0.0)
            
            # Calculate instantaneous stimulation effect with stronger modulation
            stim_effect = 1.0 + 0.8 * np.sin(2 * np.pi * frequency * t + phase)
            
            # Get the steady-state rates from simulation results
            steady_rates = simulation_results[network_id][stimulus_type]
            e_steady = steady_rates[0]
            i_steady = steady_rates[1]
            
            # Apply time-varying modulation with higher frequency for more movement
            modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation for visual effect
            
            # Calculate instantaneous rates
            e_rate = e_steady * modulation * stim_effect
            i_rate = i_steady * modulation * stim_effect
            
            # Ensure rates are positive and within bounds
            e_rate = max(0, min(e_rate, 550))
            i_rate = max(0, min(i_rate, 550))
            
            # Update the lines with cumulative data
            if frame == 0:
                lines[stimulus_type].set_data([t_ms], [(e_rate + i_rate) / 2])  # Average rate
                e_lines[stimulus_type].set_data([t_ms], [e_rate])
                i_lines[stimulus_type].set_data([t_ms], [i_rate])
            else:
                # Get existing data
                x_data, y_data = lines[stimulus_type].get_data()
                ex_data, ey_data = e_lines[stimulus_type].get_data()
                ix_data, iy_data = i_lines[stimulus_type].get_data()
                
                # Append new data
                lines[stimulus_type].set_data(np.append(x_data, t_ms), 
                                            np.append(y_data, (e_rate + i_rate) / 2))
                e_lines[stimulus_type].set_data(np.append(ex_data, t_ms), np.append(ey_data, e_rate))
                i_lines[stimulus_type].set_data(np.append(ix_data, t_ms), np.append(iy_data, i_rate))
        
        # Update text annotations
        time_text.set_text(f'Time: {t_ms:.0f} ms')
        current_stim_text.set_text(f'Currently emphasizing: {active_stimulus}')
        
        return (list(lines.values()) + list(e_lines.values()) + list(i_lines.values()) + 
                [time_text, current_stim_text])
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames,
        interval=50, blit=True, repeat=True
    )
    
    # Save animation
    filename = f'{network_id}_stimulation_response_{run_name}.gif'
    print(f"Saving animation for {network_id} ({run_name})...")
    anim.save(filename, writer='pillow', fps=20)
    print(f"Animation saved as '{filename}'")
    
    plt.close(fig)  # Close figure to free memory
    
    return anim

def create_all_networks_animations(use_real_time_data=True):
    """
    Create animations for all networks across all available simulation runs.
    """
    # Load simulation parameters once
    params = load_simulation_params()
    
    # Process each simulation result file
    for run_name, result_file in simulation_result_files:
        if not os.path.exists(result_file):
            print(f"Skipping {run_name}: file {result_file} not found")
            continue
            
        print(f"\nProcessing run: {run_name}")
        
        # Load simulation results for this run
        if use_real_time_data and STIMULATION_MODULES_AVAILABLE:
            print("Using real-time stimulation data from modules...")
            simulation_results = get_real_time_stimulation_data()
        else:
            print(f"Using saved simulation results from {result_file}...")
            simulation_results = load_simulation_results(result_file)
        
        # Create animation for each network
        for network_id in network_ids:
            if network_id in simulation_results:
                try:
                    create_individual_network_animation(network_id, simulation_results, params, run_name)
                except Exception as e:
                    print(f"Error creating animation for {network_id} ({run_name}): {e}")
            else:
                print(f"Warning: {network_id} not found in {run_name} results")

def create_stimulation_animation(use_real_time_data=True):
    """
    Create an animation showing how different stimulations affect network firing rates over time.
    
    Args:
        use_real_time_data (bool): Whether to use real-time data from stimulation modules
    """
    # Load simulation results and parameters
    if use_real_time_data and STIMULATION_MODULES_AVAILABLE:
        print("Using real-time stimulation data from modules...")
        simulation_results = get_real_time_stimulation_data()
    else:
        print("Using saved simulation results...")
        simulation_results = load_simulation_results()
    
    params = load_simulation_params()
    
    # Define stimulus types and their properties
    stimulus_types = [
        "TMS_monophasic", 
        "TMS_half_sine", 
        "TMS_biphasic", 
        "visual_stimuli",
        "pain_stimuli"
    ]
    
    # Target E/I ratios
    target_ratios = {
        'TMS_monophasic': '2.0',
        'TMS_half_sine': '2.0',
        'TMS_biphasic': '2.0',
        'visual_stimuli': '∞',
        'pain_stimuli': '1.0'
    }
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    fig.suptitle('Dynamic Response of Neural Networks to Different Stimulations', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Initialize lines for each subplot
    e_lines = []
    i_lines = []
    time_text_list = []
    stim_text_list = []
    
    # Time parameters for animation (in milliseconds)
    duration_ms = 20000  # milliseconds (20 seconds) - 5 stimulus types × 4 seconds each
    dt_ms = 20  # time step in milliseconds
    time_points_ms = np.arange(0, duration_ms, dt_ms)
    n_frames = len(time_points_ms)
    
    # Create lines for each stimulus type
    for idx, stimulus_type in enumerate(stimulus_types):
        ax = axes[idx]
        
        # Set up the axes - extended y-axis to show full range (0-550 Hz)
        ax.set_xlim(0, duration_ms)
        ax.set_ylim(0, 550)  # Extended to show full range of firing rates
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title(f'{stimulus_type}\n(Target E/I: {target_ratios[stimulus_type]})')
        ax.grid(True, alpha=0.3)
        
        # Create lines for excitatory and inhibitory rates
        e_line, = ax.plot([], [], 'r-', linewidth=0.5, label='Excitatory')
        i_line, = ax.plot([], [], 'b-', linewidth=0.5, label='Inhibitory')
        e_lines.append(e_line)
        i_lines.append(i_line)
        
        # Add text for displaying current time and stimulus info
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
        stim_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
        time_text_list.append(time_text)
        stim_text_list.append(stim_text)
        
        ax.legend()
    
    # Hide the extra subplot
    axes[5].set_visible(False)
    
    # Initialize function for animation
    def init():
        for e_line, i_line, time_text, stim_text in zip(e_lines, i_lines, time_text_list, stim_text_list):
            e_line.set_data([], [])
            i_line.set_data([], [])
            time_text.set_text('')
            stim_text.set_text('')
        return e_lines + i_lines + time_text_list + stim_text_list
    
    # Animation function
    def animate(frame):
        # Current time in milliseconds
        t_ms = time_points_ms[frame]
        # Convert to seconds for calculations
        t = t_ms / 1000.0
        
        # Update each subplot
        for idx, stimulus_type in enumerate(stimulus_types):
            # Get stimulus parameters
            stim_params = params['stimuli'][stimulus_type]
            frequency = stim_params.get('frequency', 10.0)
            phase = stim_params.get('phase', 0.0)
            
            # Calculate instantaneous stimulation effect with stronger modulation
            stim_effect = 1.0 + 0.8 * np.sin(2 * np.pi * frequency * t + phase)
            
            # Store data for this frame
            e_rates_over_time = []
            i_rates_over_time = []
            times = []
            
            # For each network, calculate how the rates evolve over time
            for network_id in network_ids:
                # Get the steady-state rates from simulation results
                steady_rates = simulation_results[network_id][stimulus_type]
                e_steady = steady_rates[0]
                i_steady = steady_rates[1]
                
                # Apply time-varying modulation with higher frequency for more movement
                # This simulates how the stimulation affects the network over time
                modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation for more visual effect
                
                # Calculate instantaneous rates with stronger modulation
                e_rate = e_steady * modulation * stim_effect
                i_rate = i_steady * modulation * stim_effect
                
                # Ensure rates are positive and within bounds
                e_rate = max(0, min(e_rate, 550))
                i_rate = max(0, min(i_rate, 550))
                
                e_rates_over_time.append(e_rate)
                i_rates_over_time.append(i_rate)
            
            # Calculate average rates across networks for display
            avg_e_rate = np.mean(e_rates_over_time)
            avg_i_rate = np.mean(i_rates_over_time)
            
            # Update the lines with cumulative data
            if frame == 0:
                e_lines[idx].set_data([t_ms], [avg_e_rate])
                i_lines[idx].set_data([t_ms], [avg_i_rate])
            else:
                # Get existing data
                ex_data, ey_data = e_lines[idx].get_data()
                ix_data, iy_data = i_lines[idx].get_data()
                
                # Append new data
                e_lines[idx].set_data(np.append(ex_data, t_ms), np.append(ey_data, avg_e_rate))
                i_lines[idx].set_data(np.append(ix_data, t_ms), np.append(iy_data, avg_i_rate))
            
            # Update text annotations
            time_text_list[idx].set_text(f'Time: {t_ms:.0f} ms')
            
            # Calculate current E/I ratio
            if avg_i_rate > 0:
                ei_ratio = avg_e_rate / avg_i_rate
                stim_text_list[idx].set_text(f'E/I: {ei_ratio:.2f}')
            else:
                stim_text_list[idx].set_text('E/I: ∞')
        
        return e_lines + i_lines + time_text_list + stim_text_list
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames,
        interval=50, blit=True, repeat=True
    )
    
    # Save animation
    print("Saving animation...")
    anim.save(f'stimulation_dynamics.gif', writer='pillow', fps=20)
    print(f"Animation saved as 'stimulation_dynamics.gif'")
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return anim

def create_network_specific_animation(use_real_time_data=True):
    """
    Create an animation showing how a specific network responds to all stimulations over time.
    
    Args:
        use_real_time_data (bool): Whether to use real-time data from stimulation modules
    """
    # Load simulation results and parameters
    if use_real_time_data and STIMULATION_MODULES_AVAILABLE:
        print("Using real-time stimulation data from modules...")
        simulation_results = get_real_time_stimulation_data()
    else:
        print("Using saved simulation results...")
        simulation_results = load_simulation_results()
    
    params = load_simulation_params()
    
    # Select one network to focus on (first one)
    selected_network = network_ids[0]
    print(f"Creating animation for network: {selected_network}")
    
    # Define stimulus types
    stimulus_types = [
        "visual_stimuli", 
        "TMS_monophasic", 
        "TMS_half_sine", 
        "TMS_biphasic", 
        "pain_stimuli"
    ]
    
    # Target E/I ratios
    target_ratios = {
        'visual_stimuli': '∞',
        'TMS_monophasic': '2.0',
        'TMS_half_sine': '2.0',
        'TMS_biphasic': '2.0',
        'pain_stimuli': '1.0'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.suptitle(f'Dynamic Response of {selected_network} to Different Stimulations', 
                 fontsize=16, fontweight='bold')
    
    # Time parameters for animation (in milliseconds)
    duration_ms = 20000  # milliseconds (20 seconds) - 5 stimulus types × 4 seconds each
    dt_ms = 20  # time step in milliseconds (reduced from 50 for smoother curves)
    time_points_ms = np.arange(0, duration_ms, dt_ms)
    n_frames = len(time_points_ms)
    
    # Set up the axes - extended y-axis to show full range (0-550 Hz)
    ax.set_xlim(0, duration_ms)
    ax.set_ylim(0, 550)  # Extended to show full range of firing rates
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.grid(True, alpha=0.3)
    
    # Create lines for each stimulus type
    lines = {}
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, stimulus_type in enumerate(stimulus_types):
        line, = ax.plot([], [], color=colors[i], linewidth=1.5,  
                       label=f'{stimulus_type} (Target E/I: {target_ratios[stimulus_type]})')
        lines[stimulus_type] = line
    
    # Add excitatory/inhibitory distinction
    e_lines = {}
    i_lines = {}
    
    for i, stimulus_type in enumerate(stimulus_types):
        e_line, = ax.plot([], [], '--', color=colors[i], linewidth=0.5, alpha=0.8)  # Increased linewidth and alpha
        i_line, = ax.plot([], [], '-.', color=colors[i], linewidth=0.5, alpha=0.8)  # Increased linewidth and alpha
        e_lines[stimulus_type] = e_line
        i_lines[stimulus_type] = i_line
    
    ax.legend(loc='upper right')
    
    # Text annotations
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    current_stim_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)
    
    # Initialize function for animation
    def init():
        for line in lines.values():
            line.set_data([], [])
        for line in e_lines.values():
            line.set_data([], [])
        for line in i_lines.values():
            line.set_data([], [])
        time_text.set_text('')
        current_stim_text.set_text('')
        return list(lines.values()) + list(e_lines.values()) + list(i_lines.values()) + [time_text, current_stim_text]
    
    # Animation function
    def animate(frame):
        # Current time in milliseconds
        t_ms = time_points_ms[frame]
        # Convert to seconds for calculations
        t = t_ms / 1000.0
        
        # Determine which stimulus is currently being emphasized (4 seconds per stimulus type)
        duration_per_stimulus = 4.0  # seconds
        current_stim_index = int(t // duration_per_stimulus) % len(stimulus_types)
        active_stimulus = stimulus_types[current_stim_index]
        
        # Update each stimulus line
        for stimulus_type in stimulus_types:
            # Get stimulus parameters
            stim_params = params['stimuli'][stimulus_type]
            frequency = stim_params.get('frequency', 10.0)
            phase = stim_params.get('phase', 0.0)
            
            # Calculate instantaneous stimulation effect with stronger modulation
            stim_effect = 1.0 + 0.8 * np.sin(2 * np.pi * frequency * t + phase)
            
            # Get the steady-state rates from simulation results
            steady_rates = simulation_results[selected_network][stimulus_type]
            e_steady = steady_rates[0]
            i_steady = steady_rates[1]
            
            # Apply time-varying modulation with optimized frequency for clearer visualization
            modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation for clearer visual effect
            
            # Calculate instantaneous rates
            e_rate = e_steady * modulation * stim_effect
            i_rate = i_steady * modulation * stim_effect
            
            # Ensure rates are positive and within bounds
            e_rate = max(0, min(e_rate, 550))
            i_rate = max(0, min(i_rate, 550))
            
            # Update the lines with cumulative data
            if frame == 0:
                lines[stimulus_type].set_data([t_ms], [(e_rate + i_rate) / 2])  # Average rate
                e_lines[stimulus_type].set_data([t_ms], [e_rate])
                i_lines[stimulus_type].set_data([t_ms], [i_rate])
            else:
                # Get existing data
                x_data, y_data = lines[stimulus_type].get_data()
                ex_data, ey_data = e_lines[stimulus_type].get_data()
                ix_data, iy_data = i_lines[stimulus_type].get_data()
                
                # Append new data
                lines[stimulus_type].set_data(np.append(x_data, t_ms), 
                                            np.append(y_data, (e_rate + i_rate) / 2))
                e_lines[stimulus_type].set_data(np.append(ex_data, t_ms), np.append(ey_data, e_rate))
                i_lines[stimulus_type].set_data(np.append(ix_data, t_ms), np.append(iy_data, i_rate))
        
        # Update text annotations
        time_text.set_text(f'Time: {t_ms:.0f} ms')
        current_stim_text.set_text(f'Currently emphasizing: {active_stimulus}')
        
        return (list(lines.values()) + list(e_lines.values()) + list(i_lines.values()) + 
                [time_text, current_stim_text])
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames,
        interval=50, blit=True, repeat=True
    )
    
    # Save animation
    print("Saving network-specific animation...")
    anim.save(f'{selected_network}_stimulation_response.gif', writer='pillow', fps=20)
    print(f"Animation saved as '{selected_network}_stimulation_response.gif'")
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return anim

def main():
    """
    Main function to create animations of stimulation dynamics.
    """
    print("Creating animated visualization of stimulation dynamics...")
    
    # Check if stimulation modules are available
    if STIMULATION_MODULES_AVAILABLE:
        print("Real-time stimulation modules are available!")
        use_real_time = True
    else:
        print("Using saved simulation results (real-time modules not available)")
        use_real_time = False
    
    # Ask user what type of animations to create
    print("\nChoose animation type:")
    print("1. All stimulations overview (averaged across networks)")
    print("2. Single network animation (first network only)")
    print("3. All networks for all runs (creates many GIFs)")
    print("4. All of the above")
    
    choice = input("Enter your choice (1-4, default=4): ").strip()
    if choice == "":
        choice = "4"
    
    if choice in ["1", "4"]:
        # Create animation showing all stimulations
        print("\n1. Creating animation showing all stimulations...")
        create_stimulation_animation(use_real_time_data=use_real_time)
    
    if choice in ["2", "4"]:
        # Create animation focusing on one network
        print("\n2. Creating network-specific animation...")
        create_network_specific_animation(use_real_time_data=use_real_time)
    
    if choice in ["3", "4"]:
        # Create animations for all networks and all runs
        print("\n3. Creating animations for all networks and all runs...")
        create_all_networks_animations(use_real_time_data=use_real_time)
    
    print("\nAnimations created successfully!")

if __name__ == '__main__':
    main()