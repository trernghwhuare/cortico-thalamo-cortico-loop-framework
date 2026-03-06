#!/usr/bin/env python3

"""
Combined Network animation using graph-tool and matplotlib.
This script creates an animated visualization of neural networks showing both
dynamic network topology changes and node state transitions.
"""
import os
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "16"
import json
import gi
gi.require_version('Gtk', '3.0')
# from gi.repository import Gtk, Gdk, GLib
from random import randint, shuffle, random
import numpy as np
from graph_tool.all import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend("cairo")
from numpy.linalg import norm
from numpy.random import *
import csv
import sys


os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# %%
seed(42)
seed_rng(42)

# Constants for layout
step = 0.005       # move step
K = 0.5            # preferred edge length

# SIRS Model parameters
x = 0.004   # spontaneous outbreak probability
r = 0.4     # I->R probability
s = 0.04    # R->S probability

# Create colormaps for the three states
cmap_S = plt.get_cmap('tab20c')
cmap_I = plt.get_cmap('coolwarm')
cmap_R = plt.get_cmap('viridis')    # state colormap
# Get specific colors from colormaps (using appropriate positions)
exc_S = list(cmap_S(0.5))[:4]          # inactive state 
exc_I = list(cmap_I(0.5))[:4]          # Active state 
exc_R = list(cmap_R(0.5))[:4]          # Refractory state 
inh_S = list(cmap_S(0.5))[:6]          # inactive state 
inh_I = list(cmap_I(0.5))[:6]          # Active state 
inh_R = list(cmap_R(0.5))[:6]          # Refractor state 

def load_network_data(network_name, data_dir):
    """
    Load network data from CSV files.
    Args:
        network_name (str): Name of the network to load
        data_dir (str): Directory containing the network data files
    Returns:
        tuple: (nodes_data, edges_data, num_nodes)
    """
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
    # Load params to get number of nodes
    params_file = os.path.join(data_dir, f"{network_name}_gt_params.json")
    with open(params_file, 'r') as f:
        params_data = json.load(f)
        num_nodes = params_data['number_of_populations']
        num_input_nodes = params_data['number_input_vertices']
    return nodes_data, edges_data, num_nodes, input_nodes_data, input_edges_data, num_input_nodes



def interpolate_color(color1, color2, factor):
    """Interpolate between two colors"""
    return [
        color1[0] + (color2[0] - color1[0]) * factor,
        color1[1] + (color2[1] - color1[1]) * factor,
        color1[2] + (color2[2] - color1[2]) * factor,
        color1[3] + (color2[3] - color1[3]) * factor
    ]

def state_to_rgba(state_value):
    """Convert state vector to RGBA values for smooth transitions"""
    # Compare with tolerance since we're now using float values from colormaps
    def approx_equal(a, b, tol=1e-6):
        return abs(a[0]-b[0]) < tol and abs(a[1]-b[1]) < tol and abs(a[2]-b[2]) < tol and abs(a[3]-b[3]) < tol
    
    if approx_equal(state_value, exc_S):  # S - inactive (from tab20c colormap)
        return exc_S
    elif approx_equal(state_value, exc_I):  # I - Active (from coolwarn colormap)
        return exc_I
    elif approx_equal(state_value, exc_R):  # R - Refractory (from viridis colormap)
        return exc_R
    elif approx_equal(state_value, inh_S):  # S - inactive (from tab20c colormap)
        return inh_S
    elif approx_equal(state_value, inh_I):  # I - Active (from coolwarn colormap)
        return inh_I
    elif approx_equal(state_value, inh_R):  # R - Refractory (from viridis colormap)
        return inh_R

def main():
    network_name = "max_CTC_plus"  
    data_dir = "gt/params"
    print(f"Loading network: {network_name}")
    try:
        nodes_data, edges_data, num_nodes, input_nodes_data, input_edges_data, num_input_nodes = load_network_data(network_name, data_dir)
        g = price_network(num_nodes + num_input_nodes, c=0.8, directed=False)
        ini_exc_state = AxelrodState(g, f=10, q=30, r=int(0.005))
        ini_inh_state = PottsGlauberState(g, np.eye(4) * 0.1)

        node_id_to_index = {}
        for i, node_info in enumerate(nodes_data):
            node_id_to_index[node_info['component']] = i
        inhibitory_nodes = set()
        excitatory_nodes = set()
        for node_info in nodes_data:
            if node_info['type'] == 'Inh' and node_info['component'] in node_id_to_index:
                inhibitory_nodes.add(node_id_to_index[node_info['component']])
            elif node_info['type'] == 'Exc' and node_info['component'] in node_id_to_index:
                excitatory_nodes.add(node_id_to_index[node_info['component']])
        edge_count = 0
        for edge_info in edges_data:
            src_id = edge_info['source']
            tgt_id = edge_info['target']
            if src_id in node_id_to_index and tgt_id in node_id_to_index:
                src_idx = node_id_to_index[src_id]
                tgt_idx = node_id_to_index[tgt_id]
                # g.add_edge(src_idx, tgt_idx)
            edge_count += 1 
        input_id_to_index = {}
        for i, node_info in enumerate(input_nodes_data):
            input_id_to_index[node_info['component']] = i 
        inhibitory_inputs = set()
        excitatory_inputs = set()
        for node_info in input_nodes_data:
            if node_info['type'] == 'Inh' and node_info['component'] in input_id_to_index:
                inhibitory_inputs.add(input_id_to_index[node_info['component']])
            elif node_info['type'] == 'Exc' and node_info['component'] in input_id_to_index:
                excitatory_inputs.add(input_id_to_index[node_info['component']])
        input_edge_count = 0
        for input_edge_info in input_edges_data:
            src_id = input_edge_info['source']
            tgt_id = input_edge_info['target']
            if src_id in input_id_to_index and tgt_id in node_id_to_index:
                src_idx = input_id_to_index[src_id]
                tgt_idx = node_id_to_index[tgt_id]
                # g.add_edge(tgt_idx, tgt_idx)
            input_edge_count += 1 
        
        # Generate layout using graph-tool
        pos = sfdp_layout(g, K=K, cooling_step=0.99, C=100, multilevel=True, R=20, gamma=1)      

        ini_exc_state = g.new_vertex_property("vector<double>")
        ini_inh_state = g.new_vertex_property("vector<double>")
        for v in g.vertices():
            ini_exc_state[v] = exc_S
            ini_inh_state[v] = inh_S
        
        curr_exc_state = g.new_vertex_property("vector<double>")
        prev_exc_state = g.new_vertex_property("vector<double>")
        newly_exc_transmited = g.new_vertex_property("bool")
        exc_refractory = g.new_vertex_property("bool")
        curr_inh_state = g.new_vertex_property("vector<double>")
        prev_inh_state = g.new_vertex_property("vector<double>")
        newly_inh_transmited = g.new_vertex_property("bool")
        inh_refractory = g.new_vertex_property("bool")
        
        for v in g.vertices():
            curr_exc_state[v] = exc_S 
            curr_inh_state[v] = inh_S
            prev_exc_state[v] = ini_exc_state[v] = exc_S
            prev_inh_state[v] = ini_inh_state[v] = inh_S
        # List of edges for dynamic rewiring
        edges = list(g.edges())
        max_count = 500
        # Check if we should disable offscreen rendering
        no_offscreen = sys.argv[1] == "no_offscreen" if len(sys.argv) > 1 else False
        offscreen = not no_offscreen
        network_frames_dir = f"./frames/{network_name}/combined_adv"
        if offscreen and not os.path.exists(network_frames_dir):
            os.makedirs(network_frames_dir)
            print(f"Created network frames directory: {network_frames_dir}")
        
        # Counter for animation frames
        count = 0
        
        # Track global min/max positions for consistent view across all frames
        all_x_positions = []
        all_y_positions = []
        vertex_index_map = {v: i for i, v in enumerate(g.vertices())}
        edge_index_map = {e: i for i, e in enumerate(g.edges())}
        
        # This function will be called repeatedly to update the vertex layout and states
        def update_state():
            nonlocal count, all_x_positions, all_y_positions, vertex_index_map, edge_index_map
            # Reset properties
            newly_exc_transmited.a = False
            newly_inh_transmited.a = False
            exc_refractory.a = False
            inh_refractory.a = False
            # Count states for debugging
            exc_inactive_count = 0
            exc_active_count = 0
            exc_refractory_count = 0
            inh_inactive_count = 0
            inh_active_count = 0
            inh_refractory_count = 0
            
            # Count current states before updating
            for v in g.vertices():
                v_idx = vertex_index_map[v]
                
                # Check if this is an excitatory node (regular or input)
                is_excitatory = (v_idx in excitatory_nodes) or (v_idx in excitatory_inputs)
                # Check if this is an inhibitory node (regular or input)
                is_inhibitory = (v_idx in inhibitory_nodes) or (v_idx in inhibitory_inputs)
                
                if is_excitatory:
                    if curr_exc_state[v] == exc_S:    # exc Inactive
                        exc_inactive_count += 1
                    elif curr_exc_state[v] == exc_I:  # exc Active
                        exc_active_count += 1
                        newly_exc_transmited[v] = True
                    elif curr_exc_state[v] == exc_R:  # exc Refractory
                        exc_refractory_count += 1
                
                if is_inhibitory:
                    if curr_inh_state[v] == inh_S:    # inh Inactive
                        inh_inactive_count += 1
                    elif curr_inh_state[v] == inh_I:  # inh Active
                        inh_active_count += 1
                        newly_inh_transmited[v] = True
                    elif curr_inh_state[v] == inh_R:  # inh Refractory
                        inh_refractory_count += 1
            
            print(f"Frame {count}: exc_S={exc_inactive_count} | exc_I={exc_active_count} | exc_R={exc_refractory_count} | inh_S={inh_inactive_count} | inh_I={inh_active_count} | inh_R={inh_refractory_count}")
            
            if count == 0 and exc_active_count == 0 and inh_active_count == 0:
                vs = list(g.vertices())
                shuffle(vs)
                
                # Separate nodes by type
                exc_nodes = [v for v in vs if vertex_index_map[v] in excitatory_nodes]
                inh_nodes = [v for v in vs if vertex_index_map[v] in inhibitory_nodes]
                exc_input_nodes = [v for v in vs if vertex_index_map[v] in excitatory_inputs]
                inh_input_nodes = [v for v in vs if vertex_index_map[v] in inhibitory_inputs]

                # Activate excitatory nodes
                num_initial_exc_infections = min(20, len(exc_nodes), len(exc_input_nodes))
                for i in range(num_initial_exc_infections):
                    if i < len(exc_nodes):
                        curr_exc_state[exc_nodes[i]] = exc_I
                        newly_exc_transmited[exc_nodes[i]] = True
                    if i < len(exc_input_nodes):
                        curr_exc_state[exc_input_nodes[i]] = exc_I
                        newly_exc_transmited[exc_input_nodes[i]] = True

                # Activate inhibitory nodes
                num_initial_inh_infections = min(20, len(inh_nodes), len(inh_input_nodes))
                for i in range(num_initial_inh_infections):
                    if i < len(inh_nodes):
                        curr_inh_state[inh_nodes[i]] = inh_I
                        newly_inh_transmited[inh_nodes[i]] = True
                    if i < len(inh_input_nodes):
                        curr_inh_state[inh_input_nodes[i]] = inh_I
                        newly_inh_transmited[inh_input_nodes[i]] = True
                    
            # Update node states (SIRS model)
            vs = list(g.vertices())
            shuffle(vs)
            for v in vs:
                v_idx = vertex_index_map[v]
                
                is_excitatory = (v_idx in excitatory_nodes) or (v_idx in excitatory_inputs)
                is_inhibitory = (v_idx in inhibitory_nodes) or (v_idx in inhibitory_inputs)
                
                # Update excitatory neuron states
                if is_excitatory:
                    if curr_exc_state[v] == exc_I:  # Active
                        if random() < r:
                            curr_exc_state[v] = exc_R
                    elif curr_exc_state[v] == exc_S:  # inactive
                        if random() < x:
                            curr_exc_state[v] = exc_I
                        else:
                            ns = list(v.out_neighbors())
                            if len(ns) > 0:
                                w = ns[randint(0, len(ns))]  # choose a random neighbor
                                if curr_exc_state[w] == exc_S and random() < 0.5:  # Activateion probability
                                    curr_exc_state[v] = exc_I
                                    newly_exc_transmited[v] = True
                                elif curr_exc_state[w] == exc_S and random() > 0.5:
                                    curr_exc_state[v] = inh_I
                                    newly_exc_transmited[v] = True
                    elif random() < s:  # Recover probability
                        curr_exc_state[v] = exc_S
                    if curr_exc_state[v] == exc_R:  # Refractory
                        exc_refractory[v] = True
                    
                # Update inhibitory neuron states
                if is_inhibitory:
                    if curr_inh_state[v] == inh_I:  # Active
                        if random() < r:
                            curr_inh_state[v] = inh_R
                    elif curr_inh_state[v] == inh_S:  # inactive
                        if random() < x:
                            curr_inh_state[v] = inh_I
                        else:
                            ns = list(v.out_neighbors())
                            if len(ns) > 0:
                                w = ns[randint(0, len(ns))]  
                                if curr_inh_state[w] == inh_I and random() < 0.5:  
                                    curr_inh_state[v] = inh_I
                                    newly_inh_transmited[v] = True
                                elif curr_inh_state[w] == inh_I and random() > 0.5:
                                    curr_inh_state[v] = exc_I
                                    newly_inh_transmited[v] = True
                    elif random() < s:
                        curr_inh_state[v] = inh_S
                    if curr_inh_state[v] == inh_R:  # Refractory
                        inh_refractory[v] = True
                        
            # Perform one iteration of the layout step, starting from the previous positions
            sfdp_layout(g, pos=pos, K=K, init_step=step, max_iter=1)

            # Only perform edge rewiring if there are edges in the graph
            if len(edges) > 0 and count % 2 == 0:  # Rewire every 2nd frame for better performance
                for i in range(500):  # rewiring iterations for performance
                    edge_index = randint(0, len(edges))
                    e = list(edges[edge_index])
                    shuffle(e)
                    s1, t1 = e
                    t2 = g.vertex(randint(0, g.num_vertices()))
                    if (norm(pos[s1].a - pos[t2].a) <= norm(pos[s1].a - pos[t1].a) and
                        s1 != t2 and                      # no self-loops
                        t1.out_degree() > 0 and            
                        t2 not in s1.out_neighbors()):    # no parallel edges
                        g.remove_edge(edges[edge_index])
                        edges[edge_index] = g.add_edge(s1, t2)

            # Save frame as PNG
            if offscreen:
                # Save every 5th frame for better performance
                if count % 5 != 0 and count != max_count - 1:
                    # Just increment counter and continue without saving
                    count += 1
                    # Store current states as previous states
                    for v in g.vertices():
                        prev_exc_state[v] = curr_exc_state[v]
                        prev_inh_state[v] = curr_inh_state[v]
                    return True
                
                # Save frame as PNG file
                try:
                    # Create a matplotlib figure for this frame with even dimensions
                    fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=150)
                    x_pos = [pos[v][0] for v in g.vertices()]
                    y_pos = [pos[v][1] for v in g.vertices()]
                    colors = []
                    sizes = []
                    for v in g.vertices():
                        v_idx = vertex_index_map[v]
                        is_excitatory = (v_idx in excitatory_nodes) or (v_idx in excitatory_inputs)
                        if is_excitatory:
                            exc_state = ini_exc_state[v]
                            colors.append(state_to_rgba(exc_state))
                            sizes.append(20)
                        else:  
                            inh_state = ini_inh_state[v]
                            colors.append(state_to_rgba(inh_state))
                            sizes.append(15)
                    ax.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.9, edgecolors='white', linewidth=0.5)

                    # Draw halos around active vertices for better visibility
                    exc_active_x_pos = []
                    exc_active_y_pos = []
                    inh_active_x_pos = []
                    inh_active_y_pos = []
                    exc_active_colors = []
                    exc_active_sizes = []
                    inh_active_colors = []
                    inh_active_sizes = []
                    for v in g.vertices():
                        if curr_exc_state[v] == exc_I:  # If vertex is active (infected)
                            exc_active_x_pos.append(pos[v][0])
                            exc_active_y_pos.append(pos[v][1])
                            exc_active_colors.append(state_to_rgba(curr_exc_state[v]))
                            exc_active_sizes.append(40)
                        elif curr_inh_state[v] == inh_I:  # If vertex is active (infected)
                            inh_active_x_pos.append(pos[v][0])
                            inh_active_y_pos.append(pos[v][1])
                            inh_active_colors.append(state_to_rgba(curr_inh_state[v]))
                            inh_active_sizes.append(30)
                    if exc_active_x_pos:
                        ax.scatter(exc_active_x_pos, exc_active_y_pos, c=exc_active_colors, # cmap='autumn', 
                                   s=exc_active_sizes, alpha=0.9, edgecolors='white', linewidth=0.5, zorder=3)
                    elif inh_active_x_pos:
                        ax.scatter(inh_active_x_pos, inh_active_y_pos, c=inh_active_colors, # cmap='autumn_r', 
                                   s=inh_active_sizes, alpha=0.9, edgecolors='white', linewidth=0.5, zorder=3)

                    exc_inactive_x_pos = []
                    exc_inactive_y_pos = []
                    inh_inactive_x_pos = []
                    inh_inactive_y_pos = []
                    exc_inactive_colors = []
                    exc_inactive_sizes = []
                    inh_inactive_colors = []
                    inh_inactive_sizes = []
                    for v in g.vertices():
                        if curr_exc_state[v] == exc_S:  # If vertex is inactive
                            exc_inactive_x_pos.append(pos[v][0])
                            exc_inactive_y_pos.append(pos[v][1])
                            exc_inactive_colors.append(state_to_rgba(curr_exc_state[v]))
                            exc_inactive_sizes.append(20)
                        elif curr_inh_state[v] == inh_S:  # If vertex is inactive
                            inh_inactive_x_pos.append(pos[v][0])
                            inh_inactive_y_pos.append(pos[v][1])
                            inh_inactive_colors.append(state_to_rgba(curr_inh_state[v]))
                            inh_inactive_sizes.append(15)
                    if exc_inactive_x_pos:
                        ax.scatter(exc_inactive_x_pos, exc_inactive_y_pos, c=exc_inactive_colors, # cmap='tab20c', 
                                   s=exc_inactive_sizes, alpha=0.5, edgecolors='white', linewidth=0.5, zorder=1)
                    elif inh_inactive_x_pos:
                        ax.scatter(inh_inactive_x_pos, inh_inactive_y_pos, c=inh_inactive_colors, # cmap='tab20c_r', 
                                   s=inh_inactive_sizes, alpha=0.5, edgecolors='white', linewidth=0.5, zorder=1)

                    exc_refractory_x_pos = []
                    exc_refractory_y_pos = []
                    inh_refractory_x_pos = []
                    inh_refractory_y_pos = []
                    exc_refrac_colors = []
                    exc_refrac_sizes = []
                    inh_refrac_colors = []
                    inh_refrac_sizes = []
                    for v in g.vertices():
                        if curr_exc_state[v] == exc_R:
                            exc_refractory_x_pos.append(pos[v][0])
                            exc_refractory_y_pos.append(pos[v][1])
                            exc_refrac_colors.append(state_to_rgba(curr_exc_state[v]))
                            exc_refrac_sizes.append(30)
                        elif curr_inh_state[v] == inh_R:
                            inh_refractory_x_pos.append(pos[v][0])
                            inh_refractory_y_pos.append(pos[v][1])
                            inh_refrac_colors.append(state_to_rgba(curr_inh_state[v]))
                            inh_refrac_sizes.append(20)
                    if exc_refractory_x_pos:
                        ax.scatter(exc_refractory_x_pos, exc_refractory_y_pos, c=exc_refrac_colors, # cmap='viridis', 
                                   s=exc_refrac_sizes, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=2)
                    elif inh_refractory_x_pos:
                        ax.scatter(inh_refractory_x_pos, inh_refractory_y_pos, c=inh_refrac_colors, # cmap='viridis_r', 
                                   s=inh_refrac_sizes, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=2)
                    ax.set_facecolor('white')  # Set axes background to white

                    # Plot edges with enhanced visualization
                    edges_list = list(g.edges())
                    subsample_rate = max(1, len(edges_list) // 5000000)  # Increase the number of edges shown
                    plotted_ee_edges = 0
                    plotted_ei_edges = 0
                    plotted_ie_edges = 0
                    plotted_ii_edges = 0
                    node_types = {}
                    for i, node_info in enumerate(nodes_data): # Map regular network nodes
                        node_type = node_info['type'] 
                        node_types[i] = 'Exc' if node_type.startswith('E') else 'Inh'
                    for j, input_node_info in enumerate(input_nodes_data): # Map input nodes 
                        input_node_type = input_node_info['type'] 
                        node_types[num_nodes + j] = 'Exc' if input_node_type.startswith('E') else 'Inh'

                    ee_edges = []
                    ei_edges = []
                    ie_edges = []
                    ii_edges = []
                    input_ee_edges = []
                    input_ii_edges = []
                    for e in edges_list:
                        src_idx = int(e.source())
                        tgt_idx = int(e.target())
                        src_type = node_types.get(src_idx, 'Exc')  # Default to Exc
                        tgt_type = node_types.get(tgt_idx, 'Exc')  # Default to Exc
                        is_input_edge = src_idx >= num_nodes
                        if is_input_edge:
                            if src_type == 'Exc':  # Source is excitatory
                                input_ee_edges.append(e)
                            elif src_type == 'Inh':  # Source is inhibitory
                                input_ii_edges.append(e)
                        else:
                            # Internal edges classified by both source and target
                            if src_type == 'Exc' and tgt_type == 'Exc':
                                ee_edges.append(e)
                            elif src_type == 'Exc' and tgt_type == 'Inh':
                                ei_edges.append(e)
                            elif src_type == 'Inh' and tgt_type == 'Exc':
                                ie_edges.append(e)
                            elif src_type == 'Inh' and tgt_type == 'Inh':
                                ii_edges.append(e)
                
                    if len(ee_edges) > 0:
                        subsample_rate = max(1, len(ee_edges) // 5000000)  # Adjust subsample rate for EE edges
                        plotted_ee_edges = 0
                        for edge_idx, current_edge in enumerate(ee_edges):
                            current_edge = ee_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_ee_edges < 5000000:
                                ees, eet = current_edge
                                ee_x_coords = [pos[ees][0], pos[eet][0]]
                                ee_y_coords = [pos[ees][1], pos[eet][1]]
                                src_e_state = curr_exc_state[ees]
                                dst_e_state = curr_exc_state[eet]
                                src_i_state = curr_inh_state[ees]
                                dst_i_state = curr_inh_state[eet]
                            if src_e_state == exc_I or dst_e_state == exc_I:  # If either node is active
                                ax.plot(ee_x_coords, ee_y_coords, 'red', zorder=3, alpha=0.9, linewidth=1.0, linestyle='-')
                            elif src_e_state == exc_R or dst_e_state == exc_R:  # If either node is refractory
                                ax.plot(ee_x_coords, ee_y_coords, 'blue', zorder=3, alpha=0.7, linewidth=0.7, linestyle='-')
                            elif src_e_state == exc_S or dst_e_state == exc_S:
                                ax.plot(ee_x_coords, ee_y_coords, 'green', zorder=3, alpha=0.5, linewidth=0.5, linestyle='-')
                            plotted_ee_edges += 1
                    # Plot Input_EE edges
                    if len(input_ee_edges) > 0:
                        subsample_rate = max(1, len(input_ee_edges) // 5000000)  # Adjust subsample rate for Input_EE edges
                        plotted_input_ee_edges = 0
                        for edge_idx, current_edge in enumerate(input_ee_edges):
                            current_edge = input_ee_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_input_ee_edges < 5000000:
                                ees1, eet1 = current_edge
                                input_ee_x_coords = [pos[ees1][0], pos[eet1][0]]
                                input_ee_y_coords = [pos[ees1][1], pos[eet1][1]]
                                src_e_state = curr_exc_state[ees1]
                                dst_e_state = curr_exc_state[eet1]
                                src_i_state = curr_inh_state[ees1]
                                dst_i_state = curr_inh_state[eet1]
                            if src_e_state == exc_I or dst_e_state == exc_I:
                                ax.plot(input_ee_x_coords, input_ee_y_coords, 'magenta', zorder=3, alpha=0.9, linewidth=1.0, linestyle='-')
                            elif src_e_state == exc_R or dst_e_state == exc_R:
                                ax.plot(input_ee_x_coords, input_ee_y_coords, 'lightgreen', zorder=3, alpha=0.7, linewidth=0.7, linestyle='-')
                            elif src_e_state == exc_S or dst_e_state == exc_S:
                                ax.plot(input_ee_x_coords, input_ee_y_coords, 'lightcoral', zorder=3, alpha=0.5, linewidth=0.5, linestyle='-')
                            plotted_input_ee_edges += 1            
                    # Plot II edges (Inhibitory to Inhibitory)
                    if len(ii_edges) > 0:
                        subsample_rate = max(1, len(ii_edges) // 5000000)  # Adjust subsample rate for II edges
                        plotted_ii_edges = 0
                        for edge_idx, current_edge in enumerate(ii_edges):
                            current_edge = ii_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_ii_edges < 5000000:
                                iis, iit = current_edge
                                ii_x_coords = [pos[iis][0], pos[iit][0]]
                                ii_y_coords = [pos[iis][1], pos[iit][1]]
                                src_e_state = curr_exc_state[iis]
                                dst_e_state = curr_exc_state[iit]
                                src_i_state = curr_inh_state[iis]
                                dst_i_state = curr_inh_state[iit]
                            if src_i_state == inh_I or dst_i_state == inh_I:
                                ax.plot(ii_x_coords, ii_y_coords, 'purple', zorder=3, alpha=0.9, linewidth=1.0, linestyle='-')
                            elif src_i_state == inh_R or dst_i_state == inh_R:
                                ax.plot(ii_x_coords, ii_y_coords, 'navy', zorder=3, alpha=0.7, linewidth=0.7, linestyle='-')
                            elif src_i_state == inh_S or dst_i_state == inh_S:
                                ax.plot(ii_x_coords, ii_y_coords, 'darkgreen', zorder=3, alpha=0.5, linewidth=0.5, linestyle='-')
                            plotted_ii_edges += 1
                    # Plot Input_II edges
                    if len(input_ii_edges) > 0:
                        subsample_rate = max(1, len(input_ii_edges) // 5000000)
                        plotted_input_ii_edges = 0
                        for edge_idx, current_edge in enumerate(input_ii_edges):
                            current_edge = input_ii_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_input_ii_edges < 5000000:
                                iis1, iit1 = current_edge
                                input_ii_x_coords = [pos[iis1][0], pos[iit1][0]]
                                input_ii_y_coords = [pos[iis1][1], pos[iit1][1]]
                                src_e_state = curr_exc_state[iis1]
                                dst_e_state = curr_exc_state[iit1]
                                src_i_state = curr_inh_state[iis1]
                                dst_i_state = curr_inh_state[iit1]
                            if src_i_state == inh_I or dst_i_state == inh_I:
                                ax.plot(input_ii_x_coords, input_ii_y_coords, 'cyan', zorder=3, alpha=0.9, linewidth=1.0, linestyle='-')
                            elif src_i_state == inh_R or dst_i_state == inh_R:
                                ax.plot(input_ii_x_coords, input_ii_y_coords, 'yellowgreen', zorder=3, alpha=0.7, linewidth=0.7, linestyle='-')
                            elif src_i_state == inh_S or dst_i_state == inh_S:
                                ax.plot(input_ii_x_coords, input_ii_y_coords, 'violet', zorder=3, alpha=0.5, linewidth=0.5, linestyle='-')
                            plotted_input_ii_edges += 1
                    # Plot EI edges (Excitatory to Inhibitory)
                    if len(ei_edges) > 0:
                        subsample_rate = max(1, len(ei_edges) // 5000000)  # Adjust subsample rate for EI edges
                        plotted_ei_edges = 0
                        for edge_idx, current_edge in enumerate(ei_edges):
                            current_edge = ei_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_ei_edges < 5000000:
                                eis, eit = current_edge
                                ei_x_coords = [pos[eis][0], pos[eit][0]]
                                ei_y_coords = [pos[eis][1], pos[eit][1]]
                                src_e_state = curr_exc_state[eis]
                                dst_e_state = curr_exc_state[eit]
                                src_i_state = curr_inh_state[eis]
                                dst_i_state = curr_inh_state[eit]
                            if src_e_state == exc_I or dst_i_state == inh_I:
                                ax.plot(ei_x_coords, ei_y_coords, 'darkred', zorder=3, alpha=0.9, linewidth=1.0, linestyle=':')
                            elif src_e_state == exc_R or dst_i_state == inh_R:
                                ax.plot(ei_x_coords, ei_y_coords, 'lime', zorder=3, alpha=0.7, linewidth=0.7, linestyle=':')
                            elif src_e_state == exc_S or dst_i_state == inh_S:
                                ax.plot(ei_x_coords, ei_y_coords, 'grey', zorder=3, alpha=0.5, linewidth=0.5, linestyle=':')
                            plotted_ei_edges += 1
                    # Plot IE edges (Inhibitory to Excitatory)
                    if len(ie_edges) > 0:
                        subsample_rate = max(1, len(ie_edges) // 5000000)  # Adjust subsample rate for IE edges
                        plotted_ie_edges = 0
                        for edge_idx, current_edge in enumerate(ie_edges):
                            current_edge = ie_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_ie_edges < 5000000:
                                ies, iet = current_edge
                                ie_x_coords = [pos[ies][0], pos[iet][0]]
                                ie_y_coords = [pos[ies][1], pos[iet][1]]
                                src_e_state = curr_exc_state[ies]
                                dst_e_state = curr_exc_state[iet]
                                src_i_state = curr_inh_state[ies]
                                dst_i_state = curr_inh_state[iet]
                            if src_i_state == inh_I or dst_e_state == exc_I:
                                ax.plot(ie_x_coords, ie_y_coords, 'pink', zorder=3, alpha=0.9, linewidth=1.0, linestyle=':')
                            elif src_i_state == inh_R or dst_e_state == exc_R:
                                ax.plot(ie_x_coords, ie_y_coords, 'yellow', zorder=3, alpha=0.7, linewidth=0.7, linestyle=':')
                            elif src_i_state == inh_S or dst_e_state == exc_S:
                                ax.plot(ie_x_coords, ie_y_coords, 'black', zorder=3, alpha=0.5, linewidth=0.5, linestyle=':')
                            plotted_ie_edges += 1
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='red', lw=2, label='exc_I (EE)', linestyle='-', alpha=0.9),
                        Line2D([0], [0], color='purple', lw=2, label='inh_I (II)', linestyle='-', alpha=0.9),
                        Line2D([0], [0], color='darkred', lw=2, label='exc_I -> inh_I (EI)', linestyle=':', alpha=0.9),
                        Line2D([0], [0], color='pink', lw=2, label='inh_I -> exc_I (IE)', linestyle=':', alpha=0.9),
                        
                        Line2D([0], [0], color='blue', lw=2, label='exc_R (EE)', linestyle='-', alpha=0.7),
                        Line2D([0], [0], color='navy', lw=2, label='inh_R (II)', linestyle='-', alpha=0.7),
                        Line2D([0], [0], color='lime', lw=2, label='exc_R -> inh_R (EI)', linestyle=':', alpha=0.7),
                        Line2D([0], [0], color='yellow', lw=2, label='inh_R -> exc_R (IE)', linestyle=':', alpha=0.7),
                        
                        Line2D([0], [0], color='green', lw=2, label='exc_S (EE)', linestyle='-', alpha=0.5),
                        Line2D([0], [0], color='darkgreen', lw=2, label='inh_S (II)', linestyle='-', alpha=0.5),
                        Line2D([0], [0], color='grey', lw=2, label='exc_S -> inh_S (EI)', linestyle=':', alpha=0.5),
                        Line2D([0], [0], color='black', lw=2, label='inh_S -> exc_S (IE)', linestyle=':', alpha=0.5),
                       
                        Line2D([0], [0], color='magenta', lw=2, label='exc_I (Input_EE)', linestyle='-', alpha=0.9),
                        Line2D([0], [0], color='cyan', lw=2, label='inh_I (Input_II)', linestyle='-', alpha=0.9),
                        Line2D([0], [0], color='lightgreen', lw=2, label='exc_R (Input_EE)', linestyle='-', alpha=0.7),
                        Line2D([0], [0], color='yellowgreen', lw=2, label='inh_R (Input_II)', linestyle='-', alpha=0.7),
                        Line2D([0], [0], color='lightcoral', lw=2, label='exc_S (Input_EE)', linestyle='-', alpha=0.5),
                        Line2D([0], [0], color='violet', lw=2, label='inh_S (Input_II)', linestyle='-', alpha=0.5),
                    ]
                    
                    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=8)
                    
                    ax.set_title(f'{network_name} Dynamic S->I->R->S epidemic model Frame {count}\n exc_S={exc_inactive_count} | exc_I={exc_active_count} | exc_R={exc_refractory_count} | inh_S={inh_inactive_count} | inh_I={inh_active_count} | inh_R={inh_refractory_count}\n EE={plotted_ee_edges} | EI={plotted_ei_edges} | IE={plotted_ie_edges} | II={plotted_ii_edges} | Input_EE={plotted_input_ee_edges} | Input_II={plotted_input_ii_edges}', fontsize=10)
                    ax.set_aspect('equal')
                    
                    # Set fixed axis limits to show gradual spreading effect
                    # Calculate global limits based on all positions across all frames
                    if len(all_x_positions) > 0 and len(all_y_positions) > 0:
                        x_min, x_max = min(all_x_positions), max(all_x_positions)
                        y_min, y_max = min(all_y_positions), max(all_y_positions)
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        
                        # Add fixed padding around the network
                        padding = 0.1
                        padding_x = x_range * padding
                        padding_y = y_range * padding
                        
                        # Apply fixed axis limits with padding
                        ax.set_xlim(x_min - padding_x, x_max + padding_x)
                        ax.set_ylim(y_min - padding_y, y_max + padding_y)
                    ax.grid(True, alpha=0.2)
                        
                    # Save frame as PNG file with even dimensions
                    plt.savefig(f'{network_frames_dir}/frame_{count:04d}.png', dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    
                    print(f"Saved frame {count}")
                except Exception as e:
                    print(f"Error saving frame {count}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Store current states as previous states
            for v in g.vertices():
                prev_exc_state[v] = curr_exc_state[v]
                prev_inh_state[v] = curr_inh_state[v]
            
            # Increment the counter
            count += 1
            
            # Stop after a reasonable number of frames to avoid infinite loop
            if count >= max_count:
                print("Reached frame limit, stopping animation")
                return False

            # We need to return True so that the function will be called again
            return True
        
        print("Starting animation loop...")
        # Generate frames one by one
        while update_state():
            pass
            
        print("Animation finished")
        
        # Provide instructions for combining frames
        if offscreen:
            print(f"\nFrames saved to {network_frames_dir}/ directory")
            print("\nTo combine frames into a video, you can use ffmpeg:")
            print(f"ffmpeg -framerate 10 -pattern_type glob -i '{network_frames_dir}/frame_*.png' -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p {network_name}_combined_adv.mp4")
            print("\nAlternatively, you can create an animated GIF using ImageMagick:")
            print(f"convert -delay 10 {network_frames_dir}/*.png {network_name}_combined_adv.gif")
            print("\nOr create an animated GIF with optimization:")
            print(f"convert -delay 10 -loop 0 {network_frames_dir}/*.png -scale 800x800 -coalesce -fuzz 5% -layers Optimize {network_name}_combined_adv.gif")
            print("\nThis will create an animated GIF from the saved frames.")
            print("\nTo run without saving frames, use: python 66_4_gt_combined_ani.py no_offscreen")
        else:
            print(f"\nRunning {network_name} without saving frames.")
            print("To save frames, run without arguments: python 66_4_gt_combined_ani.py")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files: {e}")
        print("Make sure the gt/params directory contains the required files.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()