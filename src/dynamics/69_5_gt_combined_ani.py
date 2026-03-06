#!/usr/bin/env python3

"""
Combined Network animation using graph-tool and matplotlib.
This script creates an animated visualization of neural networks showing both
dynamic network topology changes and node state transitions.
"""
import os
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "12"
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
import scipy.stats
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
cmap_S = plt.get_cmap('tab20c')    # inactive state colormap
cmap_I = plt.get_cmap('coolwarm')  # Active state colormap  
cmap_R = plt.get_cmap('viridis')  # Refractory state colormap

# Get specific colors from colormaps (using appropriate positions)
S = list(cmap_S(0.5))[:4]          # inactive state - tab20c at position 0.1
I = list(cmap_I(0.5))[:4]          # Active state - coolwarm at position 0.8
R = list(cmap_R(0.5))[:4]          # Refractory state - viridis at position 0.2


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
    
    if approx_equal(state_value, S):  # S - inactive (from tab20c colormap)
        return S
    elif approx_equal(state_value, I):  # I - Active (from coolwarn colormap)
        return I
    elif approx_equal(state_value, R):  # R - Refractory (from viridis colormap)
        return R

def main():
    network_name = "max_CTC_plus"  # Select network to visualize
    data_dir = "gt/params"
    print(f"Loading network: {network_name}")
    try:
        # Load data
        nodes_data, edges_data, num_nodes, input_nodes_data, input_edges_data, num_input_nodes = load_network_data(network_name, data_dir)
        
        g = price_network(num_nodes + num_input_nodes, c=0.8, directed=False)
        ini_state = AxelrodState(g, f=10, q=30, r=int(0.005))
        
        node_id_to_index = {}
        for i, node_info in enumerate(nodes_data):
            node_id_to_index[node_info['component']] = i
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
        input_edge_count = 0
        for input_edge_info in input_edges_data:
            src_id = input_edge_info['source']
            tgt_id = input_edge_info['target']
            if src_id in input_id_to_index and tgt_id in node_id_to_index:
                src_idx = input_id_to_index[src_id]
                tgt_idx = node_id_to_index[tgt_id]
                # g.add_edge(src_idx, tgt_idx)
            input_edge_count += 1
        
        # Generate layout using graph-tool
        pos = sfdp_layout(g, K=K, cooling_step=0.99, C=100, multilevel=True, R=20, gamma=1)      
        # Initialize SIRS state properties
        ini_state = g.new_vertex_property("vector<double>")
        for v in g.vertices():
            ini_state[v] = S
        curr_state = g.new_vertex_property("vector<double>")
        prev_state = g.new_vertex_property("vector<double>")
        newly_transmited = g.new_vertex_property("bool")
        refractory = g.new_vertex_property("bool")
        for v in g.vertices():
            curr_state[v] = S  # Start all nodes in inactive state
            prev_state[v] = ini_state[v] = S
        # List of edges for dynamic rewiring
        edges = list(g.edges())
        max_count = 300
        # Check if we should disable offscreen rendering
        no_offscreen = sys.argv[1] == "no_offscreen" if len(sys.argv) > 1 else False
        offscreen = not no_offscreen
        network_frames_dir = f"./frames/{network_name}/combined"
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
            newly_transmited.a = False
            refractory.a = False
            # Count states for debugging
            inactive_count = 0
            active_count = 0
            refractory_count = 0
            
            # Count current states before updating
            for v in g.vertices():
                if curr_state[v] == S:  # inactive
                    inactive_count += 1
                elif curr_state[v] == I:  # Active
                    active_count += 1
                elif curr_state[v] == R:  # Refractory
                    refractory_count += 1
            print(f"Frame {count}: Inactive={inactive_count}, Active={active_count}, Refractory={refractory_count}")
            
            # Randomly make a few nodes initially infected to start the simulation
            if count == 0 and active_count == 0:
                # Infect a small number of random nodes to start the simulation
                vs = list(g.vertices())
                shuffle(vs)
                nodes = [v for v in vs if vertex_index_map[v] in node_id_to_index]
                input_nodes = [v for v in vs if vertex_index_map[v] in input_id_to_index]

                num_initial_infections = min(50, len(nodes), len(input_nodes))  # Infect at most 25 nodes
                for i in range(num_initial_infections):
                    if i < len(nodes):
                        curr_state[nodes[i]] = I
                        newly_transmited[nodes[i]] = True
                    if i < len(input_nodes):
                        curr_state[input_nodes[i]] = I
                        newly_transmited[input_nodes[i]] = True

                    
            # Update node states (SIRS model)
            vs = list(g.vertices())
            shuffle(vs)
            for v in vs:
                if curr_state[v] == I:  # Active
                    # With probability r, recover (become R)
                    if random() < r:
                        curr_state[v] = R
                    else:
                        # Try to infect inactive neighbors
                        ns = list(v.out_neighbors())
                        if len(ns) > 0:
                            w = ns[randint(0, len(ns))]  # choose a random neighbor
                            # If neighbor is inactive, infect it with some probability
                            if curr_state[w] == S and random() < 0.5:  # Infection probability
                                curr_state[w] = I
                                
                elif curr_state[v] == S:  # inactive
                    # With probability x, spontaneously become infected
                    if random() < x:
                        curr_state[v] = I
                        
                elif curr_state[v] == R:  # Refractory
                    # With probability s, become inactive again
                    if random() < s:
                        curr_state[v] = S

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
                        t1.out_degree() > 1 and           # no isolated vertices
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
                        prev_state[v] = curr_state[v]
                    return True
                
                # Save frame as PNG file
                try:
                    # Create a matplotlib figure for this frame with even dimensions
                    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=200)
                    fig.patch.set_facecolor('white')

                    # Extract positions for plotting
                    x_pos = [pos[v][0] for v in g.vertices()]
                    y_pos = [pos[v][1] for v in g.vertices()]
                    colors = []
                    sizes = []
                    for v in g.vertices():
                        colors.append(state_to_rgba(S))
                        sizes.append(15)
                    ax.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=0.5)

                    # Draw halos around active vertices for better visibility
                    active_x_pos = []
                    active_y_pos = []
                    active_colors = []
                    acitve_sizes = []
                    for v in g.vertices():
                        if curr_state[v] == I:  # If vertex is active (infected)
                            active_x_pos.append(pos[v][0])
                            active_y_pos.append(pos[v][1])
                            active_colors.append(state_to_rgba(I))
                            acitve_sizes.append(30)
                    if active_x_pos:
                        ax.scatter(active_x_pos, active_y_pos, c=active_colors, # cmap='autumn', 
                                   s=acitve_sizes, alpha=0.5, edgecolors='white', linewidth=0.5)

                    inactive_x_pos = []
                    inactive_y_pos = []
                    inactive_colors = []
                    inactive_sizes = []
                    for v in g.vertices():
                        if curr_state[v] == S:  # If vertex is inactive
                            inactive_x_pos.append(pos[v][0])
                            inactive_y_pos.append(pos[v][1])
                            inactive_colors.append(state_to_rgba(S))
                            inactive_sizes.append(1)
                    if inactive_x_pos:
                        # sizes = np.random.uniform(1, 5, len(inactive_x_pos))
                        # colors = np.random.uniform(0.1, 0.5, len(inactive_x_pos))
                        ax.scatter(inactive_x_pos, inactive_y_pos, c=inactive_colors, # cmap='tab20c', 
                                   s=inactive_sizes, alpha=0.5, edgecolors='white', linewidth=0.5)

                    refractory_x_pos = []
                    refractory_y_pos = []
                    refrac_colors = []
                    refrac_sizes = []
                    for v in g.vertices():
                        if curr_state[v] == R:
                            refractory_x_pos.append(pos[v][0])
                            refractory_y_pos.append(pos[v][1])
                            refrac_colors.append(state_to_rgba(R))
                            refrac_sizes.append(15)
                    if refractory_x_pos:
                        # sizes = np.random.uniform(5, 15, len(refractory_x_pos))
                        # colors = np.random.uniform(0.1, 0.3,len(refractory_x_pos))
                        ax.scatter(refractory_x_pos, refractory_y_pos, c=refrac_colors, # cmap='viridis', 
                                   s=refrac_sizes, alpha=0.7, edgecolors='white', linewidth=0.5)
                    
                    # Plot edges with enhanced visualization
                    edges_list = list(g.edges())
                    subsample_rate = max(1, len(edges_list) // 5000000)  # Increase the number of edges shown
                    plotted_edges = 0
                    plotted_input_edges = 0 

                    renewal_edges = []
                    renewal_input_edges = []
                    for e in g.edges():
                        src_idx = int(e.source())
                        tgt_idx = int(e.target())
                        if src_idx < num_nodes and tgt_idx < num_nodes:
                            renewal_edges.append(e)
                        elif src_idx >= num_nodes and tgt_idx < num_nodes:
                            renewal_input_edges.append(e)
                       
                    if len(renewal_edges) > 0:
                        subsample_rate = max(1, len(renewal_edges) // 5000000)  # the more edges the delicate
                        plotted_edges = 0
                        for edge_idx, current_edge in enumerate(renewal_edges):
                            current_edge = renewal_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_edges < 5000000:  # Cap at 5000000 edges
                                s1, t1 = current_edge
                                x_coords = [pos[s1][0], pos[t1][0]]
                                y_coords = [pos[s1][1], pos[t1][1]]
                                src_state = curr_state[s1]
                                dst_state = curr_state[t1]
                                if src_state == I or dst_state == I:  # If either node is active
                                    ax.plot(x_coords, y_coords, 'red', zorder=3, alpha=0.7, linewidth=1.0, linestyle='--')
                                elif src_state == R or dst_state == R:  # If either node is refractory
                                    ax.plot(x_coords, y_coords, 'blue', zorder=3, alpha=0.5, linewidth=0.7, linestyle='-')
                                elif src_state == S or dst_state == S:
                                    ax.plot(x_coords, y_coords, 'green', zorder=3, alpha=0.3, linewidth=0.5, linestyle=':')
                            plotted_edges += 1
                    if len(renewal_input_edges) > 0:
                        subsample_rate = max(1, len(renewal_input_edges) // 5000000)  # the more edges the delicate
                        plotted_input_edges = 0
                        for edge_idx, current_edge in enumerate(renewal_input_edges):
                            current_edge = renewal_input_edges[edge_idx]
                            if edge_idx % subsample_rate == 0 and plotted_input_edges < 5000000 :  # Cap at 100000 edges
                                s1, t1 = current_edge
                                input_x_coords = [pos[s1][0], pos[t1][0]]
                                input_y_coords = [pos[s1][1], pos[t1][1]]
                                src_state = curr_state[s1]
                                dst_state = curr_state[t1]
                                if src_state == I or dst_state == I:  # If either node is active
                                    ax.plot(input_x_coords, input_y_coords, 'purple', zorder=4, alpha=0.7, linewidth=1.0, linestyle='--')
                                elif src_state == R or dst_state == R:  # If either node is refractory
                                    ax.plot(input_x_coords, input_y_coords, 'yellow', zorder=4,alpha=0.5, linewidth=0.7, linestyle='-')
                                elif src_state == S or dst_state == S:
                                    ax.plot(input_x_coords, input_y_coords, 'darkgreen', zorder=4,alpha=0.3, linewidth=0.5, linestyle=':')
                            plotted_input_edges += 1
            
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='red', lw=2, label='I (edges)', linestyle='--', alpha=0.7),
                        Line2D([0], [0], color='blue', lw=2, label='R (edges)', linestyle='-', alpha=0.5),
                        Line2D([0], [0], color='green', lw=2, label='S (edges)', linestyle=':', alpha=0.3),
                        Line2D([0], [0], color='purple', lw=2, label='I (input edges)', linestyle='--', alpha=0.7),
                        Line2D([0], [0], color='yellow', lw=2, label='R (input edges)', linestyle='-', alpha=0.5),
                        Line2D([0], [0], color='darkgreen', lw=2, label='S (input edges)', linestyle=':', alpha=0.3),
                    ]
                    ax.legend(handles=legend_elements, loc='upper right',  ncol=2, frameon=True, fontsize=12)
                    ax.set_title(f'{network_name}\n Dynamic S->I->R->S epidemic model Frame {count}\n Inactive: {inactive_count} | Active: {active_count} | Refractory: {refractory_count}\n Edges: {len(renewal_edges)} | Input Edges: {len(renewal_input_edges)}', fontsize=14)
                    ax.set_aspect('equal')
                    
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
                    plt.savefig(f'{network_frames_dir}/frame_{count:04d}.png', dpi=200, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    
                    print(f"Saved frame {count}")
                except Exception as e:
                    print(f"Error saving frame {count}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Store current states as previous states
            for v in g.vertices():
                prev_state[v] = curr_state[v]
            
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
            print(f"ffmpeg -framerate 10 -pattern_type glob -i '{network_frames_dir}/frame_*.png' -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p {network_name}_combined.mp4")
            print("\nAlternatively, you can create an animated GIF using ImageMagick:")
            print(f"convert -delay 10 {network_frames_dir}/*.png {network_name}_combined.gif")
            print("\nOr create an animated GIF with optimization:")
            print(f"convert -delay 10 -loop 0 {network_frames_dir}/*.png -scale 800x800 -coalesce -fuzz 5% -layers Optimize {network_name}_combined.gif")
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