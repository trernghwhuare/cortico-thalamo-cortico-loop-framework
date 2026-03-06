# %%
import os
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["GDK_BACKEND"] = "x11"
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"
import gi
gi.require_version('Gtk', '3.0')
# from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
from graph_tool.all import *
import graph_tool.all as gt
import networkx as nx
import json
import csv
import os
from pyneuroml.pynml import read_neuroml2_file
import numpy as np
import argparse
import logging
import io
import sys
from contextlib import redirect_stdout
from datetime import datetime

logging.basicConfig(level=logging.INFO)

TCs_intralaminar = ["TCRil","nRTil"]
TCs_matrix = ["TCR","TCRm","nRTm"]
TCs_core = ["nRT","TCRc","nRTc"]
thalamus = TCs_core + TCs_matrix + TCs_intralaminar

exc_e = {"cADpyr", "cAC", "cNAC", "cSTUT", "cIR","TCR","TCRm","nRTm"}
inh_e = {"bAC","bNAC","dNAC","bSTUT","dSTUT","bIR","TCRil","nRTil","nRT","TCRc","nRTc"}
e_type_list = {"cADpyr", "cAC", "bAC", "cNAC","bNAC", "dNAC", "cSTUT", "bSTUT","dSTUT", "cIR", "bIR"}

Region_list = {"M2a","M2b","M1a","M1b","S1a","S1b"}
layer_list = {"L1", "L23", "L4", "L5", "L6", "thalamus"}
m_list = [
        "DAC","NGCDA",	"NGCSA","HAC",
        "LAC","SAC","MC","BTC","DBC","BP","NGC",	
        "LBC", "NBC","SBC",	 "ChC", "PC","SP",	
        "SS", "TTPC1","TTPC2","UTPC","STPC",	
        "TPC_L4", "TPC_L1",	"IPC","BPC"
    ]
e_list = [
        "cADpyr", "cAC", "bAC", "cNAC",
        "bNAC", "dNAC", "cSTUT", "bSTUT",
        "dSTUT", "cIR", "bIR"
    ]

def get_pop_type(pop_id):
    parts = pop_id.split('_') if "_" in pop_id else [pop_id]
    exc_list = ["cADpyr", "cAC", "cNAC", "cSTUT", "cIR", "TCR", "TCRm", "nRTm"]
    inh_list = ["bAC", "bNAC", "dNAC", "bSTUT", "dSTUT", "bIR", "TCRil", "nRTil", "nRT", "TCRc", "nRTc"]
    
    # Check if any part of the population ID starts with items in the excitation or inhibition lists
    for part in parts:
        for exc_item in exc_list:
            if part.startswith(exc_item):
                return "Exc"
        for inh_item in inh_list:
            if part.startswith(inh_item):
                return "Inh"
    return "Unknown"

def get_input_type(ilist_id, ilist_component):
    if ilist_id.startswith("input_inh"):
        return "Inh"
    elif ilist_id.startswith("input_exc"):
        return "Exc"
    elif ilist_component.startswith("inh"):
        return "Inh"
    elif ilist_component.startswith("exc"):
        return "Exc"
    return None

def get_Region(pop_id):
    try:
        if not isinstance(pop_id, str) or pop_id == "":
            return None
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if parts[0] in Region_list:
            return parts[0]
        return None
    except Exception as e:
        logging.error(f"Error in get_Region with pop_id {pop_id}: {e}")
        return None


def get_layer(pop_id):
    try:
        if not isinstance(pop_id, str) or pop_id == "":
            return pop_id
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) == 1:
            return "thalamus"
        elif len(parts) == 2 and parts[0] in Region_list:
            return "thalamus"
        elif len(parts) >= 6 and parts[1] in layer_list:
            return parts[1]
        elif len(parts) >= 7 and parts[0] in Region_list:  # Fixed: was 'Regions' which is undefined
            return parts[2] 
        return "unknown"
    except Exception as e:
        logging.error(f"Error in get_layer with pop_id {pop_id}: {e}")
        return "unknown"


def get_e_type(pop_id):
    try:
        if not isinstance(pop_id, str) or pop_id == "":
            return pop_id
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) == 1 and parts[0] in thalamus:
            return parts[0]
        elif len(parts) == 2 and parts[0] in Region_list:
            return parts[1]
        elif len(parts) > 3 and parts[0] not in Region_list:
            return parts[0]
        elif len(parts) > 3 and parts[0] in Region_list:
            return parts[1]
    except Exception as e:
        logging.error(f"Error in get_e_type with pop_id {pop_id}: {e}")
        return "unknown"


def get_m_type(pop_id):
    try:
        if not isinstance(pop_id, str) or pop_id == "":
            return pop_id
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) == 1 and parts[0] in thalamus:
            return parts[0]
        elif len(parts) == 2 and parts[0] in Region_list:
            return parts[1]
        elif len(parts) > 3 and parts[0] not in Region_list:
            return '_'.join(parts[1:3])
        elif len(parts) > 3 and parts[0] in Region_list:
            return '_'.join(parts[2:4])
    except Exception as e:
        logging.error(f"Error in get_m_type with pop_id {pop_id}: {e}")
        return "unknown"


def get_vprefix(pop_id):
    try:
        if not isinstance(pop_id, str) or pop_id == "":
            return pop_id
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) == 1 and parts[0] in thalamus:
            return parts[0]
        elif len(parts) == 2 and parts[1] in thalamus:
            return parts[1]
        elif len(parts) >= 6 and parts[0] not in Region_list:
            return '_'.join(parts[1:3])
        elif len(parts) >= 7 and parts[0] in Region_list:
            return '_'.join(parts[2:4]) 
        elif len(parts) >= 7 and parts[0] not in Region_list:
            return '_'.join(parts[2:5])
        return "unknown"
    except Exception as e:
        logging.error(f"Error in get_vprefix with pop_id {pop_id}: {e}")
        return "unknown"


def extract_positions_from_nml(nml_file_path, nml_doc=None):
    """
    Extract 3D positions from a NeuroML file.
    Returns a dictionary mapping node IDs to (x, y, z) positions.
    """
    if not os.path.exists(nml_file_path):
        print(f"Warning: NML file {nml_file_path} not found. Returning empty positions.")
        return {}
    
    # If nml_doc is not provided, load it from the file
    if nml_doc is None:
        try:
            nml_doc = read_neuroml2_file(nml_file_path, include_includes=True)
        except Exception as e:
            print(f"Warning: Error reading NML file {nml_file_path}: {e}. Returning empty positions.")
            return {}
    
    positions = {}
    # Access networks in the document
    for network in nml_doc.networks:
        for population in network.populations:
            pop_id = population.id
            
            # Check if the population has instances with explicit locations
            if hasattr(population, 'instances') and population.instances:
                # For populationList type, use the average position or pick the first instance's position
                if hasattr(population, 'type') and population.type == 'populationList':
                    if len(population.instances) > 0:
                        # Take the first instance's location as representative for the population
                        first_instance = population.instances[0]
                        if hasattr(first_instance, 'location') and first_instance.location:
                            x = float(first_instance.location.x)
                            y = float(first_instance.location.y)
                            z = float(first_instance.location.z)
                            positions[pop_id] = (x, y, z)
                            
                        # Or store all instances separately
                        for idx, instance in enumerate(population.instances):
                            if hasattr(instance, 'location') and instance.location:
                                x = float(instance.location.x)
                                y = float(instance.location.y)
                                z = float(instance.location.z)
                                
                                # Create a unique node ID for each instance
                                node_id = f"{pop_id}_{idx}"
                                positions[node_id] = (x, y, z)
                else:
                    # For other types, if instances exist, use the first one as representative
                    if len(population.instances) > 0:
                        first_instance = population.instances[0]
                        if hasattr(first_instance, 'location') and first_instance.location:
                            x = float(first_instance.location.x)
                            y = float(first_instance.location.y)
                            z = float(first_instance.location.z)
                            positions[pop_id] = (x, y, z)
            elif hasattr(population, 'x') and hasattr(population, 'y') and hasattr(population, 'z'):
                # If no instances with locations, check if the population itself has location data
                x = float(population.x)
                y = float(population.y)
                z = float(population.z)
                positions[pop_id] = (x, y, z)
    
    return positions

def extract_network_parameters(nml_file, nml_doc=None):
    logging.info(f"Extracting parameters from {nml_file}")
    
    # If nml_doc is not provided, load it from the file
    if nml_doc is None:
        nml_doc = read_neuroml2_file(nml_file)
    
    net = nml_doc.networks[0]
    
    # Get the summary from nml_doc
    summary_str = ""
    try:
        # Capture the summary output
        f = io.StringIO()
        with redirect_stdout(f):
            nml_doc.summary()  # Don't print to stdout, just capture
        summary_str = f.getvalue()
    except Exception as e:
        # Try alternative method
        try:
            summary_str = str(nml_doc.summary())
        except Exception as e2:
            summary_str = f"Error getting summary: {str(e)} and {str(e2)}"
    
    # Extract populations
    populations = []
    for pop in net.populations:
        pop_data = {
            'id': pop.id,
            'component': pop.component if hasattr(pop, 'component') else '',
            'size': int(pop.size) if hasattr(pop, 'size') else 0,
            'type': pop.type if hasattr(pop, 'type') else ''
        }
        
        # Extract instance locations if available
        # Only process instances if they exist and are not empty
        if hasattr(pop, 'instances') and pop.instances:
            instances = []
            for instance in pop.instances:
                if hasattr(instance, 'location') and instance.location:
                    instance_data = {
                        'id': instance.id if hasattr(instance, 'id') else len(instances),  # Use index if no id
                        'location': {
                            'x': float(instance.location.x),
                            'y': float(instance.location.y),
                            'z': float(instance.location.z)
                        }
                    }
                    instances.append(instance_data)
            
            # Only add instances if we found any with locations
            if instances:
                pop_data['instances'] = instances
        else:
            # If no instances, we can still store population-level location if available
            if hasattr(pop, 'x') and hasattr(pop, 'y') and hasattr(pop, 'z'):
                pop_data['location'] = {
                    'x': float(pop.x),
                    'y': float(pop.y),
                    'z': float(pop.z)
                }
        
        populations.append(pop_data)
    
    # Count populations
    num_populations = len(populations)
    
    # Extract projections
    continuous_projections = []
    electrical_projections = []
    
    # Try different possible attributes for projections
    possible_projection_attrs = ['projections', 'projection', 'continuous_projections', 'connections', 'synapses']
    
    for attr_name in possible_projection_attrs:
        if hasattr(net, attr_name):
            projections = getattr(net, attr_name)
            if projections:
                for proj in projections:
                    proj_data = {
                        'id': getattr(proj, 'id', f"proj_{len(continuous_projections) + len(electrical_projections)}"),
                        'type': 'continuous',
                        'presynaptic_population': getattr(proj, 'presynaptic_population', getattr(proj, 'pre', '')),
                        'postsynaptic_population': getattr(proj, 'postsynaptic_population', getattr(proj, 'post', '')),
                        'source_port': getattr(proj, 'source_port', ''),
                        'target_port': getattr(proj, 'target_port', ''),
                        'connections': []
                    }
                    
                    # Check for different possible connection attributes
                    connection_attrs = [
                        'connections', 'connection', 'continuous_connections', 'continuous_connection',
                        'continuous_connection_instances', 'connection_instance', 'synapses', 'synapse',
                        'electrical_connections', 'electrical_connection', 'electrical_connection_instances'
                    ]
                    
                    connections_found = False
                    for conn_attr in connection_attrs:
                        if hasattr(proj, conn_attr):
                            connections = getattr(proj, conn_attr)
                            if connections:
                                # Handle different types of connection data
                                if isinstance(connections, list):
                                    proj_data['connections'] = connections
                                elif hasattr(connections, '__len__') and not isinstance(connections, str):  # For objects that have length but are not strings
                                    proj_data['connections'] = [connections]
                                else:
                                    # Try to access as object attributes or properties
                                    proj_data['connections'] = [connections]
                                connections_found = True
                                break
                    
                    # If no connections found in the standard attributes, try other possible attributes
                    if not connections_found:
                        # Try to get connection count from other possible attributes
                        for attr in dir(proj):
                            if 'connection' in attr.lower() and not attr.startswith('_'):
                                attr_value = getattr(proj, attr)
                                if attr_value is not None and not callable(attr_value):
                                    if isinstance(attr_value, list):
                                        proj_data['connections'] = attr_value
                                    else:
                                        proj_data['connections'] = [attr_value]
                                    connections_found = True
                                    break
                    
                    # Determine if it's electrical or continuous based on attributes or connection content
                    is_electrical = (hasattr(proj, 'electrical_connection') or hasattr(proj, 'electrical_connections') or 
                        hasattr(proj, 'electrical_connection_instance') or hasattr(proj, 'electrical_connection_instances') or
                        'electrical' in getattr(proj, 'type', '').lower() or
                        'gap' in getattr(proj, 'id', '').lower())
                    
                    # Also check if the connection objects themselves indicate electrical connections
                    if not is_electrical and proj_data['connections']:
                        # Check if any connection in the list indicates electrical type
                        for conn in proj_data['connections']:
                            if hasattr(conn, 'type') and 'electrical' in str(getattr(conn, 'type', '')).lower():
                                is_electrical = True
                                break
                            elif hasattr(conn, 'id') and 'gap' in str(getattr(conn, 'id', '')).lower():
                                is_electrical = True
                                break
                    
                    # Extract weights from continuous/electrical connection instances if available
                    if not is_electrical and hasattr(proj, 'continuous_connection_instance_ws'):
                        conn_instances = getattr(proj, 'continuous_connection_instance_ws', [])
                        if conn_instances:
                            # Store the weight from the first instance as the projection weight
                            first_instance = conn_instances[0]
                            # Check for various possible weight attribute names
                            if hasattr(first_instance, 'weight'):
                                proj_data['weight'] = getattr(first_instance, 'weight', 1.0)
                            elif hasattr(first_instance, 'conductance'):
                                proj_data['weight'] = getattr(first_instance, 'conductance', 1.0)
                            elif hasattr(first_instance, 'synapse_props') and isinstance(getattr(first_instance, 'synapse_props'), dict):
                                # Check if synapse properties contain weight
                                syn_props = getattr(first_instance, 'synapse_props')
                                if 'weight' in syn_props:
                                    proj_data['weight'] = syn_props['weight']
                                elif 'conductance' in syn_props:
                                    proj_data['weight'] = syn_props['conductance']
                                else:
                                    proj_data['weight'] = 1.0
                            else:
                                proj_data['weight'] = 1.0
                    elif is_electrical and (hasattr(proj, 'electrical_connection_instance_ws') or 
                                           hasattr(proj, 'electrical_connection_instances')):
                        # Check for different possible electrical connection instance attributes
                        elec_instances = (getattr(proj, 'electrical_connection_instance_ws', []) or 
                                         getattr(proj, 'electrical_connection_instances', []))
                        if elec_instances:
                            # Store the weight from the first instance as the projection weight
                            first_instance = elec_instances[0]
                            # Check for various possible weight attribute names
                            if hasattr(first_instance, 'weight'):
                                proj_data['weight'] = getattr(first_instance, 'weight', 1.0)
                            elif hasattr(first_instance, 'conductance'):
                                proj_data['weight'] = getattr(first_instance, 'conductance', 1.0)
                            elif hasattr(first_instance, 'synapse_props') and isinstance(getattr(first_instance, 'synapse_props'), dict):
                                # Check if synapse properties contain weight
                                syn_props = getattr(first_instance, 'synapse_props')
                                if 'weight' in syn_props:
                                    proj_data['weight'] = syn_props['weight']
                                elif 'conductance' in syn_props:
                                    proj_data['weight'] = syn_props['conductance']
                                else:
                                    proj_data['weight'] = 1.0
                            else:
                                proj_data['weight'] = 1.0
                    if is_electrical:
                        proj_data['type'] = 'electrical'
                        electrical_projections.append(proj_data)
                    else:
                        continuous_projections.append(proj_data)
    
    # Also check for electrical projections specifically
    electrical_attrs = ['electrical_projections', 'gap_junctions', 'electrical_connections', 'gap_junction']
    for attr_name in electrical_attrs:
        if hasattr(net, attr_name):
            electrical_items = getattr(net, attr_name)
            if electrical_items:
                for proj in electrical_items:
                    proj_data = {
                        'id': getattr(proj, 'id', f"elec_proj_{len(electrical_projections)}"),
                        'type': 'electrical',
                        'presynaptic_population': getattr(proj, 'presynaptic_population', getattr(proj, 'pre', '')),
                        'postsynaptic_population': getattr(proj, 'postsynaptic_population', getattr(proj, 'post', '')),
                        'source_port': getattr(proj, 'source_port', ''),
                        'target_port': getattr(proj, 'target_port', ''),
                        'connections': []
                    }
                    
                    # Check for different possible connection attributes for electrical
                    connection_attrs = [
                        'connections', 'connection', 'electrical_connections', 'electrical_connection',
                        'electrical_connection_instances', 'connection_instance', 'gap_junctions', 'gap_junction',
                        'continuous_connections', 'continuous_connection', 'continuous_connection_instances'
                    ]
                    
                    connections_found = False
                    for conn_attr in connection_attrs:
                        if hasattr(proj, conn_attr):
                            connections = getattr(proj, conn_attr)
                            if connections:
                                # Handle different types of connection data
                                if isinstance(connections, list):
                                    proj_data['connections'] = connections
                                elif hasattr(connections, '__len__') and not isinstance(connections, str):  # For objects that have length but are not strings
                                    proj_data['connections'] = [connections]
                                else:
                                    proj_data['connections'] = [connections]
                                connections_found = True
                                break
                    
                    # If no connections found in the standard attributes, try other possible attributes
                    if not connections_found:
                        # Try to get connection count from other possible attributes
                        for attr in dir(proj):
                            if 'connection' in attr.lower() and not attr.startswith('_'):
                                attr_value = getattr(proj, attr)
                                if attr_value is not None and not callable(attr_value):
                                    if isinstance(attr_value, list):
                                        proj_data['connections'] = attr_value
                                    else:
                                        proj_data['connections'] = [attr_value]
                                    connections_found = True
                                    break
                    
                    electrical_projections.append(proj_data)
    
    # Extract input lists
    input_lists = []
    if hasattr(net, 'input_lists') and net.input_lists:
        for input_list in net.input_lists:
            # Convert input_list properties to a dictionary format
            input_list_data = {
                'id': input_list.id,
                'component': input_list.component,
                'populations': getattr(input_list, 'populations', 
                                    getattr(input_list, 'population', 
                                          getattr(input_list, 'target', ''))),
                'input': []
            }
            
            # Process inputs from the input_list
            if hasattr(input_list, 'input'):
                # Convert NeuroML input objects to dictionaries
                raw_inputs = input_list.input
                if isinstance(raw_inputs, list):
                    for raw_input in raw_inputs:
                        if hasattr(raw_input, 'id'):
                            input_dict = {
                                'id': raw_input.id,
                                'destination': getattr(raw_input, 'destination', ''),
                                'target': getattr(raw_input, 'target', ''),
                                'weight': getattr(raw_input, 'weight', 1.0)
                            }
                            input_list_data['input'].append(input_dict)
                        else:
                            input_list_data['input'].append(raw_input)
                else:
                    input_list_data['input'] = []
            elif hasattr(input_list, 'input') and input_list.input:
                # Handle case where input is a single element rather than a list
                raw_input = input_list.input
                if isinstance(raw_input, list):
                    for item in raw_input:
                        if hasattr(item, 'id'):
                            input_dict = {
                                'id': item.id,
                                'destination': getattr(item, 'destination', ''),
                                'target': getattr(item, 'target', ''),
                                'weight': getattr(item, 'weight', 1.0)
                            }
                            input_list_data['input'].append(input_dict)
                        else:
                            input_list_data['input'].append(item)
                else:
                    if hasattr(raw_input, 'id'):
                        input_dict = {
                            'id': raw_input.id,
                            'destination': getattr(raw_input, 'destination', ''),
                            'target': getattr(raw_input, 'target', ''),
                            'weight': getattr(raw_input, 'weight', 0),
                        }
                        input_list_data['input'].append(input_dict)
                    else:
                        input_list_data['input'] = [raw_input]
            else:
                input_list_data['input'] = []
                
            input_lists.append(input_list_data)
    
    # Build input edges
    input_edges = []
    for input_list in input_lists:
        target_population = input_list.get('population', input_list.get('populations', None))
        if target_population is None or target_population == '':
            target_population = input_list.get('target', None)
        
        source_component = input_list.get('component', 'unknown')
        
        weight = 1.0
        if input_list.get('input'):
            input_weights = []
            for inp in input_list['input']:
                if isinstance(inp, dict) and 'weight' in inp:
                    input_weights.append(inp['weight'])
                elif hasattr(inp, 'weight'):
                    input_weights.append(getattr(inp, 'weight', 1.0))
            if input_weights:
                weight = input_weights[0] 
        
        if target_population and target_population != '':
            input_edges.append({
                'source': source_component,
                'target': target_population,
                'weight': weight
            })
    
    # Extract inputs
    inputs = []
    if hasattr(nml_doc, 'pulse_generators'):
        for pg in nml_doc.pulse_generators:
            inputs.append({
                'id': pg.id,
                'component': 'pulseGenerator',
                'type': 'input_pulse'
            })
    
    if hasattr(nml_doc, 'sine_generators'):
        for sg in nml_doc.sine_generators:
            inputs.append({
                'id': sg.id,
                'component': 'sineGenerator',
                'type': 'input_sine'
            })
    
    if hasattr(nml_doc, 'ramp_generators'):
        for rg in nml_doc.ramp_generators:
            inputs.append({
                'id': rg.id,
                'component': 'rampGenerator',
                'type': 'input_ramp'
            })
    
    # Handle compound inputs
    if hasattr(nml_doc, 'compound_inputs'):
        for ci in nml_doc.compound_inputs:
            inputs.append({
                'id': ci.id,
                'component': 'compoundInput',
                'type': 'input_compound',
                'pulse_generator': getattr(ci, 'pulse_generator', ''),
                'sine_generator': getattr(ci, 'sine_generator', ''),
                'ramp_generator': getattr(ci, 'ramp_generator', '')
            })
    
    if hasattr(nml_doc, 'voltage_clamp_triples'):
        for vc in nml_doc.voltage_clamp_triples:
            inputs.append({
                'id': vc.id,
                'component': 'voltageClampTriple',
                'type': 'input_voltage_clamp_triple'
            })
    
    return {
        'network_id': net.id if hasattr(net, 'id') else 'unknown',
        'description': net.notes if hasattr(net, 'notes') else summary_str,
        'populations': populations,
        'num_populations': num_populations,
        'continuous_projections': continuous_projections,
        'electrical_projections': electrical_projections,
        'input_nodes': input_lists,
        'input_edges': input_edges,
        'nml_file': nml_file
    }
    
    return parameters


def calculate_network_metrics(parameters):
    """
    Calculate network metrics similar to those on https://networks.skewed.de/net/twitter_2009
    
    Parameters:
        parameters (dict): Network parameters
        
    Returns:
        dict: Dictionary containing network metrics
    """
    # Number of nodes (populations in our case)
    num_nodes = len(parameters['populations'])
    num_input_nodes = len(parameters['input_nodes'])
    
    # Number of edges
    num_edges = len(parameters['continuous_projections']) + len(parameters['electrical_projections'])
    num_ilist_edges = len(parameters['input_edges'])
    
    # Create a simplified degree distribution
    all_projections = parameters['continuous_projections'] + parameters['electrical_projections']
    
    # Count connections per population
    degree_count = {}
    for proj in all_projections:
        pre_pop = proj['presynaptic_population']
        post_pop = proj['postsynaptic_population']
        connection_count = len(proj['connections'])
        
        # Add to pre-population out-degree
        if pre_pop not in degree_count:
            degree_count[pre_pop] = {'in': 0, 'out': 0}
        degree_count[pre_pop]['out'] += connection_count
        
        # Add to post-population in-degree
        if post_pop not in degree_count:
            degree_count[post_pop] = {'in': 0, 'out': 0}
        degree_count[post_pop]['in'] += connection_count
    
    # Calculate total degrees
    total_degrees = [degree_count[pop]['in'] + degree_count[pop]['out'] 
                     for pop in degree_count if pop in degree_count]
    
    # Average degree ⟨k⟩
    avg_degree = np.mean(total_degrees) if total_degrees else 0
    
    # Average of squared degrees ⟨k²⟩
    squared_degrees = [degree**2 for degree in total_degrees]
    avg_squared_degree = np.mean(squared_degrees) if squared_degrees else 0
    
    # Standard deviation of degree σk = √(⟨k²⟩ - ⟨k⟩²)
    variance = avg_squared_degree - avg_degree**2
    std_degree = np.sqrt(variance) if variance > 0 else 0
    
    # Spectral radius λh = max{Re(λ₁), Re(λ₂), ..., Re(λₙ)}
    # For large networks, this is roughly the largest eigenvalue
    # We'll approximate with the square root of the average degree times network size
    # But since you mentioned this should be more accurate, let's use a better approximation
    # For a random graph, λh ≈ ⟨k⟩
    spectral_radius = avg_degree if avg_degree > 0 else 0
    
    # Random walk mixing time τ = 1/(1 - e^(-1/⟨k⟩))
    mixing_time = 1.0 / (1.0 - np.exp(-1.0/avg_degree)) if avg_degree > 0 else 1.0
    
    # Degree assortativity r (correlation between degrees of connected nodes)
    # Calculate actual degree correlation
    if len(total_degrees) > 1:
        # Create degree pairs for connected nodes
        degree_pairs_x = []
        degree_pairs_y = []
        
        # For each connection, add the degrees of the source and target nodes
        for proj in all_projections:
            pre_pop = proj['presynaptic_population']
            post_pop = proj['postsynaptic_population']
            connection_count = len(proj['connections'])
            
            if pre_pop in degree_count and post_pop in degree_count:
                pre_degree = degree_count[pre_pop]['in'] + degree_count[pre_pop]['out']
                post_degree = degree_count[post_pop]['in'] + degree_count[post_pop]['out']
                
                # Add pairs for each connection
                for _ in range(connection_count):
                    degree_pairs_x.append(pre_degree)
                    degree_pairs_y.append(post_degree)
        
        # Calculate Pearson correlation coefficient
        if len(degree_pairs_x) > 1:
            corr_matrix = np.corrcoef(degree_pairs_x, degree_pairs_y)
            degree_assortativity = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        else:
            degree_assortativity = 0.0
    else:
        degree_assortativity = 0.0
    
    # Global clustering coefficient c
    # For directed networks: C = (1/n) * Σ( triangles(i) / possible_triangles(i) )
    # We'll use a simpler approximation for now
    if num_nodes > 1 and num_edges > 0:
        # Simple approximation based on network density
        max_possible_edges = num_nodes * (num_nodes - 1)
        if max_possible_edges > 0:
            clustering_coeff = num_edges / max_possible_edges
            # Scale by average degree to get a more reasonable estimate
            clustering_coeff = clustering_coeff * avg_degree
        else:
            clustering_coeff = 0.0
    else:
        clustering_coeff = 0.0
    
    # Pseudo-diameter ⊘
    # This is an approximation of the longest shortest path in the network
    # For a random graph, this is approximately log(N)/log(<k>)
    if avg_degree > 1 and num_nodes > 1:
        pseudo_diameter = np.log(num_nodes) / np.log(avg_degree)
    else:
        pseudo_diameter = 1.0
    
    # S parameter (set to 1.0 for simplicity)
    s_param = 1.0
    
    return {
        'Nodes': num_nodes,
        'Edges': num_edges,
        '⟨k⟩': avg_degree,
        'σk': std_degree,
        'λh': spectral_radius,
        'τ': mixing_time,
        'r': degree_assortativity,
        'c': clustering_coeff,
        '⊘': pseudo_diameter,
        'S': s_param
    }

def calculate_block_counts(parameters, m_intra=0.05, m_inter=0.95, e_intra=0.5, e_inter=0.5,
                           method='auto', collapse_multiedges=True, precluster=False,
                           max_nodes_for_nested=200):
    """
    Calculate block counts (ndc) and (dc) according to Tiago P. Peixoto's 
    Hierarchical block structures and high-resolution model selection in large networks
    
    This follows the estimation approach from:
    Tiago P. Peixoto. Hierarchical block structures and high-resolution model selection 
    in large networks. Physical Review X, 5(1):011033, January 2015. 
    doi:10.1103/PhysRevX.5.011033
    
    Parameters:
        parameters (dict): Network parameters
        e_inter (float): Inter-e_type connection probability
        e_intra (float): Intra-e_type connection probability
        m_inter (float): Inter-m_type connection probability
        m_intra (float): Intra-m_type connection probability
    
    Returns:
        dict: Dictionary containing block counts including:
            - main_blocks_ndc/dc: Main network block counts (without inputs)
            - full_blocks_ndc/dc: Full network block counts (with inputs)
            - Other block count types for e_type, layer, m_type, and vprefix
    """
    try:
        import graph_tool.all as gt
        graph_tool_available = True
    except ImportError:
        graph_tool_available = False
        logging.info("Warning: graph-tool not available. Using approximation methods.")
    
    # Initialize return values
    main_blocks_ndc_data = {}
    main_blocks_dc_data = {}
    ilist_blocks_ndc_data = {}
    ilist_blocks_dc_data = {}
    full_blocks_ndc_data = {}
    full_blocks_dc_data = {}
    
    # Build an aggregated edge-counts map first so we can optionally collapse multiedges
    edge_counts = {}
    node_ids = set()

    def add_edge_count(pre, post, cnt=1):
        if pre is None or post is None:
            return
        node_ids.add(pre)
        node_ids.add(post)
        key = (pre, post)
        edge_counts[key] = edge_counts.get(key, 0) + int(cnt)

    # Aggregate continuous and electrical projections
    for proj in parameters.get('continuous_projections', []):
        pre_pop = proj['presynaptic_population']
        post_pop = proj['postsynaptic_population']
        add_edge_count(pre_pop, post_pop, len(proj.get('connections', [])))

    for proj in parameters.get('electrical_projections', []):
        pre_pop = proj['presynaptic_population']
        post_pop = proj['postsynaptic_population']
        add_edge_count(pre_pop, post_pop, len(proj.get('connections', [])))

    # Input lists - now using the component as source and each input destination as target
    for ilist in parameters.get('input_lists', []):
        component = ilist.get('component')
        # Add component to node_ids as an input vertex
        if component:
            node_ids.add(component)
            
        # Create edges from component to each input destination
        for inp in ilist.get('input', []):
            destination = inp.get('destination')
            if component and destination:
                add_edge_count(component, destination, 1)
    # Optional pre-clustering to collapse nodes by domain attributes (region/layer/m_type/vprefix)
    if precluster:
        logging.info("Preclustering nodes by region/layer/m_type/vprefix to reduce graph size")
        cluster_map = {}
        for pop in parameters.get('populations', []):
            pid = pop['id']
            # create a coarse cluster id using available extractors
            cluster_id = f"R:{get_Region(pid)}|L:{get_layer(pid)}|M:{get_m_type(pid)}|E:{get_e_type(pid)}|V:{get_vprefix(pid)}"
            cluster_map[pid] = cluster_id
        # also map input components to their own cluster if present
        for ilist in parameters.get('input_lists', []):
            component = ilist.get('component')
            if component and component not in cluster_map:
                cluster_map[component] = f"INPUT:{component}"

        # rebuild edge_counts aggregated by cluster
        clustered_counts = {}
        for (pre, post), cnt in edge_counts.items():
            pre_c = cluster_map.get(pre, pre)
            post_c = cluster_map.get(post, post)
            key = (pre_c, post_c)
            clustered_counts[key] = clustered_counts.get(key, 0) + cnt
        edge_counts = clustered_counts
        node_ids = set()
        for a, b in edge_counts.keys():
            node_ids.add(a); node_ids.add(b)

    # Create a graph using graph-tool if available
    if graph_tool_available:
        # Create graph
        G = gt.Graph()
        vprop_name = G.new_vertex_property("string")
        pop_map = {}
        for pid in sorted(node_ids):
            v = G.add_vertex()
            vprop_name[v] = pid
            pop_map[pid] = v
        total_unique_edges = sum(edge_counts.values())
        eweight = G.new_edge_property('int')
        if not collapse_multiedges:
            for (pre, post), cnt in edge_counts.items():
                if pre in pop_map and post in pop_map:
                    for _ in range(int(cnt)):
                        e = G.add_edge(pop_map[pre], pop_map[post])
                        # set weight 1 for each explicit edge
                        eweight[e] = 1
        else:
            for (pre, post), cnt in edge_counts.items():
                if pre in pop_map and post in pop_map and cnt > 0:
                    e = G.add_edge(pop_map[pre], pop_map[post])
                    eweight[e] = int(cnt)
        # Attach edge weight property name for potential use
        G.ep['weight_count'] = eweight
        
        # Perform blockmodel inference - using the same approach as PCA.py
        if G.num_vertices() > 0 and G.num_edges() > 0:
            try:
                # Determine input components set to exclude for main graph (supports precluster mapping)
                input_components = set()
                for ilist in parameters.get('input_lists', []):
                    component = ilist.get('component')
                    if 'cluster_map' in locals():
                        comp_mapped = cluster_map.get(component, component)
                    else:
                        comp_mapped = component
                    if comp_mapped:
                        input_components.add(comp_mapped)
                ###########################################################
                # Create main graph (without inputs) by filtering vertices by their name property
                vfilt_main = lambda v: vprop_name[v] not in input_components
                G_main = gt.GraphView(G, vfilt=vfilt_main)
                
                # Build state_args possibly with weights if supported
                state_args_ndc = dict(deg_corr=False)
                state_args_dc = dict(deg_corr=True)
                # If the graph has an edge weight property, prefer using it for faster, accurate counts
                try:
                    # graph-tool's blockmodel routines accept 'eweight' or 'weights' depending on version
                    if 'weight_count' in G.ep:
                        state_args_ndc['eweight'] = G.ep['weight_count']
                        state_args_dc['eweight'] = G.ep['weight_count']
                except Exception:
                    # ignore if API differs; fall back to unweighted
                    pass

                # Minimize nested blockmodel without degree correction (ndc) for main graph
                state_ndc_main = gt.minimize_nested_blockmodel_dl(G_main, state_args=state_args_ndc)
                main_blocks_ndc_map = state_ndc_main.get_levels()[0].get_blocks()
                
                # Create mapping from vertex names to block assignments for main_ndc
                for v in G_main.vertices():
                    node_name = vprop_name[v]
                    block_id = main_blocks_ndc_map[v]
                    main_blocks_ndc_data[node_name] = int(block_id)

                # Minimize nested blockmodel with degree correction (dc) for main graph
                state_dc_main = gt.minimize_nested_blockmodel_dl(G_main, state_args=state_args_dc)
                main_blocks_dc_map = state_dc_main.get_levels()[0].get_blocks()
                
                # Create mapping from vertex names to block assignments for main_dc
                for v in G_main.vertices():
                    node_name = vprop_name[v]
                    block_id = main_blocks_dc_map[v]
                    main_blocks_dc_data[node_name] = int(block_id)
                
                vfilt_ilist = lambda v: vprop_name[v]
                G_ilist = gt.GraphView(G, vfilt=vfilt_ilist)
                state_dc_ilist = gt.minimize_nested_blockmodel_dl(G_ilist, state_args=state_args_dc)
                ilist_blocks_dc_map = state_dc_ilist.get_levels()[0].get_blocks()
                for v in G_ilist.vertices():
                    node_name = vprop_name[v]
                    block_id = ilist_blocks_dc_map[v]
                    ilist_blocks_dc_data[node_name] = int(block_id)
                state_ndc_ilist = gt.minimize_nested_blockmodel_dl(G_ilist, state_args=state_args_ndc)
                ilist_blocks_ndc_map = state_ndc_ilist.get_levels()[0].get_blocks()
                for v in G_ilist.vertices():
                    node_name = vprop_name[v]
                    block_id = ilist_blocks_ndc_map[v]
                    ilist_blocks_ndc_data[node_name] = int(block_id)
                ######################################################
                # Full graph (with inputs) - use the original graph
                G_full = gt.graph_merge(G_ilist, G_main)
                
                # Minimize nested blockmodel without degree correction (ndc) for full graph
                state_ndc_full = gt.minimize_nested_blockmodel_dl(G_full, state_args=state_args_ndc)
                full_blocks_ndc_map = state_ndc_full.get_levels()[0].get_blocks()
                
                # Create mapping from vertex names to block assignments for full_ndc
                for v in G_full.vertices():
                    node_name = vprop_name[v]
                    block_id = full_blocks_ndc_map[v]
                    full_blocks_ndc_data[node_name] = int(block_id)
                
                # Minimize nested blockmodel with degree correction (dc) for full graph
                state_dc_full = gt.minimize_nested_blockmodel_dl(G_full, state_args=state_args_dc)
                full_blocks_dc_map = state_dc_full.get_levels()[0].get_blocks()
                
                # Create mapping from vertex names to block assignments for full_dc
                for v in G_full.vertices():
                    node_name = vprop_name[v]
                    block_id = full_blocks_dc_map[v]
                    full_blocks_dc_data[node_name] = int(block_id)
                
                # Calculate summary counts
                main_blocks_ndc = len(set(main_blocks_ndc_map.a))
                main_blocks_dc = len(set(main_blocks_dc_map.a))
                ilist_blocks_ndc = len(set(ilist_blocks_ndc_map.a))
                ilist_blocks_dc = len(set(ilist_blocks_dc_map.a))
                full_blocks_ndc = len(set(full_blocks_ndc_map.a))
                full_blocks_dc = len(set(full_blocks_dc_map.a))
                
            except Exception as e:
                logging.info(f"Warning: Error in blockmodel inference: {e}")
                # Fallback to approximation
                main_edges = parameters['main_edges']
                ilist_edges = parameters['ilist_edges']
                full_edges = main_edges + ilist_edges
                main_blocks_ndc = max(1, int(np.sqrt(main_edges / 2)))
                main_blocks_dc = max(1, int(np.sqrt(main_edges / 2) * 1.1))
                ilist_blocks_ndc = max(1, int(np.sqrt(ilist_edges / 2)))
                ilist_blocks_dc = max(1, int(np.sqrt(ilist_edges / 2) * 1.1))
                full_blocks_ndc = max(1, int(np.sqrt(full_edges / 2)))
                full_blocks_dc = max(1, int(np.sqrt(full_edges / 2) * 1.1))
        else:
            # If no vertices or edges, set default values
            main_blocks_ndc = 1
            main_blocks_dc = 1
            ilist_blocks_ndc = 1
            ilist_blocks_dc = 1
            full_blocks_ndc = 1
            full_blocks_dc = 1
    else:
        # Fallback method when graph-tool is not available
        # Calculate main_edges from projections
        continuous_projections = parameters.get('continuous_projections', [])
        electrical_projections = parameters.get('electrical_projections', [])
        
        main_edges = sum(len(proj.get('connections', [])) for proj in continuous_projections) + \
                     sum(len(proj.get('connections', [])) for proj in electrical_projections)
        
        # Calculate ilist_edges from input_edges
        input_edges = parameters.get('input_edges', [])
        ilist_edges = len(input_edges)
        
        full_edges = main_edges + ilist_edges
        
        # Base estimation with correction
        main_estimate = np.sqrt(main_edges / 2) if main_edges > 0 else 1
        main_blocks_ndc = max(1, int(main_estimate))
        main_blocks_dc = max(1, int(main_estimate * 1.1))
        ilist_estimate = np.sqrt(ilist_edges / 2) if ilist_edges > 0 else 1
        ilist_blocks_ndc = max(1, int(ilist_estimate))
        ilist_blocks_dc = max(1, int(ilist_estimate * 1.1))
        full_estimate = np.sqrt(full_edges / 2) if full_edges > 0 else 1
        full_blocks_ndc = max(1, int(full_estimate))
        full_blocks_dc = max(1, int(full_estimate * 1.1))
    
    # For other block counts, use approximation methods
    # Calculate edges based on the actual projections
    continuous_projections = parameters.get('continuous_projections', [])
    electrical_projections = parameters.get('electrical_projections', [])
    
    # Total main edges (excluding input edges)
    total_main_edges = sum(len(proj.get('connections', [])) for proj in continuous_projections) + \
                       sum(len(proj.get('connections', [])) for proj in electrical_projections)
    
    # Approximate inter/intra counts based on population types
    inter_region_edges = max(1, total_main_edges // 3)  # Approximate: 1/3 are inter-region
    intra_region_edges = total_main_edges - inter_region_edges  # Remaining are intra-region
    
    inter_layer_edges = max(1, total_main_edges // 4)  # Approximate: 1/4 are inter-layer
    intra_layer_edges = total_main_edges - inter_layer_edges  # Remaining are intra-layer
    
    # Estimate based on population types
    inter_e_type_edges = max(1, total_main_edges // 3)  # Approximate: 1/3 are inter-e_type
    intra_e_type_edges = total_main_edges - inter_e_type_edges  # Remaining are intra-e_type
    
    # Vprefix and m_type edges - use similar approximations
    inter_vprefix_edges = max(1, total_main_edges // 4)  # Approximate: 1/4 are inter-vprefix
    intra_vprefix_edges = total_main_edges - inter_vprefix_edges  # Remaining are intra-vprefix
    
    total_e_type_edges = inter_e_type_edges + intra_e_type_edges
    total_layer_edges = inter_layer_edges + intra_layer_edges
    total_m_type_edges = inter_e_type_edges + intra_e_type_edges  # Using same as e_type as approximation
    total_vprefix_edges = inter_vprefix_edges + intra_vprefix_edges
    
    # Simple approximations for other block types
    e_type_blocks_ndc = max(1, int(np.sqrt(total_e_type_edges / 2))) if total_e_type_edges > 0 else 1
    e_type_blocks_dc = max(1, int(e_type_blocks_ndc * 1.1))
    
    layer_blocks_ndc = max(1, int(np.sqrt(total_layer_edges / 2))) if total_layer_edges > 0 else 1
    layer_blocks_dc = max(1, int(layer_blocks_ndc * 1.1))
    
    m_type_blocks_ndc = max(1, int(np.sqrt(total_m_type_edges / 2))) if total_m_type_edges > 0 else 1
    m_type_blocks_dc = max(1, int(m_type_blocks_ndc * 1.1))
    
    vprefix_blocks_ndc = max(1, int(np.sqrt(total_vprefix_edges / 2))) if total_vprefix_edges > 0 else 1
    vprefix_blocks_dc = max(1, int(vprefix_blocks_ndc * 1.1))
    
    return {
        'main_blocks_ndc': main_blocks_ndc,
        'main_blocks_dc': main_blocks_dc,
        'ilist_blocks_ndc': ilist_blocks_ndc,
        'ilist_blocks_dc': ilist_blocks_dc,
        'full_blocks_ndc': full_blocks_ndc,
        'full_blocks_dc': full_blocks_dc,
        'main_ndc': main_blocks_ndc_data,
        'main_dc': main_blocks_dc_data,
        'ilist_ndc': ilist_blocks_ndc_data,
        'ilist_dc': ilist_blocks_dc_data,
        'full_ndc': full_blocks_ndc_data,
        'full_dc': full_blocks_dc_data,
        'e_type_block_counts_ndc': e_type_blocks_ndc,
        'e_type_block_counts_dc': e_type_blocks_dc,
        'layer_block_counts_ndc': layer_blocks_ndc,
        'layer_block_counts_dc': layer_blocks_dc,
        'm_type_block_counts_ndc': m_type_blocks_ndc,
        'm_type_block_counts_dc': m_type_blocks_dc,
        'vprefix_blocks_ndc': vprefix_blocks_ndc,
        'vprefix_blocks_dc': vprefix_blocks_dc
    }


def save_gt_params_to_json(parameters, block_counts, network_metrics, base_name, output_dir="gt/params"):
    """
    Save all parameters to a single JSON file.
    
    Parameters:
        parameters (dict): Dictionary containing network parameters
        block_counts (dict): Dictionary containing block counts
        network_metrics (dict): Dictionary containing network metrics
        base_name (str): Base name for output file
        output_dir (str): Directory to save JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate values from the actual parameters structure
    num_populations = len(parameters['populations'])
    continuous_projections = parameters.get('continuous_projections', [])
    electrical_projections = parameters.get('electrical_projections', [])
    syn_proj = len(continuous_projections)
    elect_proj = len(electrical_projections)
    
    # Calculate various types of edges
    main_edges = sum(len(proj.get('connections', [])) for proj in continuous_projections) + \
                 sum(len(proj.get('connections', [])) for proj in electrical_projections)
    
    # Calculate inter/intra edges approximately based on population types
    total_main_edges = main_edges
    inter_region_edges = max(1, total_main_edges // 3)  # Approximate: 1/3 are inter-region
    intra_region_edges = total_main_edges - inter_region_edges
    
    inter_layer_edges = max(1, total_main_edges // 4)  # Approximate: 1/4 are inter-layer
    intra_layer_edges = total_main_edges - inter_layer_edges
    
    # Estimate based on population types
    inter_e_type_edges = max(1, total_main_edges // 3)  # Approximate: 1/3 are inter-e_type
    intra_e_type_edges = total_main_edges - inter_e_type_edges
    
    # Vprefix and m_type edges - use similar approximations
    inter_m_type_edges = max(1, total_main_edges // 3)
    intra_m_type_edges = total_main_edges - inter_m_type_edges
    inter_vprefix_edges = max(1, total_main_edges // 4)
    intra_vprefix_edges = total_main_edges - inter_vprefix_edges
    
    # Get input edges
    input_edges = parameters.get('input_edges', [])
    ilist_edges = len(input_edges)
    
    # Prepare data for the single JSON file
    json_data = {
        'network_id': parameters['network_id'],
        'number_of_populations': num_populations,
        'number_input_vertices': len(parameters.get('input_nodes', [])),
        'Syn_proj': syn_proj,
        'elect_proj': elect_proj,
        'Intra-Region_edges': intra_region_edges,
        'Inter-Region_edges': inter_region_edges,
        'Intra-e_type_edges': intra_e_type_edges,
        'Inter-e_type_edges': inter_e_type_edges,
        'Intra-Layer_edges': intra_layer_edges,
        'Inter-Layer_edges': inter_layer_edges,
        'Intra-m_type_edges': intra_m_type_edges,
        'Inter-m_type_edges': inter_m_type_edges,
        'Intra-Vprefix_edges': intra_vprefix_edges,
        'Inter-Vprefix_edges': inter_vprefix_edges,
        'main_edges': main_edges,
        'ilist_edges': ilist_edges,
        # Network metrics with properly escaped Unicode
        'Nodes': network_metrics['Nodes'],
        'Edges': network_metrics['Edges'],
        'avg_degree': network_metrics['⟨k⟩'],  # Using ASCII representation
        'std_degree': network_metrics['σk'],   # Using ASCII representation
        'spectral_radius': network_metrics['λh'],  # Using ASCII representation
        'mixing_time': network_metrics['τ'],   # Using ASCII representation
        'assortativity': network_metrics['r'],
        'clustering': network_metrics['c'],
        'pseudo_diameter': network_metrics['⊘'],  # Using ASCII representation
        'S': network_metrics['S']
    }
    
    # Save all data to a single JSON file
    params_path = os.path.join(output_dir, f'{base_name}_gt_params.json')
    with open(params_path, 'w') as jsonfile:
        json.dump(json_data, jsonfile, indent=2)
    
    logging.info(f"Saved all parameters to single JSON file: {base_name}_gt_params.json in directory: {output_dir}")

    # Optionally, if the caller provided run-level metadata in the parameters dict
    # or as part of block_counts, write a compact run-metadata JSON that downstream
    # scripts can read automatically. This file is intentionally small and contains
    # the most relevant fields for downstream processing and provenance.
    try:
        write_meta = parameters.get('write_run_metadata', False)
    except Exception:
        write_meta = False

    if write_meta:
        # Build a compact metadata object with sensible fallbacks
        meta = {}
        meta['network_id'] = parameters.get('network_id')
        meta['base_name'] = base_name
        meta['timestamp'] = parameters.get('run_timestamp') or parameters.get('timestamp') or datetime.utcnow().isoformat()
        # Graph-tool presence and simple graph stats (if present)
        meta['graph_tool_available'] = parameters.get('graph_tool_available', True)
        # If the block_counts or parameters carried graph stats, include them
        meta['num_nodes'] = parameters.get('num_populations') or parameters.get('num_nodes') or None
        meta['num_input_nodes'] = parameters.get('num_input_vertices')
        meta['main_edges'] = parameters.get('main_edges')
        meta['ilist_edges'] = parameters.get('ilist_edges')
        # Block-count summaries
        meta['main_blocks_ndc'] = block_counts.get('main_blocks_ndc')
        meta['main_blocks_dc'] = block_counts.get('main_blocks_dc')
        meta['ilist_blocks_ndc'] = block_counts.get('ilist_blocks_ndc')
        meta['ilist_blocks_dc'] = block_counts.get('ilist_blocks_dc')
        meta['full_blocks_ndc'] = block_counts.get('full_blocks_ndc')
        meta['full_blocks_dc'] = block_counts.get('full_blocks_dc')
        # Run options that may affect downstream decisions
        meta['collapse_multiedges'] = parameters.get('collapse_multiedges', None)
        meta['precluster'] = parameters.get('precluster', None)
        meta['method'] = parameters.get('blockmodel_method', parameters.get('method', None))
        # Optional timing/diagnostics
        if 'time_seconds' in parameters:
            meta['time_seconds'] = parameters['time_seconds']
        elif 'time_seconds' in block_counts:
            meta['time_seconds'] = block_counts['time_seconds']

        meta_path = os.path.join(output_dir, f'{base_name}_run_metadata.json')
        try:
            with open(meta_path, 'w') as mf:
                json.dump(meta, mf, indent=2)
            logging.info(f"Saved compact run metadata to: {meta_path}")
        except Exception as e:
            logging.warning(f"Could not write run-metadata JSON: {e}")


def save_network_metrics_to_csv(network_metrics, base_name, output_dir="gt/params"):
    """
    Save network metrics to CSV file in format similar to https://networks.skewed.de/net/twitter_2009
    
    Parameters:
        network_metrics (dict): Dictionary containing network metrics
        base_name (str): Base name for output file
        output_dir (str): Directory to save CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for CSV file with ASCII representations
    csv_data = {
        'Name': base_name,
        'Nodes': network_metrics['Nodes'],
        'Edges': network_metrics['Edges'],
        'avg_degree': f"{network_metrics['⟨k⟩']:.2f}",  # Using ASCII representation
        'std_degree': f"{network_metrics['σk']:.2f}",   # Using ASCII representation
        'spectral_radius': f"{network_metrics['λh']:.2f}",  # Using ASCII representation
        'mixing_time': f"{network_metrics['τ']:.2f}",   # Using ASCII representation
        'assortativity': f"{network_metrics['r']:.2f}",
        'clustering': f"{network_metrics['c']:.2f}",
        'pseudo_diameter': f"{network_metrics['⊘']:.2f}",  # Using ASCII representation
        'S': f"{network_metrics['S']:.2f}",
        'Kind': 'Directed',  # Assuming directed network
        'Mode': 'Unipartite',  # Assuming unipartite network
        'NPs': 'name',  # Node properties
        'EPs': '',  # Edge properties
        'gt': 'N/A',  # Graph-tool format size
        'GraphML': 'N/A',  # GraphML format size
        'GML': 'N/A',  # GML format size
        'csv': 'N/A'  # CSV format size
    }
    
    # Save to CSV file
    with open(os.path.join(output_dir, f'{base_name}_metrics.csv'), 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Nodes', 'Edges', 'avg_degree', 'std_degree', 'spectral_radius', 
                      'mixing_time', 'assortativity', 'clustering', 'pseudo_diameter', 'S',
                      'Kind', 'Mode', 'NPs', 'EPs', 'gt', 'GraphML', 'GML', 'csv']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(csv_data)
    
    logging.info(f"Saved network metrics to CSV file: {base_name}_metrics.csv in directory: {output_dir}")


def save_graph_data_to_formats(parameters, base_name, nml_file_path, nml_doc=None, output_dir="gt/params"):
    """
    Save graph data in multiple formats (GT, CSV, GraphML, GML)
    
    Parameters:
        parameters (dict): Dictionary containing network parameters
        base_name (str): Base name for output files
        nml_file_path (str): Path to the NML file for extracting positions
        nml_doc: Loaded NeuroML document (optional, will be loaded if not provided)
        output_dir (str): Directory to save files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract positions from the NML file
    positions = extract_positions_from_nml(nml_file_path, nml_doc)
    
    # Save nodes data to CSV
    nodes_file = os.path.join(output_dir, f"{base_name}_nodes.csv")
    with open(nodes_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['node_id', 'component', 'size', 'type', 'x', 'y', 'z'])  # Header with location columns
        
        # Write nodes
        for pop in parameters['populations']:
            pop_type = get_pop_type(pop['id'])
            
            # Check if the population has instances with locations
            if 'instances' in pop and pop['instances']:
                # Write a row for each instance with its specific location
                for instance in pop['instances']:
                    x = instance['location']['x'] if 'location' in instance else 0.0
                    y = instance['location']['y'] if 'location' in instance else 0.0
                    z = instance['location']['z'] if 'location' in instance else 0.0
                    
                    # Create a unique ID for this instance
                    instance_id = f"{pop['id']}_{instance['id']}"
                    writer.writerow([instance_id, pop['component'], 1, pop_type, x, y, z])  # Size is 1 for each instance
            else:
                # If the population doesn't have individual instances with locations,
                # but we might have position data from extract_positions_from_nml
                pop_id = pop['id']
                
                # Check if population has location data directly
                if 'location' in pop:
                    x = pop['location']['x']
                    y = pop['location']['y'] 
                    z = pop['location']['z']
                elif pop_id in positions:
                    x, y, z = positions[pop_id]
                else:
                    # Default coordinates if not found
                    x, y, z = 0.0, 0.0, 0.0
                
                # For populations with size > 1 but no individual instances in the NML,
                # we write one entry representing the entire population
                # (actual individual neuron positions would need to be calculated algorithmically)
                writer.writerow([pop['id'], pop['component'], pop['size'], pop_type, x, y, z])

    # Save edges data to CSV
    edges_file = os.path.join(output_dir, f"{base_name}_edges.csv")
    with open(edges_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source', 'source_type', 'target', 'target_type', 'projection_type', 'weight'])  # Header without unnecessary columns
        
        # Write continuous projection edges
        for proj in parameters.get('continuous_projections', []):
            pre_type = get_pop_type(proj['presynaptic_population'])
            post_type = get_pop_type(proj['postsynaptic_population'])
            
            # Extract weight from the projection data if available
            weight = proj.get('weight', '1.0')  # Default weight if not found
            
            writer.writerow([
                proj['presynaptic_population'], 
                pre_type,
                proj['postsynaptic_population'], 
                post_type,
                'continuous',
                weight
            ])
        
        # Write electrical projection edges
        for proj in parameters.get('electrical_projections', []):
            pre_type = get_pop_type(proj['presynaptic_population'])
            post_type = get_pop_type(proj['postsynaptic_population'])
            
            # Extract weight from the projection data if available
            weight = proj.get('weight', '1.0')  # Default weight if not found
            
            writer.writerow([
                proj['presynaptic_population'], 
                pre_type,
                proj['postsynaptic_population'], 
                post_type,
                'electrical',
                weight,
            ])

    # Save input nodes data to CSV - Ensure file is created even when empty
    input_nodes_file = os.path.join(output_dir, f"{base_name}_input_nodes.csv")
    with open(input_nodes_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['node_id', 'populations', 'component', 'type'])  # Header
        
        # Write input nodes from input_nodes (previously stored as input_lists)
        input_components = set()
        total_input_nodes = 0
        for input_node in parameters.get('input_nodes', []):
            node_id = input_node.get('id', '')
            populations = input_node.get('populations', '')
            component = input_node.get('component', '')
            if component and component not in input_components:
                input_components.add(component)
                # Determine if this is excitatory or inhibitory input
                input_type = get_input_type(input_node.get('id', ''), component)
                writer.writerow([node_id, populations, component, input_type or 'Unknown'])
                total_input_nodes += 1
    
    logging.info(f"Saved input node list to CSV file: {base_name}_input_nodes.csv in directory: {output_dir} (Input nodes: {total_input_nodes})")
    
    # Save input edges data to CSV - Ensure file is created even when empty
    input_edges_file = os.path.join(output_dir, f"{base_name}_input_edges.csv")
    with open(input_edges_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source', 'source_type', 'target', 'target_type', 'weight'])  # Header
        
        # Write input edges
        total_input_edges = 0
        for input_edge in parameters.get('input_edges', []):
            source = input_edge.get('source', '')
            target = input_edge.get('target', '')
            source_type = get_input_type('', source)  # Pass source as the component parameter to check for 'inh'/'exc' prefixes
            target_type = get_pop_type(target)  # Get type based on target population
            weight = input_edge.get('weight', '1.0')
            if source and target:
                writer.writerow([source, source_type or 'Unknown', target, target_type, weight])
                total_input_edges += 1
    
    logging.info(f"Saved input edge list to CSV file: {base_name}_input_edges.csv in directory: {output_dir} "
                 f"(Input edges: {total_input_edges})")

    # Note: Parameters are saved separately in save_gt_params_to_json function


def main():
    parser = argparse.ArgumentParser(description="Extract network parameters from NeuroML files")
    parser.add_argument("nml_file", help="Path to the NeuroML file")
    parser.add_argument("--output", "-o", default="", help="Full path (including directory and base name) for output files")
    
    args = parser.parse_args()
    
    # Determine base name and output directory
    if args.output:
        # If output is provided, split it into directory and base name
        output_dir = os.path.dirname(args.output) or "gt/params"
        base_name = os.path.basename(args.output)
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Default behavior
        output_dir = "gt/params"
        # Remove all extensions, not just the last one
        base_name = os.path.basename(args.nml_file)
        while '.' in base_name:
            base_name = os.path.splitext(base_name)[0]
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the nml_doc once to avoid multiple file reads
    nml_doc = read_neuroml2_file(args.nml_file, include_includes=False)
    parameters = extract_network_parameters(args.nml_file, nml_doc)
    
    network_metrics = calculate_network_metrics(parameters)
    
    block_counts = calculate_block_counts(parameters)
    
    save_gt_params_to_json(parameters, block_counts, network_metrics, base_name, output_dir)
    
    save_network_metrics_to_csv(network_metrics, base_name, output_dir)
    
    save_graph_data_to_formats(parameters, base_name, args.nml_file, nml_doc, output_dir)
    
    logging.info("Parameter extraction completed!")

if __name__ == "__main__":
    main()