# %%
import os
os.environ['GDK_BACKEND'] = 'broadway'
os.environ['GSK_RENDERER'] = 'cairo'
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "12"
import subprocess
import numpy as np
from pyneuroml.pynml import read_neuroml2_file, run_lems_with_jneuroml, run_lems_with_jneuroml_neuron
import subprocess
import gi
gi.require_version('Gtk', '3.0')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.switch_backend("cairo")
# from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
from graph_tool.all import *
import graph_tool.all as gt
from tqdm import tqdm
import time
import logging
import gc
from numpy.random import *
from numpy.linalg import norm
os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# %%
TCs_intralaminar = ["TCRil","nRTil"]
TCs_matrix = ["TCR","TCRm","nRTm"]
TCs_core = ["nRT","TCRc","nRTc"]
thalamus = TCs_core + TCs_matrix + TCs_intralaminar

intra_m = {"DAC","NGCDA","NGCSA","HAC","SLAC","MC","BTC","DBC","BP","NGC","LBC", "NBC","SBC","ChC"}
pyr_m = {"PC","SP","SS", "TTPC1","TTPC2","UTPC","STPC","IPC","BPC","TPC_L4","TPC_L1"}

exc_e = {"cADpyr", "cAC", "cNAC", "cSTUT", "cIR","TCR","TCRm","nRTm"}
inh_e = {"bAC","bNAC","dNAC","bSTUT","dSTUT","bIR","TCRil","nRTil","nRT","TCRc","nRTc"}
e_type_list = {"cADpyr", "cAC", "bAC", "cNAC","bNAC", "dNAC", "cSTUT", "bSTUT","dSTUT", "cIR", "bIR"}

layer_list = {"L1","L23","L4","L5","L6","thalamus"}
Region_list = {"M2a","M2b","M1a","M1b","S1a","S1b"}

def visualize_network(nml_net_file, V_intra, V_inter, L_intra, L_inter, base_name):
    logging.info(f"process {nml_net_file}")
    nml_doc = read_neuroml2_file(nml_net_file)
    N = len(nml_doc.networks[0].populations) + len(nml_doc.networks[0].input_lists)
    G = price_network(N, directed=True)
    # Properties for vertices and edges
    vprop_name = G.new_vertex_property("string")
    vprop_type = G.new_vertex_property("string")
    vprop_size = G.new_vertex_property("double")
    eprop_name = G.new_edge_property("string")
    eprop_type = G.new_edge_property("string")
    eprop_width = G.new_edge_property("double")
    eprop_color = G.new_edge_property("vector<double>")
    eprop_dash = G.new_edge_property("vector<double>")
    

    def get_pop_type(pop_id):
        if not isinstance(pop_id, str) or pop_id == "":
            return pop_id
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) >= 2 and parts[0] in Region_list:
            for exc_type in exc_e:
                if parts[1].startswith(exc_type):
                    return 'exc'
            for inh_type in inh_e:
                if parts[1].startswith(inh_type):
                    return 'inh'
        elif len(parts) >= 2 and parts[0] not in Region_list:
            for exc_type in exc_e:
                if parts[0].startswith(exc_type):
                    return 'exc'
            for inh_type in inh_e:
                if parts[0].startswith(inh_type):
                    return 'inh'
        elif len(parts) == 1:
            for exc_type in exc_e:
                if parts[0] in exc_e:
                    return 'exc'
            for inh_type in inh_e:
                if parts[0] in inh_e:
                    return 'inh'
        else:
            return parts[0]

     
    def get_Vprefix(pop_id):
        try:
            if not isinstance(pop_id, str) or pop_id == "":
                return pop_id
            parts = pop_id.split('_') if "_" in pop_id else [pop_id]
            if len(parts) == 1 and parts[0] in thalamus:
                return parts[0]
            elif len(parts) == 2 and parts[0] in Region_list and parts[1] in thalamus:
                return parts[1]
            elif len(parts) >= 6 and parts[1] in layer_list:
                return '_'.join(parts[1:3])
            elif len(parts) >= 7 and parts[0] in Region_list and parts[2] in layer_list:
                return '_'.join(parts[2:4]) 
            elif len(parts) >= 7 and parts[1] in layer_list:
                return '_'.join(parts[2:5])
            return "unknown"
        except Exception as e:
            logging.error(f"Error in get_Vprefix with pop_id {pop_id}: {e}")
            return "unknown"
    
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

    def get_input_type(ilist_id):
        ilist_id_lower = ilist_id.lower()
        if 'exc' in ilist_id_lower:
            return 'exc'
        elif 'inh' in ilist_id_lower:
            return 'inh'
        else:
            return 'default'

    def get_gen_type(ilist_id):
        if '_PG_' in ilist_id:
            return 'pulse_generators'
        elif '_ComInp_' in ilist_id:
            return 'compound_inputs'
        elif '_VC_' in ilist_id:
            return 'voltage_clamp_triples'
        else:
            return 'unknown'
       
    try:
        pop_map = {}
        pop_type_stats = {}
        pop_states = {}
        for pop in nml_doc.networks[0].populations:
            v1 = G.add_vertex()
            vprop_name[v1] = pop.id
            vertex_type = get_Vprefix(pop.id)
            pop_type = get_pop_type(pop.id)
            vprop_type[v1] = pop_type
            size = float(pop.size) if hasattr(pop, 'size') else 1.0
            vprop_size[v1] = np.log1p(size) * 2  # Logarithmic scaling for better visibility
            v1.population = pop
            pop_map[pop.id] = v1
            pop_states[vertex_type] = pop_states.get(vertex_type, 0) + 1
            pop_type_stats[pop_type] = pop_type_stats.get(pop_type, 0) + 1
        logging.info(f"Detected {len(pop_map)} pop vertices from {base_name} status : %s", pop_type_stats)
        logging.info(f"pop_states: %s", pop_states)

        input_map = {}
        input_type_stats = {}
        input_states = {}
        if hasattr(nml_doc, 'pulse_generators') and nml_doc.pulse_generators:
            for pg in nml_doc.pulse_generators:
                v2 = G.add_vertex()
                vprop_name[v2] = pg.id
                vertex_type = get_gen_type(pg.id)
                pg_type = get_input_type(pg.id)
                vprop_type[v2] = pg_type  # Use consistent type identifier
                vprop_size[v2] = 2
                input_map[pg.id] = v2
                input_states[vertex_type] = input_states.get(vertex_type, 0) + 1
                input_type_stats[pg_type] = input_type_stats.get(pg_type, 0) + 1
        if hasattr(nml_doc, 'compound_inputs') and nml_doc.compound_inputs:
            for ci in nml_doc.compound_inputs:
                v3 = G.add_vertex()
                vprop_name[v3] = ci.id
                vertex_type = get_gen_type(ci.id)
                ci_type = get_input_type(ci.id)
                vprop_type[v3] = ci_type  # Use consistent type identifier
                vprop_size[v3] = 2
                input_map[ci.id] = v3
                input_states[vertex_type] = input_states.get(vertex_type, 0) + 1
                input_type_stats[ci_type] = input_type_stats.get(ci_type, 0) + 1  
        
        if hasattr(nml_doc, 'voltage_clamp_triples') and nml_doc.voltage_clamp_triples:
            for vc in nml_doc.voltage_clamp_triples:
                v4 = G.add_vertex()
                vprop_name[v4] = vc.id
                vertex_type = get_gen_type(vc.id)
                vc_type = get_input_type(vc.id)
                vprop_type[v4] = vc_type
                vprop_size[v4] = 2
                input_map[vc.id] = v4
                input_states[vertex_type] = input_states.get(vertex_type, 0) + 1
                input_type_stats[vc_type] = input_type_stats.get(vc_type, 0) + 1
        logging.info(f"Detected {len(input_map)} input vertices type breakdown : %s", input_type_stats)
        logging.info(f"input_states: %s", input_states)

        # Step 2: Add edges with different types
        edge_count = {'continuous': 0, 'electrical': 0}
        edge_type = G.new_edge_property("string")  # New property for edge types
        edge_weight = G.new_edge_property("double")

        for Syn_proj in nml_doc.networks[0].continuous_projections:
            pre = Syn_proj.presynaptic_population
            post = Syn_proj.postsynaptic_population
            if (pre in pop_map and post in pop_map
                and hasattr(Syn_proj, 'continuous_connection_instance_ws')
                and len(Syn_proj.continuous_connection_instance_ws) > 0):
                Vprefix_pre = get_Vprefix(pre)
                Vprefix_post = get_Vprefix(post)
                Vprob = V_intra if Vprefix_pre == Vprefix_post else V_inter
                v_pass = (np.random.rand() < Vprob)
                layer_pre = get_layer(pre)
                layer_post = get_layer(post)
                Lprob = L_intra if layer_pre == layer_post else L_inter
                l_pass = (np.random.rand() < Lprob)
                inter_module = (Vprefix_pre != Vprefix_post)  or (layer_pre != layer_post)
                if inter_module:
                    if (v_pass and l_pass and np.random.rand() < 0.9):  
                        Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
                        e1 = G.add_edge(pop_map[pre], pop_map[post])
                        if v_pass and l_pass:
                            eprop_width[e1] = 0.2  
                            # eprop_color[e1] = [0.0, 0.45, 0.8, 0.65]  
                            eprop_dash[e1] = []
                        else:
                            eprop_width[e1] = 0.9 
                            # eprop_color[e1] = [0.0, 0.8, 0.25, 0.35]
                            eprop_dash[e1] = [0.05, 0.18]
                        
                        edge_weight[e1] = Syn_w
                        edge_type[e1] = 'Proj'
                        eprop_type[e1] = 'continuous'
                        eprop_dash[e1] = []
                        edge_count['continuous'] += 1
                    elif (v_pass and l_pass):
                        Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
                        e1 = G.add_edge(pop_map[pre], pop_map[post])
                
                        if v_pass and l_pass:
                            eprop_width[e1] = 0.2  
                            # eprop_color[e1] = [0.0, 0.45, 0.8, 0.65]  
                            eprop_dash[e1] = []
                        else:
                            eprop_width[e1] = 0.9
                            # eprop_color[e1] = [0.0, 0.8, 0.25, 0.35]
                            eprop_dash[e1] = [0.05, 0.18]

                        edge_weight[e1] = Syn_w
                        eprop_width[e1] = Syn_w
                        edge_type[e1] = 'Proj'
                        eprop_type[e1] = 'continuous'
                        eprop_dash[e1] = []
                        edge_count['continuous'] += 1
        
        logging.info(f"Added {edge_count['continuous']} Syn_proj edges")

        # Add electrical projections with community-aware edge weights
        for elect_proj in nml_doc.networks[0].electrical_projections:
            pre = elect_proj.presynaptic_population
            post = elect_proj.postsynaptic_population
            elect_w = elect_proj.electrical_connection_instance_ws[0].weight
            if (pre in pop_map and post in pop_map
                and hasattr(elect_proj, 'electrical_connection_instance_ws') 
                and len(elect_proj.electrical_connection_instance_ws) > 0) :
                
                Vprefix_pre = get_Vprefix(pre)
                Vprefix_post = get_Vprefix(post)
                Vprob = V_intra if Vprefix_pre == Vprefix_post else V_inter
                v_pass = (np.random.rand() < Vprob)
                
                layer_pre = get_layer(pre)
                layer_post = get_layer(post)
                Lprob = L_intra if layer_pre == layer_post else L_inter
                l_pass = (np.random.rand() < Lprob)

                inter_module = (Vprefix_pre != Vprefix_post) or (layer_pre != layer_post)
                if inter_module:
                    if (v_pass and l_pass and np.random.rand() < 0.9):  
                        
                        e2 = G.add_edge(pop_map[elect_proj.presynaptic_population], 
                                        pop_map[elect_proj.postsynaptic_population])
                        
                        if v_pass and l_pass:
                            eprop_width[e2] = 0.25 
                            # eprop_color[e2] = [1.0, 0.1, 0.1, 0.75]  
                            eprop_dash[e2] = []
                        else:
                            eprop_width[e2] = 0.12
                            # eprop_color[e2] = [0.8, 0.0, 0.6, 0.45]
                            eprop_dash[e2] = [0.08, 0.22]

                        edge_type[e2] = 'Proj'
                        edge_weight[e2] = elect_w
                        eprop_width[e2] = elect_w
                        eprop_type[e2] = 'electrical'
                        eprop_dash[e2] = [0.2, 0.2]
                        edge_count['electrical'] += 1
                elif (v_pass and l_pass):
                    e2 = G.add_edge(pop_map[elect_proj.presynaptic_population], 
                                pop_map[elect_proj.postsynaptic_population])
                    if v_pass and l_pass:
                        eprop_width[e2] = 0.25 
                        # eprop_color[e2] = [1.0, 0.1, 0.1, 0.75]  
                        eprop_dash[e2] = []
                    else:
                        eprop_width[e2] = 0.12
                        # eprop_color[e2] = [0.8, 0.0, 0.6, 0.45]
                        eprop_dash[e2] = [0.08, 0.22]
                    edge_weight[e2] = elect_w
                    edge_type[e2] = 'Proj'
                    eprop_type[e2] = 'electrical'
                    edge_count['electrical'] += 1     
        logging.info(f"Added {edge_count['electrical']} elect_proj edges")

        V_intra = 0
        V_inter = 0
        L_intra = 0
        L_inter = 0
        for e in G.edges():
            pre = vprop_name[e.source()]
            post = vprop_name[e.target()]
            if get_Vprefix(pre) == get_Vprefix(post):
                V_intra += 1
            else:
                V_inter += 1
            if get_layer(pre) == get_layer(post):
                L_intra += 1
            else:
                L_inter += 1
        logging.info(f"Intra Vprefix-edges: {V_intra}, Inter Vprefix-edges: {V_inter}")
        logging.info(f"Intra layer-edges: {L_intra}, Inter layer-edges: {L_inter}")
        
        input_edges = {'exc_input': 0, 'inh_input': 0}
        destination_stats = {}
        total_input_edges = 0
        
        # Check if input_lists exist before processing
        if hasattr(nml_doc.networks[0], 'input_lists') and nml_doc.networks[0].input_lists:
            for ilist in nml_doc.networks[0].input_lists:
                pre = ilist.component
                post = ilist.populations
                # Only proceed if this input source was actually created
                if pre not in input_map:
                    logging.debug(f"Skipping input_list with unknown component: {pre}")
                    continue
                if post not in pop_map:
                    logging.debug(f"Skipping input_list targeting unknown population: {post}")
                    continue
                inputs = ilist.input if isinstance(ilist.input, list) else [ilist.input] if hasattr(ilist, 'input') else []
                for input_item in inputs:
                    try:
                        destination = getattr(input_item, 'destination', None) 
                        destination_stats[destination] = destination_stats.get(destination, 0) + 1

                        if destination == 'AMPA_NMDA':
                            e3 = G.add_edge(input_map[pre], pop_map[post])
                            total_input_edges += 1
                            eprop_name[e3] = destination
                            eprop_type[e3] = "AMPA_NMDA"    
                            edge_type[e3] = "exc_input"
                            input_map[destination] = e3
                            input_edges['exc_input'] += 1
                            input_w = getattr(input_item, 'weight', 1.0)
                            edge_weight[e3] = input_w
                            eprop_width[e3] = 0.1
                            # eprop_color[e3] = [1.0, 0.5, 0.0, 0.5]
                        elif destination == 'GABA':
                            e4 = G.add_edge(input_map[pre], pop_map[post])
                            total_input_edges += 1
                            eprop_name[e4] = destination
                            eprop_type[e4] = "GABA"
                            edge_type[e4] = "inh_input"
                            input_map[destination] = e4
                            input_edges['inh_input'] += 1
                            input_w = getattr(input_item, 'weight', 1.0)
                            edge_weight[e4] = input_w
                            eprop_width[e4] = 0.1
                            # eprop_color[e4] = [1.0, 0.0, 0.0, 0.5]  # Reddish for inputs
                        elif destination == 'GapJ':
                            # For GapJ, we need to determine type from the pre-component
                            if get_input_type(pre) == 'exc':
                                e5 = G.add_edge(input_map[pre], pop_map[post])
                                total_input_edges += 1
                                eprop_name[e5] = destination
                                eprop_type[e5] = "GapJ"
                                edge_type[e5] = "exc_input"
                                input_map[destination] = e5
                                input_edges['exc_input'] += 1
                                input_w = getattr(input_item, 'weight', 1.0)
                                edge_weight[e5] = input_w
                            else:
                                e6 = G.add_edge(input_map[pre], pop_map[post])
                                total_input_edges += 1
                                eprop_name[e6] = destination
                                eprop_type[e6] = "GapJ"
                                edge_type[e6] = "inh_input"
                                input_map[destination] = e6
                                input_edges['inh_input'] += 1
                                input_w = getattr(input_item, 'weight', 1.0)
                                edge_weight[e6] = input_w
                    except Exception as ex:
                        logging.warning(f"Failed to add input edge from {pre} to {post}: {ex}")
        # Log detailed statistics
        logging.info("Destination breakdown: %s", destination_stats)
        logging.info("Detected %d input edges: %d exc | %d inh", total_input_edges, input_edges['exc_input'], input_edges['inh_input'])

        # Step 3: Community detection first
        # gt.remove_self_loops(G)
        
        state = gt.minimize_nested_blockmodel_dl(G)
        gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))
        
        tree, prop, vprop = gt.get_hierarchy_tree(state)
        ecount = tree.num_edges()
        vcount = tree.num_vertices()
        logging.info(f"Tree has {vcount} vertices and {ecount} edges")
        
        levels = state.get_levels() # Get the hierarchy levels
        logging.info(f"Detected {len(levels)} hierarchy levels")
        
        b = levels[0].get_blocks() # Get block structure from highest level
        Vprefixs = set(get_Vprefix(vprop_name[v]) for v in G.vertices() if vprop_name[v] in pop_map)
        Regions = set(get_Region(vprop_name[v]) for v in G.vertices() if vprop_name[v] in pop_map)
        layers = set(get_layer(vprop_name[v]) for v in G.vertices() if vprop_name[v] in pop_map)
        logging.info(f"Vprefixs: {Vprefixs}")
        logging.info(f"Regions: {Regions}")
        logging.info(f"layers: {layers}")

        main_vertices = [v for v in G.vertices() if vprop_name[v] == pop.id]
        main_edges = [e for e in G.edges() if e.source() in main_vertices and e.target() in main_vertices]
        G_main = gt.GraphView(G, vfilt=lambda v: v in main_vertices, efilt=lambda e: e in main_edges)
        state_ndc_main = gt.minimize_nested_blockmodel_dl(G_main, state_args=dict(deg_corr=False))
        state_dc_main  = gt.minimize_nested_blockmodel_dl(G_main, state_args=dict(deg_corr=True))
        logging.info("Block counts (ndc): %s", np.unique(state_ndc_main.get_levels()[0].get_blocks().a, return_counts=True))
        logging.info("Block counts (dc): %s", np.unique(state_dc_main.get_levels()[0].get_blocks().a, return_counts=True))
        main_ndc_b = state_ndc_main.get_levels()[0].get_blocks()
        main_dc_b = state_dc_main.get_levels()[0].get_blocks()
        main_comm_ndc = len(set(main_ndc_b.a))
        main_comm_dc = len(set(main_dc_b.a))
        logging.info(f"main_vertices_ndc: {main_comm_ndc} communities | main_vertices_dc: {main_comm_dc} communities")

        input_vertices = [v for v in G.vertices() if vprop_name[v] != pop.id]
        full_vertices = [v for v in G.vertices()]
        ilist_edges = [e for e in G.edges() if e.source() in input_vertices]
        full_edges = [e for e in G.edges()]
        G_ilist = gt.GraphView(G, vfilt=lambda v: v in input_vertices, efilt=lambda e: e in ilist_edges)
        G_full = gt.GraphView(G, vfilt=lambda v: v in full_vertices, efilt=lambda e: e in full_edges)
        state_ndc_ilist = gt.minimize_nested_blockmodel_dl(G_ilist, state_args=dict(deg_corr=False))
        state_dc_ilist = gt.minimize_nested_blockmodel_dl(G_ilist, state_args=dict(deg_corr=True))
        state_ndc_full = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=False))
        state_dc_full  = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=True))
        logging.info("Non-degree-corrected ILIST:%s", state_ndc_ilist.entropy())
        logging.info("Non-degree-corrected FULL:%s", state_ndc_full.entropy())
        logging.info("Degree-corrected ILIST:%s", state_dc_ilist.entropy())
        logging.info("Degree-corrected FULL:%s", state_dc_full.entropy())
        logging.info(u"ln \u039b:\t\t\t:%s", state_ndc_ilist.entropy() - state_dc_ilist.entropy())
        logging.info(u"ln \u039b:\t\t\t:%s", state_ndc_full.entropy() - state_dc_full.entropy())
        ilist_ndc_b = state_ndc_ilist.get_levels()[0].get_blocks()
        full_ndc_b = state_ndc_full.get_levels()[0].get_blocks()
        ilist_dc_b = state_dc_ilist.get_levels()[0].get_blocks()
        full_dc_b = state_dc_full.get_levels()[0].get_blocks()
        logging.info("ilist Block counts (ndc): %s", np.unique(ilist_ndc_b.a, return_counts=True))
        logging.info("full Block counts (ndc): %s", np.unique(full_ndc_b.a, return_counts=True))
        logging.info("ilist Block counts (dc): %s", np.unique(ilist_dc_b.a, return_counts=True))
        logging.info("full Block counts (dc): %s", np.unique(full_dc_b.a, return_counts=True))

        group_keys = []
        for v in G.vertices():
            vertex_type = vprop_type[v]
            group_keys.append(vertex_type)
        unique_groups = sorted(dict.fromkeys(group_keys).keys())
        group_pos = {g: i for i, g in enumerate(unique_groups)}
        vprop_group = G.new_vp("int")
        for v, gkey in zip(G.vertices(), group_keys):
            vprop_group[v] = group_pos.get(gkey, 0)    
        
        # Step 4: Create proper position property map
        pos = G.new_vertex_property("vector<double>")
        G.vp["pos"] = pos
        # Step 5: Fine-tune layout with sfdp/frl/al/rtl/pl/rl using proper position initialization
        # pos = gt.sfdp_layout(G, pos=pos, groups=vprop_group, C=15.0, K=3.0, p=4.0, gamma=0.01, theta=0.3, max_iter=3000, mu=8, weighted_coarse=True)
        pos = gt.sfdp_layout(G, groups=vprop_group)

        # Helper: validate/repair pos to avoid degenerate transforms
        def _ensure_pos_valid(pos_prop, G, min_jitter=1e-3):
            try:
                # build numpy array of shape (n,2)
                arr = np.vstack([np.asarray(pos_prop[v]) for v in G.vertices()]) if G.num_vertices() > 0 else np.zeros((0,2))
            except Exception:
                arr = None
            if arr is None or arr.size == 0 or not np.isfinite(arr).all():
                # compute fresh layout
                new_pos = gt.sfdp_layout(G)
                for v in G.vertices():
                    pos_prop[v] = list(new_pos[v])
                return pos_prop
            # if all x or all y identical -> add tiny jitter
            if np.ptp(arr[:,0]) == 0.0:
                jitter = np.random.uniform(-min_jitter, min_jitter, size=arr.shape[0])
                for i, v in enumerate(G.vertices()):
                    p = list(pos_prop[v])
                    p[0] = float(p[0]) + float(jitter[i])
                    pos_prop[v] = p
            if np.ptp(arr[:,1]) == 0.0:
                jitter = np.random.uniform(-min_jitter, min_jitter, size=arr.shape[0])
                for i, v in enumerate(G.vertices()):
                    p = list(pos_prop[v])
                    p[1] = float(p[1]) + float(jitter[i])
                    pos_prop[v] = p
                    
            # Add small amount of random noise to all positions to prevent numerical issues
            for i, v in enumerate(G.vertices()):
                pos_prop[v][0] += np.random.normal(0, 1e-6)
                pos_prop[v][1] += np.random.normal(0, 1e-6)
            # final sanity: if still degenerate, recompute sfdp
            arr = np.vstack([np.asarray(pos_prop[v]) for v in G.vertices()]) if G.num_vertices() > 0 else np.zeros((0,2))
            if np.ptp(arr[:,0]) == 0.0 or np.ptp(arr[:,1]) == 0.0:
                new_pos = gt.sfdp_layout(G)
                for v in G.vertices():
                    pos_prop[v] = list(new_pos[v])
            return pos_prop
        # Safe graph draw wrapper with fallback to recompute layout
        def safe_graph_draw(*args, pos_prop=None, Gref=None, retries=1, **kwargs):
            if pos_prop is not None and Gref is not None:
                _ensure_pos_valid(pos_prop, Gref)
            try:
                gt.graph_draw(*args, pos=pos_prop, **kwargs) if 'pos' in gt.graph_draw.__code__.co_varnames else gt.graph_draw(*args, **kwargs)
                return True
            except RuntimeError as re:
                logging.warning(f"graph_draw failed with RuntimeError: {re}. Attempting fallback layout.")
                try:
                    # recompute layout and try again
                    new_pos = gt.sfdp_layout(Gref)
                    if Gref is not None:
                        for v in Gref.vertices():
                            pos_prop[v] = list(new_pos[v])
                    gt.graph_draw(*args, pos=pos_prop, **kwargs)
                    return True
                except Exception as e2:
                    logging.error(f"Fallback graph_draw also failed: {e2}")
                    return False
            except Exception as e:
                logging.error(f"graph_draw unexpected error: {e}")
                return False
            
        # Step 6: Draw with community colors and edge types
        #############################################################################################################################
        prev_exc_state = G.new_vertex_property("vector<double>")
        curr_exc_state = G.new_vertex_property("vector<double>")
        prev_inh_state = G.new_vertex_property("vector<double>")
        curr_inh_state = G.new_vertex_property("vector<double>")
        exc_transmited = G.new_vertex_property("bool")
        inh_transmited = G.new_vertex_property("bool")
        exc_refractory = G.new_vertex_property("bool")
        inh_refractory = G.new_vertex_property("bool")
        w = gt.max_cardinality_matching(G, edges=True, heuristic=True, brute_force=True)
        
        def create_graph_tool_animation(G, pos, state, output_file, vertex_fill_color=None,
                                        vertex_color=None, vertex_size=None, edge_color=None, 
                                        edge_pen_width=None, frames=10,
                                        mode="graph_draw", **kwargs):
            fixed_pos = gt.sfdp_layout(G, cooling_step=0.99)
            res = gt.max_independent_vertex_set(G)
            frame_files = []
            for i in range(frames):
                progress = i / frames
                exc_pop_vertex = [v for v in G.vertices() if vprop_name[v] == pop.id and vprop_type[v] == 'exc']
                inh_pop_vertex = [v for v in G.vertices() if vprop_name[v] == pop.id and vprop_type[v] == 'inh']
                exc_input_vertex = [v for v in G.vertices() if vprop_name[v] != pop.id and vprop_type[v] == 'exc']
                inh_input_vertex = [v for v in G.vertices() if vprop_name[v] != pop.id and vprop_type[v] == 'inh']
                
                exc_inactive_pop = int(progress * len(exc_pop_vertex))  
                inh_inactive_pop = int(progress * len(inh_pop_vertex))
                exc_active_pop = int(progress * 0.7 * len(exc_pop_vertex)) 
                inh_active_pop = int(progress * 0.6 * len(inh_pop_vertex)) 
                exc_refrac_pop = int(progress * 0.3 * len(exc_pop_vertex))
                inh_refrac_pop = int(progress * 0.4 * len(inh_pop_vertex))
                exc_inactive_input = int(progress * len(exc_input_vertex))
                inh_inactive_input = int(progress * len(inh_input_vertex))
                exc_active_input = int(progress * 0.7 * len(exc_input_vertex))
                inh_active_input = int(progress * 0.6 * len(inh_input_vertex))
                exc_refrac_input = int(progress * 0.3 * len(exc_input_vertex))
                inh_refrac_input = int(progress * 0.4 * len(inh_input_vertex))

                for idx, v in enumerate(G.vertices()):
                    prev_exc_state[v] = list(matplotlib.cm.viridis(0.1))[:4] # exc_S
                    prev_inh_state[v] = list(matplotlib.cm.viridis_r(0.1))[:4] # inh_S
                    curr_exc_state[v] = prev_exc_state[v]
                    curr_inh_state[v] = prev_inh_state[v]
                    exc_refractory.a = False
                    exc_transmited.a = False
                    inh_refractory.a = False
                    inh_transmited.a = False
                    vertex_type = vprop_type[v]
                    if vertex_type == 'exc':
                        if idx < exc_refrac_pop:
                            curr_exc_state[v] = list(matplotlib.cm.viridis(0.5))[:4] # exc_R  # Refractory state
                            exc_refractory[v] = True
                        elif idx < exc_active_pop:
                            curr_exc_state[v] = list(matplotlib.cm.viridis(0.8))[:4] # exc_I  # Active state
                            exc_transmited[v] = True
                        elif idx < exc_inactive_pop:
                            curr_exc_state[v] = list(matplotlib.cm.viridis(0.3))[:4] # exc_S  # Inactive state
                        elif idx < exc_refrac_input:
                            curr_exc_state[v] = list(matplotlib.cm.viridis_r(0.6))[:5]
                            exc_refractory[v] = True
                        elif idx < exc_active_input:
                            curr_exc_state[v] = list(matplotlib.cm.viridis_r(0.9))[:5]
                            exc_transmited[v] = True
                        elif idx < exc_inactive_input:
                            curr_exc_state[v] = list(matplotlib.cm.viridis(0.4))[:5]

                    elif vertex_type == 'inh':
                        if idx < inh_refrac_pop:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.5))[:4] # inh_R
                            inh_refractory[v] = True
                        elif idx < inh_active_pop:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.8))[:4] # inh_I
                            inh_transmited[v] = True
                        elif idx < inh_inactive_pop:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.3))[:4] # inh_S
                        elif idx < inh_refrac_input:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.4))[:3]
                            inh_refractory[v] = True
                        elif idx < inh_active_input:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.7))[:3]
                            inh_transmited[v] = True
                        elif idx < inh_inactive_input:
                            curr_inh_state[v] = list(matplotlib.cm.viridis_r(0.2))[:3]
                    else:
                        curr_exc_state[v] = prev_exc_state[v]
                        curr_inh_state[v] = prev_inh_state[v]
                
                ee = [e for e in G.edges() if vprop_name[e.source()] == pop.id and vprop_type[e.source()] == 'exc' and vprop_type[e.target()] == 'exc']
                ei = [e for e in G.edges() if vprop_name[e.source()] == pop.id and vprop_type[e.source()] == 'exc' and vprop_type[e.target()] == 'inh']
                ie = [e for e in G.edges() if vprop_name[e.source()] == pop.id and vprop_type[e.source()] == 'inh' and vprop_type[e.target()] == 'exc']
                ii = [e for e in G.edges() if vprop_name[e.source()] == pop.id and vprop_type[e.source()] == 'inh' and vprop_type[e.target()] == 'inh']
                input_ee = [e for e in G.edges() if vprop_name[e.source()] != pop.id and vprop_type[e.source()] == 'exc']
                input_ii = [e for e in G.edges() if vprop_name[e.source()] != pop.id and vprop_type[e.source()] == 'inh']
                
                ee_edges = int(progress * len(ee) * 0.5)
                ii_edges = int(progress * len(ii) * 0.5)
                ei_edges = int(progress * len(ei) * 0.6)
                ie_edges = int(progress * len(ie) * 0.7)
                input_ee_edges = int (progress * len(input_ee) * 0.5)
                input_ii_edges = int (progress * len(input_ii) * 0.5)
                
                for idx, e in enumerate(G.edges()):
                    if idx < ei_edges:
                        eprop_color[e] = np.random.normal((len(exc_pop_vertex)+len(inh_pop_vertex))/(2*ei_edges), .05, ei_edges)
                    elif idx < ie_edges:
                        eprop_color[e] = np.random.normal((len(exc_pop_vertex)+len(inh_pop_vertex))/(2*ie_edges), .05, ie_edges)
                    elif idx < ee_edges:
                        eprop_color[e] = np.random.normal(len(exc_pop_vertex)/(2*ee_edges), .05, ee_edges)
                    elif idx < ii_edges:
                        eprop_color[e] = np.random.normal(len(inh_pop_vertex)/(2*ii_edges), .05, ii_edges)
                    elif idx < input_ee_edges:
                        eprop_color[e] = np.random.normal(len(exc_input_vertex)/(2*input_ee_edges), .05, input_ee_edges)
                    elif idx < input_ii_edges:
                        eprop_color[e] = np.random.normal(len(exc_input_vertex)/(2*input_ii_edges), .05, input_ii_edges)
                    else:
                        eprop_color[e] = np.random.normal(G.num_vertices()/(2*G.num_edges()), .05, G.num_edges())
                cnorm = matplotlib.colors.Normalize(vmin=-abs(edge_weight.fa).max(), vmax=abs(edge_weight.fa).max())
                frame_file = f"frame_{i:03d}.png"
                gt.graph_draw(
                    G, pos=fixed_pos,
                    vertex_fill_color=res,
                    vertex_shape=b,
                    vertex_color=curr_exc_state if vprop_type == "exc" else curr_inh_state,
                    edge_color=eprop_color,
                    edge_pen_width= w.t(lambda x: 0.5*x + 1),
                    ecnorm=cnorm,
                    ecmap=matplotlib.cm.autumn,
                    bg_color=[0.98, 0.98, 0.98, 1],
                    output=frame_file,
                    output_size=(800, 800)  # Fixed size for animation frames
                )
                frame_files.append(frame_file)
            # Create animation using ImageMagick
            cmd = ["convert", "-delay", "10", "-loop", "0"] + frame_files + [output_file]
            subprocess.run(cmd, check=True)
            for f in frame_files:
                if os.path.exists(f):
                    os.remove(f)
            gc.collect()
        
        
        #############################################################################################################################
        gt.remove_parallel_edges(G)
        f = np.eye(4) * 0.1
        state_graph = gt.PottsGlauberState(G, f)
        ret_graph = state_graph.iterate_async(niter=1000 * G.num_vertices())
        gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_anchor=0, 
                      vertex_fill_color=gt.perfect_prop_hash([state_graph.s])[0], 
                      vertex_shape=gt.perfect_prop_hash([state_graph.s])[0], 
                      edge_color=w, edge_pen_width=w.t(lambda x: 0.5*x + 1),
                      edge_start_marker="bar", edge_end_marker="arrow", output_size=(800, 800),
                      bg_color=[0.98, 0.98, 0.98, 1], output=f"{base_name}_graph.png")
        
        create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state_graph, output_file=f"{base_name}_graph_basic.gif",
                                    vertex_fill_color=gt.perfect_prop_hash([state_graph.s])[0], # vcmap=matplotlib.cm.twilight,
                                    edge_color=w,edge_pen_width=w.t(lambda x: 0.5*x + 1)
                                )
        kcore = gt.kcore_decomposition(gt.GraphView(G, vfilt=gt.label_largest_component(G)))
        state_kcore = gt.NormalState(G, sigma=0.001, w=-100)
        ret_kcore = state_kcore.iterate_sync(niter=1000)
        gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_fill_color=kcore.t(lambda x: 0.5*x + 1),  
                      vertex_shape=state_kcore.s,output_size=(800, 800),
                      edge_color=w, edge_pen_width=w.t(lambda x: 0.5*x + 1), ecmap=matplotlib.cm.coolwarm,
                      bg_color=[0.98, 0.98, 0.98, 1], output=f"{base_name}_kcore.png")
        
        create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state_kcore, 
                                    output_file=f"{base_name}_kcore_basic.gif",
                                    vertex_fill_color=state_kcore.s, 
                                    vertex_shape=state_kcore.s,# vcapmap=matplotlib.cm.coolwarm,
                                    edge_color=w, edge_pen_width=w.t(lambda x: 0.5*x + 1), ecmap=matplotlib.cm.Set1,
                                )
        try:
            similarity = gt.vertex_similarity(GraphView(G, reversed=True),"inv-log-weight")
            color = G.new_vp("double")
            color.a = similarity[0].a
            state_sim = gt.CIsingGlauberState(G, beta=.2)
            ret_sim = state_sim.iterate_async(niter=1000 * G.num_vertices())
            gt.graph_draw(G, gt.sfdp_layout(G, cooling_step=0.99), vertex_fill_color=state_sim.s,# vertex_text=G.vertex_index,
                          vertex_shape=gt.perfect_prop_hash([state_sim.s])[0],output_size=(800, 800),
                          edge_color=w, edge_pen_width=w.t(lambda x: 0.5*x + 1), ecmap=matplotlib.cm.Set3,
                          bg_color=[0.98, 0.98, 0.98, 1], output=f"{base_name}_similarity.png")
            
            create_graph_tool_animation(G, gt.sfdp_layout(G, cooling_step=0.99), state_sim, 
                                        vertex_fill_color=state_sim.s, 
                                        vertex_shape=gt.perfect_prop_hash([state_sim.s])[0],
                                        # vcmap=matplotlib.cm.magma,
                                        output_file=f"{base_name}_similarity_basic.gif",
                                        edge_color=w, edge_pen_width=w.t(lambda x: 0.5*x + 1), ecmap=matplotlib.cm.Set3,
                                    )
        except Exception as e:
            logging.info(f"[WARNING] Failed to calculate vertex similarity: {e}")        
        
        #############################################################################################################################
        G = price_network(G.num_vertices())
        deg = G.degree_property_map("in")
        deg.a = 2 * (np.sqrt(deg.a) * 0.5 + 0.4)
        ebet = gt.betweenness(G)[1]
        gt.graphviz_draw(G, pos=gt.sfdp_layout(G, cooling_step=0.99), maxiter=100, ratio="compress", overlap=False, layout="sfdp",
                        vcolor=deg, vorder=deg, elen=10, vcmap=matplotlib.cm.gist_heat,
                        ecolor=ebet, eorder=ebet, output=f"{base_name}_graphviz.png")
        
        #############################################################################################################################
        metrics = {}
        centrality_metrics = {
            'pr': 'pagerank',
            'bt': 'betweenness',
            'V': 'eigenvector',
            'katz': 'katz',
            'hitsX': 'hits_authority',
            'hitsY': 'hits_hub',
            't': 'eigentrust',
            'tt': 'trust_transitivity',
            'c': 'closeness',
        }
        try:
            G.save(f"{base_name}.gt")
            logging.info(f"Saved graph file: {base_name}.gt")
        except Exception as _e:
            logging.warning(f"Could not save graph to {base_name}.gt: {_e}")
        # Compute metrics once (in-memory) and convert returned numpy arrays to graph-tool vertex properties
        try:
            from metrics.generator import compute_and_save_metrics
        except ImportError:
            import importlib.util
            spec_path = os.path.join(os.getcwd(), "metrics_analysis_project", "src", "metrics", "generator.py")
            # spec = importlib.util.spec_from_file_location("metrics.generator", spec_path)
            if os.path.exists(spec_path):
                spec = importlib.util.spec_from_file_location("metrics.generator", spec_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    compute_and_save_metrics = getattr(mod, "compute_and_save_metrics")
                else:
                    raise ImportError("Cannot load metrics.generator from spec")
            else:
                raise

        out_dir = os.path.join(os.getcwd(), "metrics_out")
        try:
            metrics_dict, npz_path, csv_path = compute_and_save_metrics(
                G, out_dir=out_dir, prefix=base_name, normalize=True, nthreads=8, save_files=True)
            logging.info(f"Computed metrics once (saved: {npz_path is not None})")
            gc.collect()
        except Exception as e:
            logging.warning(f"Failed to compute metrics via generator: {e}")
            metrics_dict = {}
        
        for k, arr in metrics_dict.items():
            try:
                arr_np = np.asarray(arr, dtype=float)
                if arr_np.size != G.num_vertices():
                    tmp = np.full(G.num_vertices(), np.nan, dtype=float)
                    tmp[:min(arr_np.size, G.num_vertices())] = arr_np[:min(arr_np.size, G.num_vertices())]
                    arr_np = tmp
                vp = G.new_vertex_property("double")
                vp.a = np.asarray(arr, dtype=float)
                metrics[k] = vp
            except Exception:
                logging.warning(f"Failed to convert metric {k} to vertex_property; filling NaNs")
                vp = G.new_vertex_property("double")
                vp.a = np.full(G.num_vertices(), np.nan, dtype=float)
                metrics[k] = vp
        # Ensure expected metric keys exist (fill missing with NaNs)
        for key in set(centrality_metrics.values()):
            if key not in metrics:
                logging.info(f"Metric '{key}' missing — filling with NaNs")
                vp = G.new_vertex_property("double")
                vp.a = np.full(G.num_vertices(), np.nan, dtype=float)
                metrics[key] = vp

        for metric_name, metric_key in tqdm(centrality_metrics.items(), desc="Centrality metrics"):
            logging.info(f"\n[INFO] Calculating and plotting {metric_name}...")
            try:
                t0 = time.time()
                metric = metrics[metric_key]
                for v in G.vertices():
                    metric[v] = metric[v]
                # Ensure metric is valid
                if not hasattr(metric, 'a') or len(metric.a) != G.num_vertices():
                    tmp = G.new_vertex_property("double")
                    tmp.a = np.full(G.num_vertices(), np.nan, dtype=float)
                    try:
                        arr = np.asarray(metric)
                        for i, v in enumerate(G.vertices()):
                            tmp[v] = float(arr[i]) if i < arr.size else np.nan
                        metric = tmp
                    except Exception:
                        metric = tmp
                _ensure_pos_valid(gt.sfdp_layout(G, cooling_step=0.99), G)
                # Try to draw with graph-tool (with fallbacks)
                output_file = f"{base_name}_graph_{metric_name}.png"
                draw_success = safe_graph_draw(
                    G, pos_prop=gt.sfdp_layout(G, cooling_step=0.99), Gref=G,
                    output=output_file,
                    vertex_fill_color=metric,
                    # vertex_size=gt.prop_to_size(metric, mi=5, ma=15),
                    vcmap=matplotlib.cm.gist_heat,
                    bg_color=[0.98, 0.98, 0.98, 1]
                )
                if draw_success:
                    logging.info(f"[INFO] Saved {output_file} in {time.time() - t0:.2f} seconds")
                else:
                    logging.info(f"[ERROR] Failed to save {output_file}")
                
                # Also try graphviz draw as backup
                t1 = time.time()
                try:
                    pos = G.own_property(pos) if hasattr(G, 'own_property') else gt.sfdp_layout(G)
                    _ensure_pos_valid(pos, G)
                    gt.graphviz_draw(
                        G, pos, maxiter=100, ratio="compress", overlap=False,layout="sfdp",
                        vcolor=metric, vorder=metric, elen=10,
                        vcmap=matplotlib.cm.gist_heat,
                        ecolor=ebet, eorder=ebet,
                        output=f"{base_name}_graphviz_{metric_name}.png")
                    logging.info(f"[INFO] Saved {base_name}_graphviz_{metric_name}.png in {time.time() - t1:.2f} seconds")
                except Exception as e:
                    logging.warning(f"graphviz_draw failed for {metric_name}: {e}")

            except Exception as e:
                logging.error(f"Failed to process metric {metric_name}: {e}")
                continue        
        #############################################################################################################################
        return 

    except Exception as e:
        logging.info(f"Error in visualization: {str(e)}")
        if 'pop.id' in locals():
            logging.info(f"Population causing error: {pop.id}")
        raise


if __name__ == "__main__":
    nml_net_files = [
        "max_CTC_plus.net.nml",
        # "M1a_max_plus.net.nml",
        # "M1_max_plus.net.nml",
        # "M2_max_plus.net.nml",
        "M2M1S1_max_plus.net.nml",
        # "S1bM1bM2b_max_plus.net.nml",
        # "M2aM1aS1a_max_plus.net.nml",
    ]
    for nml_net_file in nml_net_files:
        base_name = nml_net_file.split('.')[0]
        visualize_network(nml_net_file,V_intra=0.5, V_inter=0.5,L_inter=0.5, L_intra=0.5,
                          base_name=base_name)
        graph_types = ['graph', 'graphviz']
        metric_names = [
            'pr', 'bt', 'V', 'katz','hitsX', 'hitsY', 't', 'tt','c'
        ]
        file_paths = []
        for graph_type in graph_types:
            for metric in metric_names:
                file_name = f"{base_name}_{graph_type}_{metric}.png"
                file_paths.append(file_name)

        for file_path in file_paths:
            if os.path.exists(file_path):
                logging.info(f"File '{file_path}' size: {os.path.getsize(file_path)/(1024*1024):.2f} MB")
            else:
                logging.info(f"File '{file_path}' not found")
