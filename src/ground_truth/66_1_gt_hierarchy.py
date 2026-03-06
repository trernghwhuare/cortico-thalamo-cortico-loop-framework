import os
from matplotlib import cm
os.environ['GDK_BACKEND'] = 'broadway'
os.environ['GSK_RENDERER'] = 'cairo'
os.environ["OMP_WAIT_POLICY"] = "active"
os.environ["OMP_NUM_THREADS"] = "16"
from pyneuroml.pynml import read_neuroml2_file
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend("cairo")
from graph_tool.all import *
import graph_tool.all as gt
from tqdm import tqdm
import time
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def get_pop_type(pop_id):
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
    except Exception as e:
        logging.error(f"Error in get_Vprefix with pop_id {pop_id}: {e}")
        return "unknown"

def get_Region(pop_id):
    try:
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if parts[0] in Region_list:
            return parts[0]
        return None
    except Exception as e:
        logging.error(f"Error in get_Region with pop_id {pop_id}: {e}")
        return pop_id

def get_layer(pop_id):
    try:
        parts = pop_id.split('_') if "_" in pop_id else [pop_id]
        if len(parts) == 1:
            return "thalamus"
        elif len(parts) == 2 and parts[0] in Region_list:
            return "thalamus"
        elif len(parts) >= 6 and parts[1] in layer_list:
            return parts[1]
        elif len(parts) >= 7 and parts[0] in Region_list:  # Fixed: was 'Regions' which is undefined
            return parts[2] 
        
    except Exception as e:
        logging.error(f"Error in get_layer with pop_id {pop_id}: {e}")
        return pop_id
        

def get_input_type(ilist_id):
    if ilist_id.startswith('exc_'):
        return 'exc'
    elif ilist_id.startswith('inh_'):
        return 'inh'

def get_gen_type(ilist_id):
    if '_PG_' in ilist_id:
        return 'pulse_generators'
    elif '_ComInp_' in ilist_id:
        return 'compound_inputs'
    elif '_VC_' in ilist_id:
        return 'voltage_clamp_triples'
    else:
        return 'unknown'


def visualize_network(nml_net_file, V_intra, V_inter, R_intra,  R_inter, L_intra, L_inter, base_name):
    nml_doc = read_neuroml2_file(nml_net_file)
    G = gt.Graph()
    N = len(nml_doc.networks[0].populations) + len(nml_doc.pulse_generators)
    G = price_network(N, directed=True)
    # Properties for vertices and edges
    vprop_name = G.new_vertex_property("string")
    vprop_type = G.new_vertex_property("string")
    vprop_size = G.new_vertex_property("double")
    eprop_name = G.new_edge_property("string")
    eprop_type = G.new_edge_property("string")
    eprop_width = G.new_edge_property("double")
    eprop_color = G.new_edge_property("vector<double>")
    
    # Step 1: Create vertices using population properties
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
        # v1.population = pop
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
        # e1 = G.add_edge(pop_map[pre], pop_map[post])
        if (pre in pop_map and post in pop_map
            and hasattr(Syn_proj, 'continuous_connection_instance_ws')
            and len(Syn_proj.continuous_connection_instance_ws) > 0):
            Vprefix_pre = get_Vprefix(pre)
            Vprefix_post = get_Vprefix(post)
            Vprob = V_intra if Vprefix_pre == Vprefix_post else V_inter
            layer_pre = get_layer(pre)
            layer_post = get_layer(post)
            Lprob = L_intra if layer_pre == layer_post else L_inter
            Region_pre = get_Region(pre)
            Region_post = get_Region(post)
            Rprob = R_intra if Region_pre == Region_post else R_inter
            if np.random.rand() < Vprob and np.random.rand() < Rprob and np.random.rand() < Lprob:
                e1 = G.add_edge(pop_map[pre], pop_map[post])
                Syn_w = Syn_proj.continuous_connection_instance_ws[0].weight
                eprop_width[e1] = 0.9 
                eprop_color[e1] = [0.2, 0.5, 0.3, 1.0]  # From #2C7953
                edge_weight[e1] = Syn_w
                edge_type[e1] = 'proj'
                eprop_type[e1] = 'continuous'
                edge_count['continuous'] += 1
            
    logging.info(f"Detected {edge_count['continuous']} Syn_proj edges")

    for elect_proj in nml_doc.networks[0].electrical_projections:
        pre = elect_proj.presynaptic_population
        post = elect_proj.postsynaptic_population
        # e2 = G.add_edge(pop_map[pre],pop_map[post])
        if (pre in pop_map and post in pop_map
            and hasattr(elect_proj, 'electrical_connection_instance_ws') 
            and len(elect_proj.electrical_connection_instance_ws) > 0) :
            Vprefix_pre = get_Vprefix(pre)
            Vprefix_post = get_Vprefix(post)
            Vprob = V_intra if Vprefix_pre == Vprefix_post else V_inter
            Region_pre = get_Region(pre)
            Region_post = get_Region(post)
            Rprob = R_intra if Region_pre == Region_post else R_inter
            layer_pre = get_layer(pre)
            layer_post = get_layer(post)
            Lprob = L_intra if layer_pre == layer_post else L_inter
            if np.random.rand() < Vprob and np.random.rand() < Rprob and np.random.rand() < Lprob:
                e2 = G.add_edge(pop_map[pre],pop_map[post])
                elect_w = elect_proj.electrical_connection_instance_ws[0].weight
                eprop_width[e2] = elect_w * 2
                eprop_color[e2] = [0.1, 0.7, 0.8, 1.0]   # From #18A7C7
                edge_weight[e2] = elect_w
                edge_type[e2] = 'Proj'
                eprop_type[e2] = 'electrical'
                edge_count['electrical'] += 1 
            
            # inter_module = (Vprefix_pre != Vprefix_post) or (Region_pre != Region_post) or (layer_pre != layer_post)
            # if inter_module:
            #     # For inter-module connections, use a much lower probability
            #     if (r_pass and v_pass and l_pass and np.random.rand() < 0.1):  # Additional 10% sampling
            #         elect_w = elect_proj.electrical_connection_instance_ws[0].weight
            #         e2 = G.add_edge(pop_map[elect_proj.presynaptic_population], 
            #                     pop_map[elect_proj.postsynaptic_population])
            #         # source_block = b[e2.source()]
            #         # target_block = b[e2.target()]
                    
            #         if r_pass and v_pass:
            #             eprop_width[e2] = elect_w * 5 # if source_block != target_block else elect_w * 4
            #             eprop_color[e2] = [1.0, 0.1, 0.1, 0.75]  
            #         elif r_pass:
            #             eprop_width[e2] = elect_w * 8  # if source_block != target_block else elect_w * 6
            #             eprop_color[e2] = [1.0, 0.45, 0.0, 0.5]  
            #         else:
             
            # elif (r_pass and v_pass and l_pass):  # For intra-module connections
            #     elect_w = elect_proj.electrical_connection_instance_ws[0].weight
            #     e2 = G.add_edge(pop_map[elect_proj.presynaptic_population], 
            #                 pop_map[elect_proj.postsynaptic_population])
            #     # source_block = b[e2.source()]
            #     # target_block = b[e2.target()]
            #     if r_pass and v_pass:
            #         eprop_width[e2] = elect_w * 5 # if source_block != target_block else elect_w * 4
            #         eprop_color[e2] = [1.0, 0.1, 0.1, 0.75]  
            #     elif r_pass:
            #         eprop_width[e2] = elect_w * 8  # if source_block != target_block else elect_w * 6
            #         eprop_color[e2] = [1.0, 0.45, 0.0, 0.5]  
            #     else:
            #         eprop_width[e2] = elect_w * 10
            #         eprop_color[e2] = [0.8, 0.0, 0.6, 0.45]
            #     edge_weight[e2] = elect_w
            #     edge_type[e2] = 'Proj'
            #     eprop_type[e2] = 'electrical'
            #     edge_count['electrical'] += 1     
    logging.info(f"Detected {edge_count['electrical']} elect_proj edges")

    input_edges = {'exc_input': 0, 'inh_input': 0}
    destination_stats = {}
    total_input_edges = 0
    if hasattr(nml_doc.networks[0], 'input_lists') and nml_doc.networks[0].input_lists:
        for ilist in nml_doc.networks[0].input_lists:
            pre = ilist.component
            post = ilist.populations
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
                        input_edges['exc_input'] += 1
                        input_w = getattr(input_item, 'weight', 1.0)
                        edge_weight[e3] = input_w 
                        eprop_width[e3] = input_w * 10
                        eprop_color[e3] = [0.8, 0.4, 0.0, 1.0]    # From #D7700A
                    elif destination == 'GABA':
                        e4 = G.add_edge(input_map[pre], pop_map[post])
                        total_input_edges += 1
                        eprop_name[e4] = destination
                        eprop_type[e4] = "GABA"
                        edge_type[e4] = "inh_input"
                        input_edges['inh_input'] += 1
                        input_w = getattr(input_item, 'weight', 1.0)
                        edge_weight[e4] = input_w
                        eprop_width[e4] = input_w * 10
                        eprop_color[e2] = [0.7, 0.1, 0.8, 1.0]    # From #A814D5
                    elif destination == 'GapJ':
                        # For GapJ, we need to determine type from the pre-component
                        if get_input_type(pre) == 'exc':
                            e5 = G.add_edge(input_map[pre], pop_map[post])
                            total_input_edges += 1
                            eprop_name[e5] = destination
                            eprop_type[e5] = "GapJ"
                            edge_type[e5] = "exc_input"
                            input_edges['exc_input'] += 1
                            input_w = getattr(input_item, 'weight', 1.0)
                            edge_weight[e5] = input_w
                            eprop_width[e5] = input_w * 5
                            eprop_color[e5] = [0.4, 0.9, 0.0, 1.0]    # From #69DB0B
                        else:
                            e6 = G.add_edge(input_map[pre], pop_map[post])
                            total_input_edges += 1
                            eprop_name[e6] = destination
                            eprop_type[e6] = "GapJ"
                            edge_type[e6] = "inh_input"
                            input_edges['inh_input'] += 1
                            input_w = getattr(input_item, 'weight', 1.0)
                            edge_weight[e6] = input_w
                            eprop_width[e6] = input_w * 5
                            eprop_color[e6] = [0.0, 0.0, 1.0, 1.0]    # From #0000FF
                except Exception as ex:
                    logging.warning(f"Failed to add input edge from {pre} to {post}: {ex}")
    # Log detailed statistics
    logging.info("Destination breakdown: %s", destination_stats)
    logging.info("Detected %d input edges: %d exc | %d inh", total_input_edges, input_edges['exc_input'], input_edges['inh_input'])

    # Step 3: Community detection first
    # state = gt.minimize_nested_blockmodel_dl(G)
    # gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))
    #--------------------------------------------------------------------#
    state = gt.NestedBlockState(G) # creates a state with an initial single-group hierarchy of depth ceil(log2(g.num_vertices()).
    gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10)) # equilibrate the Markov chain
    bs = [] # Initialize bs as a local list within the function
    def collect_partitions(s):
        nonlocal bs  # Use nonlocal to modify the bs from the enclosing scope
        bs.append(s.get_bs())  # Collect the partition state
    gt.mcmc_equilibrate(state, force_niter=100, mcmc_args=dict(niter=10), callback=collect_partitions)
    
    # Ensure bs is a proper list of partitions for ModeClusterState
    if not bs or len(bs) == 0:
        print("Warning: No partitions collected, exiting...")
        return
    
    #--------------------------------------------------------------------#
    pmode = gt.PartitionModeState(bs, nested=True, converge=True)  # Disambiguate partitions 
    pv = pmode.get_marginal(G) # obtain marginals
    bs_p = pmode.get_max_nested() # Get consensus estimate - store in different variable
    state = state.copy(bs=bs_p)
    state.draw(vertex_shape="pie", vertex_pie_fractions=pv, bg_color=[1, 1, 1, 1], output=f'{base_name}_lesmis-nested-sbm-marginals.png')
    #--------------------------------------------------------------------#
    h = [np.zeros(G.num_vertices() + 1) for s in state.get_levels()]
    def collect_num_groups(s):
        for l, sl in enumerate(s.get_levels()):
            B = sl.get_nonempty_B()
            h[l][B] += 1
    gt.mcmc_equilibrate(state, force_niter=100, mcmc_args=dict(niter=10), callback=collect_num_groups)
    for i in range(10):
        for j in range(100):
            state.multiflip_mcmc_sweep(niter=10)
        state.draw(output=f"{base_name}_lesmis-partition-sample-%i.png" % i, bg_color=[1, 1, 1, 1], empty_branches=False)
    #--------------------------------------------------------------------#
    # Use the original bs list for ModeClusterState
    pmode_c = gt.ModeClusterState(bs, nested=True) # Infer partition modes
    gt.mcmc_equilibrate(pmode_c, wait=1, mcmc_args=dict(niter=1, beta=np.inf)) # Minimize the mode state itself
    modes = pmode_c.get_modes() # Get inferred modes
    for i, mode in enumerate(modes):
        b = mode.get_max_nested()    # mode's maximum
        pv = mode.get_marginal(G)    # mode's marginal distribution
        print(f"Mode {i} with size {mode.get_M()/len(bs)}")
        state = state.copy(bs=b)
        state.draw(vertex_shape="pie", vertex_pie_fractions=pv, bg_color=[1, 1, 1, 1],
                output=f"{base_name}_lesmis-partition-mode-%i.png" % i)
    #--------------------------------------------------------------------#
    # Edge weights and covariates    
    state = gt.minimize_nested_blockmodel_dl(G, state_args=dict(recs=[edge_weight], rec_types=["discrete-binomial"]))    
    for i in range(100):
        ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

    state.draw(edge_color=eprop_color, ecmap=matplotlib.cm.plasma,
            eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 2, 8, power=1),
            edge_gradient=[],  bg_color=[1, 1, 1, 1], output=f"{base_name}_moreno-train-wsbm.png")
    #--------------------------------------------------------------------#
    # Model selection
    state = gt.minimize_nested_blockmodel_dl(G, state_args=dict(recs=[edge_weight], rec_types=["real-exponential"]))
    for i in range(100):
        ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

    state.draw(edge_color=gt.prop_to_size(edge_weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
            eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 1, 4, power=1, log=True),
            edge_gradient=[],  bg_color=[1, 1, 1, 1], output=f"{base_name}_wsbm.png")
    #--------------------------------------------------------------------#
    y = edge_weight.copy()
    y.a = np.log(y.a)
    state_ln = gt.minimize_nested_blockmodel_dl(G, state_args=dict(recs=[y], rec_types=["real-normal"]))
    # improve solution with merge-split
    for i in range(100):
        ret = state_ln.multiflip_mcmc_sweep(niter=10, beta=np.inf)

    state_ln.draw(edge_color=gt.prop_to_size(edge_weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
                eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 1, 4, power=1, log=True),
                edge_gradient=[],  bg_color=[1, 1, 1, 1], output=f"{base_name}_wsbm-lognormal.png")
    #--------------------------------------------------------------------#
    # Layered networks
    state = gt.minimize_nested_blockmodel_dl(G, state_args=dict(base_type=gt.LayeredBlockState, state_args=dict(ec=edge_weight, layers=True)))
    state.draw(edge_color=eprop_color, edge_gradient=[],
            ecmap=(matplotlib.cm.coolwarm_r, .6), edge_pen_width=5,  eorder=edge_weight,
            bg_color=[1, 1, 1, 1], output="tribes-sbm-edge-layers.png")
    # tree, prop, vprop = gt.get_hierarchy_tree(state)
    # ecount = tree.num_edges()
    # vcount = tree.num_vertices()
    # logging.info(f"Tree has {vcount} vertices and {ecount} edges")
    # levels = state.get_levels()
    # logging.info(f"Detected {len(levels)} hierarchy levels")
    # for s in levels:
    #     print(s)
    #     if s.get_N() == 1:
    #         break
    # b = levels[0].get_blocks() # Get block structure from highest level
    # Vprefixs = set(get_Vprefix(vprop_name[v]) for v in G.vertices())
    # Regions = set(get_Region(vprop_name[v]) for v in G.vertices())
    # Layers = set(get_layer(vprop_name[v]) for v in G.vertices())
    # logging.info(f"Layers: {Layers} | Regions: {Regions} | Vprefixs: {Vprefixs}")
    
    # V_intra = 0
    # V_inter = 0
    # R_intra = 0
    # R_inter = 0
    # L_intra = 0
    # L_inter = 0
    # for e in G.edges():
    #     pre = vprop_name[e.source()]
    #     post = vprop_name[e.target()]
    #     if get_Vprefix(pre) == get_Vprefix(post):
    #         V_intra += 1
    #     else:
    #         V_inter += 1
    #     if get_Region(pre) == get_Region(post):
    #         R_intra += 1
    #     else:
    #         R_inter += 1
    #     if get_layer(pre) == get_layer(post):
    #         L_intra += 1
    #     else:
    #         L_inter += 1
    # logging.info(f"Intra Vprefix-edges: {V_intra}, Inter Vprefix-edges: {V_inter}")
    # logging.info(f"Intra Region-edges: {R_intra}, Inter Region-edges: {R_inter}")
    # logging.info(f"Intra layer-edges: {L_intra}, Inter layer-edges: {L_inter}")
    
    # main_vertices = [v for v in G.vertices() if vprop_name[v] in pop_map.keys()]
    # main_edges = [e for e in G.edges() if e.source() in main_vertices and e.target() in main_vertices]
    # G_main = gt.GraphView(G, vfilt=lambda v: v in main_vertices, efilt=lambda e: e in main_edges)
    # input_vertices = [v for v in G.vertices() if vprop_name[v] in input_map.keys()]
    # ilist_edges = [e for e in G.edges() if e.source() in input_vertices or e.target() in input_vertices]
    # G_ilist = gt.GraphView(G, vfilt=lambda v: v in input_vertices, efilt=lambda e: e in ilist_edges)
    # full_vertices = [v for v in G.vertices()]
    # full_edges = [e for e in G.edges()]
    # G_full = gt.GraphView(G, vfilt=lambda v: v in full_vertices, efilt=lambda e: e in full_edges)
    
    # h = [np.zeros(G_full.num_vertices() + 1) for s in state.get_levels()]
    # def collect_num_groups(s):
    #     for l, sl in enumerate(s.get_levels()):
    #         B = sl.get_nonempty_B()
    #         h[l][B] += 1

    # if G_main.num_vertices() > 0 and G_main.num_edges() > 0:
    #     state_ndc_main = gt.minimize_nested_blockmodel_dl(G_main, state_args=dict(deg_corr=False))
    #     state_dc_main  = gt.minimize_nested_blockmodel_dl(G_main, state_args=dict(deg_corr=True))
    #     # logging.info("Number of edges: %d", G.num_edges())
    #     logging.info("Block counts (ndc): %s", np.unique(state_ndc_main.get_levels()[0].get_blocks().a, return_counts=True))
    #     logging.info("Block counts (dc): %s", np.unique(state_dc_main.get_levels()[0].get_blocks().a, return_counts=True))
        
    #     main_ndc_b = state_ndc_main.get_levels()[0].get_blocks()
    #     main_dc_b = state_dc_main.get_levels()[0].get_blocks()
    #     main_comm_ndc = len(set(main_ndc_b.a))
    #     main_comm_dc = len(set(main_dc_b.a))
    #     logging.info(f"main_vertices_ndc: {main_comm_ndc} communities | main_vertices_dc: {main_comm_dc} communities")
    #     unique_main_ndc_b = np.unique(main_ndc_b.a) if main_ndc_b is not None else np.array([])
    #     unique_main_dc_b = np.unique(main_dc_b.a) if main_dc_b is not None else np.array([])
    
    # if G_ilist.num_vertices() > 0 and G_ilist.num_edges() > 0:
    #     state_ndc_ilist = gt.minimize_nested_blockmodel_dl(G_ilist, state_args=dict(deg_corr=False))
    #     state_dc_ilist = gt.minimize_nested_blockmodel_dl(G_ilist, state_args=dict(deg_corr=True))
    #     logging.info("Non-degree-corrected ILIST:%s", state_ndc_ilist.entropy())
    #     logging.info("Degree-corrected ILIST:%s", state_dc_ilist.entropy())
    #     logging.info(u"ln \u039b:\t\t\t:%s", state_ndc_ilist.entropy() - state_dc_ilist.entropy())
    #     ilist_ndc_b = state_ndc_ilist.get_levels()[0].get_blocks()
    #     ilist_dc_b = state_dc_ilist.get_levels()[0].get_blocks()
    #     logging.info("ilist Block counts (ndc): %s", np.unique(ilist_ndc_b.a, return_counts=True))
    #     logging.info("ilist Block counts (dc): %s", np.unique(ilist_dc_b.a, return_counts=True))

    #     ilist_comm_ndc = len(set(ilist_ndc_b.a))
    #     ilist_comm_dc = len(set(ilist_dc_b.a))
    #     logging.info(f"input_vertices_ndc: {ilist_comm_ndc} communities | input_vertices_dc: {ilist_comm_dc} communities")
        
    #     unique_ilist_ndc_b = np.unique(ilist_ndc_b.a)
    #     unique_ilist_dc_b = np.unique(ilist_dc_b.a)

    # if G_full.num_vertices() > 0 and G_full.num_edges() > 0:
    #     state_ndc_full = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=False))
    #     state_dc_full  = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=True))
    #     logging.info("Non-degree-corrected FULL:%s", state_ndc_full.entropy())
    #     logging.info("Degree-corrected FULL:%s", state_dc_full.entropy())
    #     logging.info(u"ln \u039b:\t\t\t:%s", state_ndc_full.entropy() - state_dc_full.entropy())
    #     full_ndc_b = state_ndc_full.get_levels()[0].get_blocks()
    #     full_dc_b = state_dc_full.get_levels()[0].get_blocks()
    #     logging.info("full Block counts (ndc): %s", np.unique(full_ndc_b.a, return_counts=True))
    #     logging.info("full Block counts (dc): %s", np.unique(full_dc_b.a, return_counts=True))

    #     full_comm_ndc = len(set(full_ndc_b.a))
    #     full_comm_dc = len(set(full_dc_b.a))
    #     logging.info(f"full_vertices_ndc: {full_comm_ndc} communities | full_vertices_dc: {full_comm_dc} communities")
    #     unique_full_ndc_b = np.unique(full_ndc_b.a)
    #     unique_full_dc_b = np.unique(full_dc_b.a)

    # num_main_ndc_comm = len(unique_main_ndc_b)
    # num_main_dc_comm = len(unique_main_dc_b)
    # num_ilist_ndc_comm = len(unique_ilist_ndc_b)
    # num_ilist_dc_comm = len(unique_ilist_dc_b)
    # num_full_ndc_comm = len(unique_full_ndc_b)
    # num_full_dc_comm = len(unique_full_dc_b)
    # num_main_ndc_comm = len(unique_main_ndc_b)
    # num_main_dc_comm = len(unique_main_dc_b)
    # num_ilist_ndc_comm = len(unique_ilist_ndc_b)
    # num_ilist_dc_comm = len(unique_ilist_dc_b)
    # num_full_ndc_comm = len(unique_full_ndc_b)
    # num_full_dc_comm = len(unique_full_dc_b)

    # # Generate a more diverse extended color palette
    # extended_colors = []
    # num_extended_colors = max(num_ilist_ndc_comm, num_ilist_dc_comm, num_main_ndc_comm, num_main_dc_comm, num_full_ndc_comm, num_full_dc_comm)
    
    # # Create a more diverse color palette with variations in hue, saturation, and value
    # for i in range(num_extended_colors):
    #     hue = (i * 1.61803398875) % 1.0  # Use golden ratio for better color distribution
    #     saturation = 0.7 + (0.2 * np.sin(i))  # Vary saturation using sine wave
    #     value = 0.8 + (0.15 * np.cos(i))  # Vary value using cosine wave
    #     # Ensure saturation and value are within valid range
    #     saturation = max(0.6, min(0.9, saturation))
    #     value = max(0.7, min(0.95, value))
    #     rgb = matplotlib.colors.hsv_to_rgb([hue, saturation, value])
    #     extended_colors.append([rgb[0], rgb[1], rgb[2], 0.85])  # Slightly higher alpha for better visibility
    
    # # Assign community colors to vertices
    # input_vertices_set = set(input_vertices)
    # main_vertices_set = set(main_vertices)
    # full_vertices_set = set(full_vertices)
    
    # # Create a mapping from block IDs to color indices for better visual grouping
    # block_to_color = {}
    # for i, ndc_b_id in enumerate(unique_full_ndc_b):
    #     # Use modulo with extended colors for better color utilization
    #     ndc_color_idx = i % len(extended_colors)
    #     block_to_color[ndc_b_id] = ndc_color_idx
    # for j, dc_b_id in enumerate(unique_full_dc_b):
    #     dc_color_idx = j % len(extended_colors)
    #     block_to_color[dc_b_id] = dc_color_idx
    # for k, ndc_b_id in enumerate(unique_ilist_ndc_b):
    #     ndc_color_idx = k % len(extended_colors)
    #     block_to_color[ndc_b_id] = ndc_color_idx
    # for l, dc_b_id in enumerate(unique_ilist_dc_b):
    #     dc_color_idx = l % len(extended_colors)
    #     block_to_color[dc_b_id] = dc_color_idx
    # for m, ndc_b_id in enumerate(unique_main_ndc_b):
    #     ndc_color_idx = m % len(extended_colors)
    #     block_to_color[ndc_b_id] = ndc_color_idx
    # for n, dc_b_id in enumerate(unique_main_dc_b):
    #     dc_color_idx = n % len(extended_colors)
    #     block_to_color[dc_b_id] = dc_color_idx


    # for v in G.vertices():
    #     if v in input_vertices_set: 
    #         v2 = G_ilist.vertex(v)
    #         dc_b_id = ilist_dc_b[v2]
    #         dc_color_idx = block_to_color.get(dc_b_id, 0)
    #         # vprop_comm_color[v2] = extended_colors[dc_color_idx]
    #         ndc_b_id = ilist_ndc_b[v2]
    #         ndc_color_idx = block_to_color.get(ndc_b_id, 0)
    #         # vprop_comm_color[v2] = extended_colors[ndc_color_idx]

    #     elif v in main_vertices_set:
    #         # For main vertices, use the extended color palette
    #         v1 = G_main.vertex(v)
    #         dc_b_id = main_dc_b[v1]
    #         dc_color_idx = block_to_color.get(dc_b_id, 0)
    #         # vprop_comm_color[v1] = extended_colors[dc_color_idx]
    #         ndc_b_id = main_ndc_b[v1]
    #         ndc_color_idx = block_to_color.get(ndc_b_id, 0)
    #         # vprop_comm_color[v1] = extended_colors[ndc_color_idx]
    #     else:
    #         v = G_full.vertex(v)
    #         dc_b_id = full_dc_b[v]
    #         dc_color_idx = block_to_color.get(dc_b_id, 0)
    #         # vprop_comm_color[v] = extended_colors[dc_color_idx]
    #         ndc_b_id = full_dc_b[v]
    #         ndc_color_idx = block_to_color.get(ndc_b_id, 0)

    
    # Step 4: Create proper position property map with improved biological organization
    pos = G.new_vp("vector<double>")
    G.vp["pos"] = pos

    # Step 5: Fine-tune layout with sfdp using proper position initialization
    pos = gt.sfdp_layout(G, pos)
 
    # Step 6: Draw with community colors and edge types
    #############################################################################################################################
    # def animate_hierarchy(state, output_gif, frames=5, sweeps_per_frame=50, **draw_kwargs):
    #     frame_files = []
    #     for i in range(frames):
    #         for j in range(sweeps_per_frame):
    #             state.multiflip_mcmc_sweep(niter=10)
    #         frame_file = f"hierarchy_frame_{i:03d}.png"
    #         state.draw(output=frame_file, **draw_kwargs)
    #         frame_files.append(frame_file)
    #     # Create GIF
    #     cmd = ["convert", "-delay", "10", "-loop", "0"] + frame_files + [output_gif]
    #     subprocess.run(cmd, check=True)
    #     # Clean up
    #     for f in frame_files:
    #         if os.path.exists(f):
    #             os.remove(f)
    #     gc.collect()

    # state_ndc_full = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=False))
    # gt.mcmc_equilibrate(state_ndc_full, wait=10, mcmc_args=dict(niter=10))
    # for i in range(10):
    #     for j in range(100):
    #         state_ndc_full.multiflip_mcmc_sweep(niter=10)
    #     state_ndc_full.draw(
    #         bg_color=[0.98, 0.98, 0.98, 1],
    #         output=f"{base_name}_ndc_hierarchy_{i}.png",
    #         empty_branches=False,
    #         # Noneoverlapping=True,
    #     )
    # animate_hierarchy(state_ndc_full, output_gif=f"{base_name}_ndc_hierarchy.gif",
    #                         frames=5,
    #                         sweeps_per_frame=50, bg_color=[0.98, 0.98, 0.98, 1])
    
    # state_dc_full  = gt.minimize_nested_blockmodel_dl(G_full, state_args=dict(deg_corr=True))
    # gt.mcmc_equilibrate(state_dc_full, wait=10, mcmc_args=dict(niter=10))
    # for i in range(10):
    #     for j in range(100):
    #         state_dc_full.multiflip_mcmc_sweep(niter=10)
    #     state_dc_full.draw(
    #         bg_color=[0.98, 0.98, 0.98, 1],
    #         output=f"{base_name}_dc_hierarchy_{i}.png",
    #         empty_branches=False,
    #         # Noneoverlapping=True,
    #     )
    # animate_hierarchy(state_dc_full, output_gif=f"{base_name}_dc_hierarchy.gif",
    #                         frames=5,
    #                         sweeps_per_frame=50, bg_color=[0.98, 0.98, 0.98, 1])
    
    # state_hierarchy = gt.NestedBlockState(G,base_type=gt.RankedBlockState, state_args=dict(eweight=gt.contract_parallel_edges(G)))
    # gt.mcmc_equilibrate(state_hierarchy, force_niter=100, mcmc_args=dict(niter=10))
    
    # gt.mcmc_equilibrate(state_hierarchy, wait=10, mcmc_args=dict(niter=10))
    # for i in range(10):
    #     for j in range(100):
    #         state_hierarchy.multiflip_mcmc_sweep(niter=10,beta=np.inf)
    #     state_hierarchy.draw(pos=pos,
    #         # edge_color=gt.prop_to_size(edge_weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
    #         # eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 1, 4, power=1, log=True),
    #         # edge_gradient[], 
    #         bg_color=[0.98, 0.98, 0.98, 1],output=f"{base_name}_hierarchy_{i}.png" ,empty_branches=False)
    # animate_hierarchy(state_hierarchy,output_gif=f"{base_name}_hierarchy.gif",frames=5,sweeps_per_frame=50,bg_color=[0.98, 0.98, 0.98, 1])

    
    #############################################################################################################################
    # def animate_ghcp(state_ghcp, tpos, shape, cts, eprop_color, output_gif,frames=5, sweeps_per_frame=50, vertex_fill_color=None, 
    #                 vertex_size=None, **kwargs):
    #     frame_files = []
    #     G = state_ghcp.g
    #     # Get top-level block state
    #     top_level = state_ghcp.get_levels()[0]
    #     num_blocks = len(set(top_level.get_blocks().a))
    #     community_colors = [
    #         [
    #             0.5 + 0.5 * np.cos(2 * np.pi * i / max(num_blocks, 1)),
    #             0.5 + 0.5 * np.cos(2 * np.pi * (i / max(num_blocks, 1) + 1/3)),
    #             0.5 + 0.5 * np.cos(2 * np.pi * (i / max(num_blocks, 1) + 2/3)),
    #             0.8
    #         ]
    #         # for i in range(max(num_blocks, 1))
    #         for i in range(num_blocks)
    #     ]
    #     vprop_block_color = G.new_vp("vector<double>")
    #     # Get current block assignments from top level
        
    #     for i in range(frames):
    #         for j in range(sweeps_per_frame):
    #             state_ghcp.multiflip_mcmc_sweep(niter=10)
    #         blocks = top_level.get_blocks()
    #         for v in G.vertices():
    #             vprop_block_color[v] = community_colors[blocks[v] % len(community_colors)]

    #         frame_file = f"ghcp_frame_{i:03d}.png"
    #         gt.graph_draw(
    #             G, pos=tpos,
    #             vertex_shape=shape,
    #             edge_control_points=cts,
    #             edge_color=eprop_color,
    #             vertex_fill_color=vprop_block_color,
    #             vertex_size=vertex_size,
    #             bg_color=[0.98, 0.98, 0.98, 1],
    #             output=frame_file,
    #             **kwargs
    #         )
    #         frame_files.append(frame_file)
    #     # Create GIF
    #     cmd = ["convert", "-delay", "10", "-loop", "0"] + frame_files + [output_gif]
    #     subprocess.run(cmd, check=True)
    #     # Clean up
    #     for f in frame_files:
    #         if os.path.exists(f):
    #             os.remove(f)

    # g_ghcp = gt.GraphView(G, vfilt=gt.label_largest_component(G))
    # state_ghcp = gt.minimize_nested_blockmodel_dl(G, state_args=dict(recs=[edge_weight],rec_types=["discrete-binomial"]))
    # gt.mcmc_equilibrate(state_ghcp, wait=10, mcmc_args=dict(niter=10))
    # deg = G.degree_property_map("in")
    # deg.a = 2 * (np.sqrt(deg.a) * 0.5 + 0.4)
    # v_prop_map = G.new_vp("double")
    # for v in G.vertices():
    #     v_prop_map[v] = deg[v]
    # ebet = gt.betweenness(G)[1]
    # e_prop_map = G.new_ep("double")
    # for e in G.edges():
    #     e_prop_map[e] = ebet[e]
    # tree, prop_map, vprop = gt.get_hierarchy_tree(state_ghcp)
    # root = tree.vertex(tree.num_vertices() - 1, use_index=False)
    # tpos = gt.radial_tree_layout(tree, root, weighted=True)
    # cts = gt.get_hierarchy_control_points(G, tree, tpos)
    # shape = b.copy()
    # shape.a %= 14
    # gt.graph_draw(g_ghcp, pos=G.own_property(tpos), 
    #             vertex_fill_color=b, 
    #             vertex_shape=shape,edge_control_points=cts,edge_color=eprop_color,
    #             vertex_pen_width=2.5,vertex_anchor=0, 
    #             #   vertex_size=gt.prop_to_size(G.new_vp("int"),mi=5, ma=10),
    #             #   edge_size=gt.prop_to_size(G.new_ep("int"), mi=0.5, ma=1.5),
    #             bg_color=[0.98, 0.98, 0.98, 1],output=f"{base_name}_ghcp.png")
    # for i in range(100):
    #     ret = state_ghcp.multiflip_mcmc_sweep(niter=10, beta=np.inf)
    # state_ghcp.draw(edge_color=edge_weight.copy("double"), ecmap=matplotlib.cm.plasma,
    #                 eorder=edge_weight, edge_pen_width=gt.prop_to_size(edge_weight, 1, 4, power=1),
    #                 edge_gradient=[], bg_color=[0.98, 0.98, 0.98, 1],output=f"{base_name}_ghcp_wsbm.png")
    # animate_ghcp(state_ghcp, tpos, shape, cts, eprop_color,output_gif=f"{base_name}_ghcp.gif",frames=5, sweeps_per_frame=50)

if __name__ == "__main__":
    nml_net_files = [
        # "TC2CT.net.nml" ,
        # "TC2PT.net.nml",
        # "TC2IT4_IT2CT.net.nml",
        # "TC2IT2PTCT.net.nml",
        "max_CTC_plus.net.nml",
        # "M1a_max_plus.net.nml",
        # "M1_max_plus.net.nml",
        # "M2_max_plus.net.nml",
        "M2M1S1_max_plus.net.nml",
        # "S1bM1bM2b_max_plus.net.nml",
        # "M2aM1aS1a_max_plus.net.nml",
    ]
    
    file_paths = []
    hierarchical_plots = ['hierarchy','ghcp']
    subtypes = ['dc_hierarchy','ndc_hierarchy']
    
    plot_types = ['png','gif']
    for nml_net_file in nml_net_files:
        base_name = nml_net_file.split('.')[0]
        visualize_network(nml_net_file, V_intra=0.5, V_inter=0.5, R_intra=0.5,  R_inter=0.5, L_intra=0.5, L_inter=0.5, base_name=base_name)
        # for plot_type in plot_types:
        #     if 'ghcp':
        #         file_name = f"{base_name}_ghcp.{plot_type}" 
        #         file_paths.append(file_name)
        # if 'hierarchy':
        #     for i in range(10):
        #         file_name = f"{base_name}_hierarchy_{i}.png"
        #         file_paths.append(file_name)
        # elif 'dc_hierarchy':
        #     file_name = f"{base_name}_dc_hierarchy.gif" 
        #     file_paths.append(file_name)
        #     for i in range(10):
        #         file_name = f"{base_name}_dc_hierarchy_{i}.png"
        #         file_paths.append(file_name)
        # elif 'ndc_hierarchy':
        #     file_name = f"{base_name}_ndc_hierarchy.gif"
        #     file_paths.append(file_name)
        #     for i in range(10):
        #         file_name = f"{base_name}_ndc_hierarchy_{i}.png"
        #         file_paths.append(file_name)
                

    for file_path in file_paths:
        if os.path.exists(file_path):
            logging.info(f"File '{file_path}' size: {os.path.getsize(file_path)/(1024*1024):.2f} MB")
        else:
            logging.info(f"File '{file_path}' not found")