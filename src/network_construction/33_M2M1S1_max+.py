import random
import pyneuroml
from pyneuroml.pynml import write_neuroml2_file, read_neuroml2_file,validate_neuroml2
from pyneuroml.plot.PlotMorphology import plot_2D
from pyneuroml.plot.PlotMorphologyVispy import plot_interactive_3D
import neuroml
from neuroml import (IncludeType, ContinuousConnectionInstanceW,ContinuousProjection, Population, 
                     ElectricalProjection, Instance, Location,NeuroMLDocument,Property,InputW,InputList,
                     PulseGenerator,AlphaSynapse,ExpTwoSynapse,DoubleSynapse,BlockingPlasticSynapse, 
                     GapJunction, ElectricalConnectionInstanceW,Region,Network)
from neuroml.utils import component_factory
import neuroml.writers as writers
import os
from datetime import datetime
import threading
from threading import Lock
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from pyneuroml import pynml
from pyneuroml.lems import LEMSSimulation
import gc
from PIL import Image
from math import ceil
import itertools
import math
import logging
import glob
detailed_cell_files = glob.glob("*.cell.nml")
detailed_cell_files.sort()    
from collections import defaultdict
from scipy.stats import pearsonr
import random
import numpy as np
import cluster_ct_cells
import cluster_it_cells
import cluster_pt_cells
import cupy as cp
from extract_member_segment_count import get_actual_segment_count

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

def setup_gpu():
    """Configure GPU properly"""
    # Force CuPy to initialize CUDA
    cp.cuda.runtime.getDeviceCount()
    # Select GPU device 0 
    device = cp.cuda.Device(0)
    device.use()
    # Optimize memory allocation


def is_cortical_cell(pop_id):
    """Check if a population ID corresponds to cortical cells (PT, CT, IT) that lack segment ID 1."""
    cortical_prefixes = [
        'cADpyr229', 'cADpyr230', 'cADpyr231', 'cADpyr232',  # PT cells
        'bSTUT213', 'cSTUT189', 'dSTUT214',  # CT cells  
        'bAC217','bIR215', 'bNAC219','dNAC222', 'cNAC187', 'cACint209',  'cIR216',  # IT cells (c prefix)
    ]
    # Also check for layer-based naming patterns
    cortical_patterns = ['L1', 'L23', 'L4', 'L5', 'L6']
    
    pop_name = str(pop_id)
    # Check prefixes
    if any(pop_name.startswith(prefix) for prefix in cortical_prefixes):
        return True
    # Check for layer patterns (cortical cells often have layer identifiers)
    if any(pattern in pop_name for pattern in cortical_patterns):
        return True
    return False

def is_thalamic_cell(pop_id):
    """Check if a population ID corresponds to thalamic cells (TC) that have segment ID 1."""
    thalamic_prefixes = [
        'TCR', 'nRT'
    ]
    pop_name = str(pop_id)
    return any(pop_name.startswith(prefix) for prefix in thalamic_prefixes)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

now = datetime.now()
print(f"Network structure (NeuroML 2beta5) for project: S1-->M1-->M2<-->M2-->M1-->S1 network saved with pyNeuroML on {now.strftime('%H:%M:%S %Y-%m-%d')}")
nml_doc = component_factory(NeuroMLDocument,id="M2M1S1_max_plus")
net = nml_doc.add("Network", id="M2M1S1_max_plus", type="networkWithTemperature",validate=False)
net.temperature="34.0degC"
net.neuro_lex_id="sao864921383"

# %%
clusters, pt_cells = cluster_pt_cells.get_pt_clusters()
PT_cell_files = [cell['source_file'] for cell in pt_cells if cell['source_file']]

# Initialize empty lists for PTs grouped by layer
PTs_L23 = []
PTs_L4 = []
PTs_L5 = []
PTs_L6 = []
layer_to_param = {'L1': 52, 'L23': 36, 'L4': 21, 'L5': 25, 'L6': 52}
for cell in pt_cells:
    base_name = cell['original_name'].replace('.cell.nml', '')
    layer_param = layer_to_param.get(cell['layer'], 5)  # default to 5 if layer not found
    cell_tuple = (base_name, layer_param, 2)
    
    if cell['layer'] == 'L23':
        PTs_L23.append(cell_tuple)
    elif cell['layer'] == 'L4':
        PTs_L4.append(cell_tuple)
    elif cell['layer'] == 'L5':
        PTs_L5.append(cell_tuple)
    elif cell['layer'] == 'L6':
        PTs_L6.append(cell_tuple)
PTs_L23.sort()
PTs_L4.sort()
PTs_L5.sort()
PTs_L6.sort()
print(f"PTs_L23: {len(PTs_L23)}")
print(f"PTs_L4: {len(PTs_L4)}")
print(f"PTs_L5: {len(PTs_L5)}")
print(f"PTs_L6: {len(PTs_L6)}")
PTs = PTs_L23 + PTs_L4 + PTs_L5 + PTs_L6

clusters, all_it_cells = cluster_it_cells.get_it_clusters_from_catalog()

# Create IT_cell_files list from the clustered data
IT_cell_files = [cell['source_file'] for cell in all_it_cells if cell['source_file']]

ITs_bAC = []
ITs_bIR = []
ITs_bNAC = []
ITs_cACint = []
ITs_cIR = []
ITs_cNAC = []
ITs_dNAC = []
IT4_bAC = []
IT4_bIR = []
IT4_bNAC = []
IT4_cACint = []
IT4_cIR = []
IT4_cNAC = []
IT4_dNAC = []
layer_to_param = {
    'L1': 52, 'L23': 36, 'L4': 21, 'L5': 25, 'L6': 52
}
for cell in all_it_cells:
    base_name = cell['original_name'].replace('.cell.nml', '')
    layer_param = layer_to_param.get(cell['layer'], 5)  # default to 5 if layer not found
    cell_tuple = (base_name, layer_param, 2)
    if cell['prefix'].startswith('bAC'):
        ITs_bAC.append(cell_tuple)
        if cell['layer'] == 'L4':
            IT4_bAC.append(cell_tuple)
    elif cell['prefix'].startswith('bIR'):
        ITs_bIR.append(cell_tuple)
        if cell['layer'] == 'L4':
            IT4_bIR.append(cell_tuple)
    elif cell['prefix'].startswith('bNAC'):
        ITs_bNAC.append(cell_tuple)
        if cell['layer'] == 'L4':
            IT4_bNAC.append(cell_tuple)
    elif cell['prefix'].startswith('cACint'):
        ITs_cACint.append(cell_tuple)
        if cell['layer'] == 'L4':
            IT4_cACint.append(cell_tuple)
    elif cell['prefix'].startswith('cIR'):
        ITs_cIR.append(cell_tuple)
        if cell['layer'] == 'L4':
            IT4_cIR.append(cell_tuple)
    elif cell['prefix'].startswith('cNAC'):
        ITs_cNAC.append(cell_tuple)
        if cell['layer'] == 'L4':
            IT4_cNAC.append(cell_tuple)
    elif cell['prefix'].startswith('dNAC'):
        ITs_dNAC.append(cell_tuple)
        if cell['layer'] == 'L4':
            IT4_dNAC.append(cell_tuple)

IT4_bAC.sort()
IT4_bIR.sort()
IT4_bNAC.sort()
IT4_cACint.sort()
IT4_cIR.sort()
IT4_cNAC.sort()
IT4_dNAC.sort()
IT4_c = IT4_cACint + IT4_cIR + IT4_cNAC
IT4_b = IT4_bAC + IT4_bIR + IT4_bNAC
IT4_d = IT4_dNAC
IT4 = IT4_c + IT4_b +  IT4_d
print(f"IT4: {len(IT4)} cells")
ITs_bAC.sort()
ITs_bIR.sort()
ITs_bNAC.sort()
ITs_cACint.sort()
ITs_cIR.sort()
ITs_cNAC.sort()
ITs_dNAC.sort()
ITs_b = ITs_bAC + ITs_bIR + ITs_bNAC
ITs_c = ITs_cACint + ITs_cIR + ITs_cNAC
ITs_d = ITs_dNAC
ITs = ITs_b + ITs_c + ITs_d
print(f"ITs: {len(ITs)} cells")

ct_clusters, ct_cells = cluster_ct_cells.get_ct_clusters_from_catalog()
CT_cell_files = [cell['source_file'] for cell in ct_cells if cell['source_file']]

CTs_bSTUT = []
CTs_cSTUT = []
CTs_dSTUT = []
for cell in ct_cells:
    base_name = cell['original_name'].replace('.cell.nml', '')
    layer_param = layer_to_param.get(cell['layer'], 5)  # default to 5 if layer not found
    cell_tuple = (base_name, layer_param, 2)  # (base_name, layer_param, axonal_number=2)
    
    if cell['prefix'].startswith('bSTUT'):
        CTs_bSTUT.append(cell_tuple)
    elif cell['prefix'].startswith('cSTUT'):
        CTs_cSTUT.append(cell_tuple)
    elif cell['prefix'].startswith('dSTUT'):
        CTs_dSTUT.append(cell_tuple)

# Sort lists to maintain consistent ordering (optional but recommended)
CTs_bSTUT.sort()
CTs_cSTUT.sort()
CTs_dSTUT.sort()
print(f"CTs_bSTUT: {len(CTs_bSTUT)} cells")
print(f"CTs_cSTUT: {len(CTs_cSTUT)} cells")  
print(f"CTs_dSTUT: {len(CTs_dSTUT)} cells")
CTs = CTs_bSTUT + CTs_cSTUT + CTs_dSTUT

TCs_intralaminar = [("TCRil",2000, 6),("nRTil",2000, 6)]
TCs_matrix = [("TCR", 6000, 6), ("TCRm", 4000, 6),("nRTm", 4000, 6)]
TCs_core = [("nRT",6000, 6),("TCRc", 4000, 6),("nRTc", 4000, 6)]
TCs = TCs_matrix + TCs_core + TCs_intralaminar


lock = threading.Lock()
continuous_proj_counter = 0
electrical_proj_counter = 0
counter_lock = threading.Lock()
# Global dictionary to track actual connections
actual_connection_stats = defaultdict(int)
synaptic_contact_stats = defaultdict(int)
electrical_conductance_stats = defaultdict(int)

connection_stats_lock = threading.Lock()  # For thread safety
synaptic_stats_lock = threading.Lock()
electrical_stats_lock = threading.Lock()
electrical_stats_updated = False
electrical_stats_finalized = False

def classify_cells():
    """Create sets for quick cell type identification"""
    exc_cells = set(cell[0] for cell in PTs + ITs_c + CTs_cSTUT + TCs_matrix)
    inh_cells = set(cell[0] for cell in ITs_b + ITs_d + CTs_bSTUT + CTs_dSTUT + TCs_intralaminar + TCs_core)
    return exc_cells, inh_cells

EXC_CELLS, INH_CELLS = classify_cells()

def update_synaptic_contact_stats(pre_cell_id, post_cell_id, count):
    """Update SYNAPTIC CONTACT statistics in a thread-safe manner"""
    pre_is_exc = pre_cell_id in EXC_CELLS
    pre_is_inh = pre_cell_id in INH_CELLS
    post_is_exc = post_cell_id in EXC_CELLS
    post_is_inh = post_cell_id in INH_CELLS
    
    with synaptic_stats_lock:
        if pre_is_exc and post_is_exc:
            synaptic_contact_stats['E-E(syn)'] += count
        elif pre_is_exc and post_is_inh:
            synaptic_contact_stats['E-I(syn)'] += count
        elif pre_is_inh and post_is_exc:
            synaptic_contact_stats['I-E(syn)'] += count
        elif pre_is_inh and post_is_inh:
            synaptic_contact_stats['I-I(syn)'] += count
        synaptic_contact_stats['total(syn)'] += count

persistent_electrical_stats = {
    'E-E(ele)': 0,
    'E-I(ele)': 0,
    'I-E(ele)': 0,
    'I-I(ele)': 0,
    'total(ele)': 0
}
def update_electrical_conductance_stats(pre_cell_id, post_cell_id, count):
    """Update electrical conductance statistics in a thread-safe manner"""
    global electrical_stats_updated

    pre_is_exc = pre_cell_id in EXC_CELLS
    pre_is_inh = pre_cell_id in INH_CELLS
    post_is_exc = post_cell_id in EXC_CELLS
    post_is_inh = post_cell_id in INH_CELLS
    
    with electrical_stats_lock:
        electrical_stats_updated = True
        if pre_is_exc and post_is_exc:
            electrical_conductance_stats['E-E(ele)'] += count
            persistent_electrical_stats['E-E(ele)'] += count
        elif pre_is_exc and post_is_inh:
            electrical_conductance_stats['E-I(ele)'] += 0
            persistent_electrical_stats['E-I(ele)'] += 0
        elif pre_is_inh and post_is_exc:
            electrical_conductance_stats['I-E(ele)'] += 0
            persistent_electrical_stats['I-E(ele)'] += 0
        elif pre_is_inh and post_is_inh:
            electrical_conductance_stats['I-I(ele)'] += count
            persistent_electrical_stats['I-I(ele)'] += count
        electrical_conductance_stats['total(ele)'] += count
        persistent_electrical_stats['total(ele)'] += count

def get_final_electrical_stats():
    """Get the final electrical statistics, prioritizing persistent stats"""
    with electrical_stats_lock:
        return {
            'total(ele)': persistent_electrical_stats['total(ele)'],
            'E-E(ele)': persistent_electrical_stats['E-E(ele)'],
            'E-I(ele)': persistent_electrical_stats['E-I(ele)'],
            'I-E(ele)': persistent_electrical_stats['I-E(ele)'],
            'I-I(ele)': persistent_electrical_stats['I-I(ele)']
        }


def print_separate_connection_stats():
    """Print connection statistics separately for synaptic and electrical conductances"""
    # Synaptic contacts
    print("\n" + "="*50)
    print("SYNAPTIC CONTACT STATISTICS")
    print("="*50)
    syn_total = synaptic_contact_stats['total(syn)'] or 1
    print(f"E-E (syn Exc->Exc): {synaptic_contact_stats['E-E(syn)']} connections ({synaptic_contact_stats['E-E(syn)']/syn_total*100:.2f}%)")
    print(f"E-I (syn Exc->Inh): {synaptic_contact_stats['E-I(syn)']} connections ({synaptic_contact_stats['E-I(syn)']/syn_total*100:.2f}%)")
    print(f"I-E (syn Inh->Exc): {synaptic_contact_stats['I-E(syn)']} connections ({synaptic_contact_stats['I-E(syn)']/syn_total*100:.2f}%)")
    print(f"I-I (syn Inh->Inh): {synaptic_contact_stats['I-I(syn)']} connections ({synaptic_contact_stats['I-I(syn)']/syn_total*100:.2f}%)")
    print(f"Total synaptic contacts: {synaptic_contact_stats['total(syn)']}")
    
    # Electrical conductances
    print("\n" + "="*50)
    print("ELECTRICAL CONDUCTANCE STATISTICS")
    print("="*50)
    
    # Get final stats using our new function
    final_stats = get_final_electrical_stats()
    ele_total = final_stats['total(ele)']
    ele_ee = final_stats['E-E(ele)']
    ele_ei = final_stats['E-I(ele)']
    ele_ie = final_stats['I-E(ele)']
    ele_ii = final_stats['I-I(ele)']
    
    # Also get current stats for comparison
    with electrical_stats_lock:
        orig_total = electrical_conductance_stats['total(ele)']
        flag_status = electrical_stats_updated
        finalized_status = electrical_stats_finalized
    
    print(f"Stats tracking flag: {flag_status}")
    print(f"Stats finalized flag: {finalized_status}")
    print(f"Persistent stats show: {persistent_electrical_stats['total(ele)']} total connections")
    print(f"Current dict shows: {orig_total} total connections")
    
    # Display final stats - always use persistent stats
    if persistent_electrical_stats['total(ele)'] > 0:
        pers_total = persistent_electrical_stats['total(ele)']
        pers_total_safe = pers_total or 1
        pers_ee = persistent_electrical_stats['E-E(ele)']
        pers_ei = persistent_electrical_stats['E-I(ele)']
        pers_ie = persistent_electrical_stats['I-E(ele)']
        pers_ii = persistent_electrical_stats['I-I(ele)']
        
        print(f"E-E (ele Exc->Exc): {pers_ee} connections ({pers_ee/pers_total_safe*100:.2f}%)")
        print(f"E-I (ele Exc->Inh): {pers_ei} connections ({pers_ei/pers_total_safe*100:.2f}%)")
        print(f"I-E (ele Inh->Exc): {pers_ie} connections ({pers_ie/pers_total_safe*100:.2f}%)")
        print(f"I-I (ele Inh->Inh): {pers_ii} connections ({pers_ii/pers_total_safe*100:.2f}%)")
        print(f"Total electrical conductances: {pers_total}")

    # Display final stats
    if ele_total > 0:
        ele_total_safe = ele_total or 1
        print(f"E-E (ele Exc->Exc): {ele_ee} connections ({ele_ee/ele_total_safe*100:.2f}%)")
        print(f"E-I (ele Exc->Inh): {ele_ei} connections ({ele_ei/ele_total_safe*100:.2f}%)")
        print(f"I-E (ele Inh->Exc): {ele_ie} connections ({ele_ie/ele_total_safe*100:.2f}%)")
        print(f"I-I (ele Inh->Inh): {ele_ii} connections ({ele_ii/ele_total_safe*100:.2f}%)")
        print(f"Total electrical conductances: {ele_total}")
    else:
        print("WARNING: No electrical conductances were created or statistics not updated")
    
    # Combined total
    overall_total = synaptic_contact_stats['total(syn)'] + ele_total
    overall_ee = synaptic_contact_stats['E-E(syn)'] + ele_ee
    overall_ei = synaptic_contact_stats['E-I(syn)'] + ele_ei
    overall_ie = synaptic_contact_stats['I-E(syn)'] + ele_ie
    overall_ii = synaptic_contact_stats['I-I(syn)'] + ele_ii
    print("\n" + "="*50)
    print("OVERALL CONNECTION STATISTICS")
    print("="*50)
    if overall_total > 0:
        syn_conn = synaptic_contact_stats['total(syn)']
        ele_conn = max(persistent_electrical_stats['total(ele)'], ele_total)
        print(f"Synaptic contacts: {syn_conn} ({syn_conn/overall_total*100:.2f}% of total)")
        print(f"Electrical conductances: {ele_conn} ({ele_conn/overall_total*100:.2f}% of total)")
    else:
        print(f"Synaptic contacts: {synaptic_contact_stats['total(syn)']} (0.00% of total)")
        print(f"Electrical conductances: 0 (0.00% of total)")
    print(f"Overall connections: {overall_total}") 
    print(f"Overall E-E (syn + ele Exc->Exc): {overall_ee} ({overall_ee/overall_total*100:.2f}%)") 
    print(f"Overall E-I (syn Exc->Inh): {overall_ei} ({overall_ei/overall_total*100:.2f}%)")
    print(f"Overall I-E (syn Inh->Exc): {overall_ie} ({overall_ie/overall_total*100:.2f}%)")
    print(f"Overall I-I (syn + ele Inh->Inh): {overall_ii} ({overall_ii/overall_total*100:.2f}%)")
    print("="*50)     

def reset_electrical_stats_tracking():
    """Reset the electrical stats tracking flag"""
    global electrical_stats_updated
    electrical_stats_updated = False
    with electrical_stats_lock:
        for key in electrical_conductance_stats.keys():
            electrical_conductance_stats[key] = 0

# %%
regions = {
    'S1a': Region(id='S1a'),
    'S1b': Region(id='S1b'),
    'M1a': Region(id='M1a'),
    'M1b': Region(id='M1b'),
    'M2a': Region(id='M2a'),
    'M2b': Region(id='M2b')
}

# Base populations
base_populations = [TCs_intralaminar, TCs_matrix, ITs, PTs, CTs, TCs_core]
m2_populations = [TCs_intralaminar, TCs_matrix, ITs, PTs, CTs]  # M2 regions don't have TCs_core

# Create individual networks
networks = {}
networks['S1a'] = Network(id="S1a", regions=[regions['S1a']], populations=base_populations)
networks['S1b'] = Network(id='S1b', regions=[regions['S1b']], populations=base_populations)
networks['M1a'] = Network(id='M1a', regions=[regions['M1a']], populations=base_populations)
networks['M1b'] = Network(id='M1b', regions=[regions['M1b']], populations=base_populations)
networks['M2a'] = Network(id='M2a', regions=[regions['M2a']], populations=m2_populations)
networks['M2b'] = Network(id='M2b', regions=[regions['M2b']], populations=m2_populations)

# Create loop networks
S1_loop = Network(id="S1a_S1b", regions=[regions['S1a'], regions['S1b']], 
                  populations=[networks['S1a'].populations, networks['S1b'].populations])
M1_loop = Network(id="M1a_M1b", regions=[regions['M1a'], regions['M1b']], 
                  populations=[networks['M1a'].populations, networks['M1b'].populations])
M2_loop = Network(id="M2a_M2b", regions=[regions['M2a'], regions['M2b']], 
                  populations=[networks['M2a'].populations, networks['M2b'].populations])
M2aM1a_loop = Network(id="M2a_M1a",regions=[regions['M2a'], regions['M1a']],populations=[networks['M2a'].populations,networks['M1a'].populations])
M1aS1a_loop = Network(id="M1a_S1a",regions=[regions['M1a'], regions['S1a']],populations=[networks['M1a'].populations,networks['S1a'].populations])                      
M2aM1aS1a_loop = Network(id="M2a_M1a_S1a",regions=[regions['M2a'], regions['M1a'], regions['S1a']],populations=[networks['M2a'].populations,networks['M1a'].populations,networks['S1a'].populations])
S1bM1b_loop = Network(id="S1b_M1b",regions=[regions['S1b'], regions['M1b']],populations=[networks['S1b'].populations,networks['M1b'].populations])
M1bM2b_loop = Network(id="M1b_M2b",regions=[regions['M1b'], regions['M2b']],populations=[networks['M1b'].populations,networks['M2b'].populations])
S1bM1bM2b_loop = Network(id="S1b_M1b_M2b",regions=[regions['S1b'], regions['M1b'], regions['M2b']],populations=[networks['S1b'].populations,networks['M1b'].populations,networks['M2b'].populations])

M1S1_loop = Network(id="M1a_S1a_S1b_M1b", 
                    regions=[regions['M1a'], regions['S1a'], regions['S1b'], regions['M1b']],
                    populations=[S1_loop.populations, M1_loop.populations])

M2M1_loop = Network(id="M2a_M1a_M1b_M2b",
                    regions=[regions['M2a'], regions['M1a'], regions['M1b'], regions['M2b']],
                    populations=[M1_loop.populations, M2_loop.populations])

M2M1S1_loop = Network(id="M2a_M1a_S1a_S1b_M1b_M2b",
                      regions=[regions['M2a'], regions['M1a'], regions['S1a'], regions['S1b'], regions['M1b'], regions['M2b']],
                      populations=[S1_loop.populations, M1_loop.populations, M2_loop.populations])

# %%
def generate_attractive_color():
    hue = random.random()
    saturation = random.uniform(0.4, 0.8)  # Medium saturation
    lightness = random.uniform(0.5, 0.9)   # Medium to light value
    def hsl_to_rgb(h, s, l):
        c = (1 - abs(2*l - 1)) * s
        x = c * (1 - abs((h*3) % 2 - 1))
        m = l - c/2
        if h < 1/6:
            return (c + m, x + m, m)
        elif h < 1/3:
            return (x + m, c + m, m)
        elif h < 0.5:
            return (m, c + m, x + m)
        elif h < 2/3:
            return (m, x + m, c + m)
        elif h < 5/6:
            return (x + m, m, c + m)
        else:
            return (c + m, m, x + m)
    r, g, b = hsl_to_rgb(hue, saturation, lightness)
    return f"{r:.1f} {g:.1f} {b:.1f}"

def get_layer_id(updated_cellID):
    if 'nRT' in updated_cellID or 'nRTc' in updated_cellID or 'TCRc' in updated_cellID:
        return "TCs_core"
    elif 'TCR' in updated_cellID or 'TCRm' in updated_cellID or 'nRTm' in updated_cellID:
        return "TCs_matrix"
    elif 'TCRil' in updated_cellID or 'nRTil' in updated_cellID:
        return "TCs_intralaminar"
    for layer in ['L1', 'L23', 'L4', 'L5', 'L6', 'TCs_intralaminar', 'TCs_matrix', 'TCs_core']:
        if layer in updated_cellID:
            return layer
    print(f"[WARN] Skipping population with unknown layer: {updated_cellID}")
    return None

def get_layer_coordinates(LayerID):
    layer_positions = {
        'L1': {'y': 297, 'height': 165},   # Thickness from circuit.json
        'L23': {'y': 245, 'height': 149 + 353},  # L2 + L3 combined
        'L4': {'y': 225, 'height': 190},
        'L5': {'y': 170, 'height': 525},
        'L6': {'y': 100, 'height': 700},
        'TCs_core': {'y': 0, 'height': 100},
        'TCs_matrix': {'y': 0, 'height': 50},
        'TCs_intralaminar': {'y': 0, 'height': 25},
    }
    return layer_positions.get(LayerID, {'y': 0, 'height': 50})

def get_region_coordinates(RegionID):
    """Get coordinates for each region including a/b subdivisions"""
    region_positions = {
        'M2a': {'x': 100, 'width': 50},
        'M2b': {'x': 150, 'width': 50},
        'M1a': {'x': 50, 'width': 100},
        'M1b': {'x': 200, 'width': 100},
        'S1a': {'x': 0, 'width': 150},
        'S1b': {'x': 250, 'width': 150}
    }
    return region_positions.get(RegionID, {'x': 0, 'width': 50})

def get_cell_parameters(cellID):
    if 'TCRil' in cellID or 'nRTil' in cellID:
        return 2000, 6
    elif 'TCR' in cellID or 'nRT' in cellID:
        return 6000, 6
    elif 'L1' in cellID:
        return 5, 2
    elif 'L23' in cellID:
        return 36, 2
    elif 'L4' in cellID:
        return 21, 2
    elif 'L5' in cellID:
        return 25, 2
    elif 'L6' in cellID:
        return 52, 2
    else:
        return 4000, 6 

# %%
def modify_cell_files(detailed_cell_files):
    modified_cells = {}
    processed_cells = set() 
    for detailed_cell_file in detailed_cell_files:
        cell_doc = read_neuroml2_file(detailed_cell_file)
        cellID = cell_doc.cells[0].id

        pop_size, axonalNum = get_cell_parameters(cellID)
        modified_cells[cellID] = {
            'cellID': cellID,
            'size': pop_size,
            'axons': axonalNum,
        }
        processed_cells.add(cellID)

        nml_doc.add(IncludeType, href=detailed_cell_file)

    return modified_cells

# %%
def create_population_instances(region, cell_params, start_index, lock):
    RegionID = region.id
    cellID = cell_params[0]  
    pop_size = cell_params[1]
    axonalNum = cell_params[2]

    specific_pop = Population(
        id=f'{RegionID}_{cellID}',
        component=cellID, 
        type="populationList",
        size=pop_size
    )
    layer_id = get_layer_id(cellID)
    color = generate_attractive_color()
    specific_pop.add(Property, tag="color", value=color)
    specific_pop.add(Property, tag="space", value=f'{RegionID}_{layer_id}')
    
    region_pos = get_region_coordinates(RegionID)
    layer_pos = get_layer_coordinates(layer_id)
    for i in range(pop_size):
        instance = Instance(id=i)
        location = Location(
            x=f"{region_pos['x'] + random.random() * region_pos['width']}",
            y=f"{layer_pos['y'] + random.random() * layer_pos['height']}",
            z=f"{random.random() * 100}"
        )
        instance.location = location
        specific_pop.instances.append(instance)
    created_populations[specific_pop.id] = specific_pop
    return specific_pop, pop_size

created_populations = {}
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    current_index = 0
    lock = threading.Lock()
    
    for cellID, cell_params in modify_cell_files(detailed_cell_files).items():
        for region in M2M1S1_loop.regions:
            futures.append(executor.submit(
                create_population_instances,
                region,
                (cellID, cell_params['size'], cell_params['axons']),
                current_index,
                lock
            ))
            current_index += cell_params['size']
    for future in tqdm(as_completed(futures), total=len(futures), desc="Creating populations"):
        pop, size = future.result()
        if pop is not None:
            net.populations.append(pop)

# %%
Syn_AMPA = ExpTwoSynapse(id=f'AMPA', gbase="0.367879nS", erev="0mV", tau_decay="2ms", tau_rise="1ms")
Syn_NMDA = BlockingPlasticSynapse(id=f'NMDA',gbase="0.8nS", erev="0mV",tau_decay="13.3ms", tau_rise="1ms")
Syn_GABAa = ExpTwoSynapse(id=f'GABAa', gbase="0.057nS", erev="-75mV", tau_decay="39ms", tau_rise="8ms")
Syn_GABAb = ExpTwoSynapse(id=f'GABAb', gbase="0.0169139nS", erev="-81mV", tau_decay="200ms", tau_rise="180ms")
Syn_GapJ = GapJunction(id=f'GapJ',conductance="3nS")

Syns = [Syn_AMPA,Syn_NMDA,Syn_GABAa,Syn_GABAb,Syn_GapJ]
for Syn in Syns:
    SynID = Syn.id
    nml_doc_Syn=NeuroMLDocument(id=f'{SynID}_doc')
    nml_doc_Syn.add(Syn)
    nml_Syn_file = f'Syn_{SynID}.synapse.nml'
    writers.NeuroMLWriter.write(nml_doc_Syn,nml_Syn_file)
    print(Syn)
    nml_doc.add(Syn)
DoubleSyns= [
    ("AMPA_NMDA", Syn_AMPA,Syn_NMDA,"./AMPA","./NMDA"),
    ("GABA",Syn_GABAa,Syn_GABAb,"./GABAa","./GABAb"),
]
for DoubleSynID,Syn1,Syn2,Syn1_path,Syn2_path in DoubleSyns:
    nml_doc_DoubleSyn = NeuroMLDocument(id=f'{DoubleSynID}_doc')
    nml_doc_DoubleSyn.add(Syn1)
    nml_doc_DoubleSyn.add(Syn2)
    nml_DoubleSyn_file =f'Syn_{DoubleSynID}.synapse.nml'
    DoubleSyn=DoubleSynapse(
        id=DoubleSynID,
        synapse1=Syn1.id,
        synapse2=Syn2.id,
        synapse1_path=Syn1_path,
        synapse2_path=Syn2_path,
    )
    writers.NeuroMLWriter.write(nml_doc_DoubleSyn,nml_DoubleSyn_file)
    print(DoubleSyn)
    nml_doc.add(DoubleSyn)

# %%
Continuous_Projections = [
    (M2M1S1_loop.regions,PTs,ITs + PTs + TCs_intralaminar + TCs_matrix, "AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (M2M1S1_loop.regions,ITs_c, PTs + CTs,"AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (M2M1S1_loop.regions,ITs_b + ITs_d, PTs + CTs,"GABA",Syn_GABAa.id,Syn_GABAb.id),
    (M2M1S1_loop.regions,CTs_cSTUT, ITs + CTs + TCs_intralaminar + TCs_matrix, "AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (M2M1S1_loop.regions,CTs_bSTUT + CTs_dSTUT, ITs + CTs + TCs_intralaminar + TCs_matrix, "GABA",Syn_GABAa.id,Syn_GABAb.id),
    (M2M1S1_loop.regions,TCs_intralaminar + TCs_matrix,ITs + TCs_intralaminar + TCs_matrix + PTs,"AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (S1bM1b_loop.regions,TCs_intralaminar + TCs_core,ITs + TCs_intralaminar + TCs_core + PTs + ITs ,"GABA",Syn_GABAa.id,Syn_GABAb.id),
    (S1bM1b_loop.regions,CTs_cSTUT,TCs_intralaminar + TCs_core, "AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (S1bM1b_loop.regions,CTs_bSTUT + CTs_dSTUT,TCs_intralaminar + TCs_core, "GABA",Syn_GABAa.id,Syn_GABAb.id),
    (S1bM1b_loop.regions,ITs_c,ITs,"AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (M1bM2b_loop.regions,ITs_c,ITs,"AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (M2aM1a_loop.regions,ITs_c,ITs,"AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (M1aS1a_loop.regions,ITs_c,ITs,"AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (M2_loop.regions,ITs_c,ITs,"AMPA_NMDA",Syn_AMPA.id,Syn_NMDA.id),
    (S1bM1b_loop.regions,ITs_d + ITs_b,ITs,"GABA",Syn_GABAa.id,Syn_GABAb.id),
    (M1bM2b_loop.regions,ITs_d + ITs_b,ITs,"GABA",Syn_GABAa.id,Syn_GABAb.id),
    (M2aM1a_loop.regions,ITs_d + ITs_b,ITs,"GABA",Syn_GABAa.id,Syn_GABAb.id),
    (M1aS1a_loop.regions,ITs_d + ITs_b,ITs,"GABA",Syn_GABAa.id,Syn_GABAb.id),
    (M2_loop.regions,ITs_d + ITs_b,ITs,"GABA",Syn_GABAa.id,Syn_GABAb.id),
]

Electrical_Projections = [
    (M2M1S1_loop.regions, TCs_intralaminar + TCs_matrix, TCs_intralaminar + TCs_matrix, "GapJ"),
    (M1S1_loop.regions, TCs_intralaminar + TCs_core, TCs_intralaminar + TCs_core, "GapJ"),
    (M2M1S1_loop.regions, PTs, PTs, "GapJ"),
    (M2M1S1_loop.regions, CTs, CTs, "GapJ"),
    (M2M1S1_loop.regions, ITs, ITs, "GapJ")
]

def calculate_connection_statistics():
    exc_cells_PTs = set([cell[0] for cell in PTs])  # Extract cell IDs
    exc_cells_ITs_c = set([cell[0] for cell in ITs_c])
    inh_cells_ITs_b = set([cell[0] for cell in ITs_b])
    inh_cells_ITs_d = set([cell[0] for cell in ITs_d])
    exc_cells_CTs_cSTUT = set([cell[0] for cell in CTs_cSTUT])
    inh_cells_CTs_bSTUT = set([cell[0] for cell in CTs_bSTUT])
    inh_cells_CTs_dSTUT = set([cell[0] for cell in CTs_dSTUT])
    exc_cells_TCs_matrix = set([cell[0] for cell in TCs_matrix])
    inh_cells_TCs_intralaminar = set([cell[0] for cell in TCs_intralaminar])
    inh_cells_TCs_core = set([cell[0] for cell in TCs_core])
    exc_cells = exc_cells_PTs | exc_cells_ITs_c | exc_cells_CTs_cSTUT | exc_cells_TCs_matrix
    inh_cells = inh_cells_ITs_b | inh_cells_ITs_d | inh_cells_CTs_bSTUT | inh_cells_CTs_dSTUT | inh_cells_TCs_intralaminar | inh_cells_TCs_core


    # Initialize counters
    syn_ee_count = 0
    syn_ei_count = 0
    syn_ie_count = 0
    syn_ii_count = 0
    ele_ee_count = 0
    ele_ei_count = 0
    ele_ie_count = 0
    ele_ii_count = 0
    
    # Process continuous projections
    for regions, prePopList, postPopList, syn_type, *syn_ids in Continuous_Projections:
        for pre_cell in prePopList:
            for post_cell in postPopList:
                pre_cell_id = pre_cell[0]
                post_cell_id = post_cell[0]
                
                # Determine cell types
                pre_is_exc = pre_cell_id in exc_cells
                pre_is_inh = pre_cell_id in inh_cells
                post_is_exc = post_cell_id in exc_cells
                post_is_inh = post_cell_id in inh_cells

                # Estimate connections per cell pair (more realistic)
                # Using a base estimate of connections between each cell type pair
                estimated_connections = max(1, int(pre_cell[1] * post_cell[1] * 0.001))
                
                # Categorize connection type
                if pre_is_exc and post_is_exc:
                    syn_ee_count += estimated_connections
                elif pre_is_exc and post_is_inh:
                    syn_ei_count += estimated_connections
                elif pre_is_inh and post_is_exc:
                    syn_ie_count += estimated_connections
                elif pre_is_inh and post_is_inh:
                    syn_ii_count += estimated_connections
        
    # Process electrical projections (gap junctions)
    for regions, prePopList, postPopList, syn_type in Electrical_Projections:
        for pre_cell in prePopList:
            for post_cell in postPopList:
                pre_cell_id = pre_cell[0]
                post_cell_id = post_cell[0]
                
                # Determine cell types
                pre_is_exc = pre_cell_id in exc_cells
                pre_is_inh = pre_cell_id in inh_cells
                post_is_exc = post_cell_id in exc_cells
                post_is_inh = post_cell_id in inh_cells
                
                # Estimate connections per cell pair (more realistic)
                estimated_connections = max(1, int(pre_cell[1] * post_cell[1] * 0.001))
                
                # Categorize connection type
                if pre_is_exc and post_is_exc:
                    ele_ee_count += estimated_connections
                elif pre_is_exc and post_is_inh:
                    ele_ei_count += 0
                elif pre_is_inh and post_is_exc:
                    ele_ie_count += 0
                elif pre_is_inh and post_is_inh:
                    ele_ii_count += estimated_connections
    
    total_syn_connections = syn_ee_count + syn_ei_count + syn_ie_count + syn_ii_count
    total_ele_connections = ele_ee_count + ele_ii_count # + ele_ei_count + ele_ie_count

    # Handle case where no connections exist yet
    if total_syn_connections == 0:
        total_syn_connections = 1 
    if total_ele_connections == 0:
        total_ele_connections = 1 
    # Calculate percentages
    syn_ee_pct = (syn_ee_count / total_syn_connections * 100)
    syn_ei_pct = (syn_ei_count / total_syn_connections * 100)
    syn_ie_pct = (syn_ie_count / total_syn_connections * 100)
    syn_ii_pct = (syn_ii_count / total_syn_connections * 100)
    ele_ee_pct = (ele_ee_count / total_ele_connections * 100)
    ele_ei_pct = (ele_ei_count / total_ele_connections * 100)
    ele_ie_pct = (ele_ie_count / total_ele_connections * 100)
    ele_ii_pct = (ele_ii_count / total_ele_connections * 100)

    return {
        'E-E(syn)': (syn_ee_count, syn_ee_pct),
        'E-I(syn)': (syn_ei_count, syn_ei_pct),
        'I-E(syn)': (syn_ie_count, syn_ie_pct),
        'I-I(syn)': (syn_ii_count, syn_ii_pct),
        'total(syn)': total_syn_connections,
        'E-E(ele)': (ele_ee_count, ele_ee_pct),
        'E-I(ele)': (ele_ei_count, ele_ei_pct),
        'I-E(ele)': (ele_ie_count, ele_ie_pct),
        'I-I(ele)': (ele_ii_count, ele_ii_pct),
        'total(ele)': total_ele_connections
    }


PG =[
    ("inh_PG", "0.0s","20.0s",f"{0.02 + 0.02 * random.random()} nA"),
    ("exc_PG", "0.0s","20.0s",f"{0.03 + 0.01 * random.random()} nA")
]

InputLists=[
    (M2M1S1_loop.regions, TCs_intralaminar + TCs_matrix,'inh_PG',"GABA"),
    (M1S1_loop.regions,TCs_intralaminar + TCs_core,'inh_PG',"GABA"),
    (M2M1S1_loop.regions, CTs_cSTUT,'exc_PG',"AMPA_NMDA"),
    (M2M1S1_loop.regions, CTs_bSTUT + CTs_dSTUT,'inh_PG',"GABA"),
    (M2M1S1_loop.regions, PTs,'exc_PG',"AMPA_NMDA"),
    (M2M1S1_loop.regions, ITs_c,'exc_PG',"AMPA_NMDA"),
    (M2M1S1_loop.regions, ITs_b + ITs_d,'inh_PG',"GABA"),

]
# ======================
# Quantum-Inspired Core
# ======================
class NeuroMLQuantumOptimizer:
    def __init__(self, networks):
        self.networks = networks
        self.connection_cache = {}
    def _quantum_connect(self, pre_size, post_size, prob):
        """Sparse matrix generation using quantum-inspired sampling"""
        total = pre_size * post_size
        sparse_indices = np.random.choice(total, size=int(total*prob), replace=True)
        return np.unravel_index(sparse_indices, (pre_size, post_size))
# ======================
# Synaptic Projections
# ======================
def create_continuous_projections(network, regions, prePopList, postPopList, DoubleSynID, preSynID, postSynID):
    """Ultra-optimized continuous projections"""
    MAX_CONNECTIONS = 100
    CHUNK_SIZE = 50
    BATCH_SIZE = 20  

    # Pre-filter combinations with stricter rules
    filtered_combinations = []
    for pre_region, post_region in itertools.product(regions, repeat=2):
        if abs(regions.index(pre_region) - regions.index(post_region)) <= 1:
            n_samples = min(len(prePopList), 150)
            pre_samples = random.sample(prePopList, min(len(prePopList), 15))
            post_samples = random.sample(postPopList, min(len(postPopList), 15))
            for pre_cell in pre_samples:
                for post_cell in post_samples:
                    filtered_combinations.append((
                        pre_region.id, post_region.id,
                        pre_cell[0], pre_cell[1], pre_cell[2],
                        post_cell[0], post_cell[1], post_cell[2],
                    ))
        
    total_items = len(filtered_combinations)
    
    # Process in batches
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for chunk_start in range(0, total_items, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_items)
            chunk = filtered_combinations[chunk_start:chunk_end]
            
            future = executor.submit(
                _process_projection_chunk_optimized,
                chunk, network, DoubleSynID,
                preSynID, postSynID, MAX_CONNECTIONS,
                BATCH_SIZE
            )
            futures.append(future)
        
        # Process with aggressive cleanup
        with tqdm(total=len(futures), desc="Processing Syn-projections chunks") as pbar:
            for future in as_completed(futures):
                try:
                    chunk_projections = future.result()
                    if chunk_projections:
                        network.continuous_projections.extend(chunk_projections)
                    pbar.update(1)
                    
                    if pbar.n % 5 == 0:
                        gc.collect()
                    
                except Exception as e:
                    print(f"Chunk error: {e}")
                    continue
    
    return True

def _process_projection_chunk_optimized(chunk, network, DoubleSynID, preSynID, postSynID, MAX_CONNECTIONS, BATCH_SIZE):
    """Ultra-optimized chunk processing"""
    proj_batch = []
    
    # Pre-allocate GPU arrays
    conn_array = np.zeros((BATCH_SIZE, MAX_CONNECTIONS), dtype=np.int32)
    weight_array = np.zeros((BATCH_SIZE, MAX_CONNECTIONS), dtype=np.float32)
    
    for i in range(0, len(chunk), BATCH_SIZE):
        batch = chunk[i:min(i+BATCH_SIZE, len(chunk))]
        batch_projs = []
        for j, (pre_region_id, post_region_id, pre_cellID, pre_size, pre_axonalNum,
                post_cellID, post_size, post_axonalNum) in enumerate(batch):
            try:
                pre_is_exc = pre_cellID in EXC_CELLS
                post_is_exc = post_cellID in EXC_CELLS
                pre_is_inh = pre_cellID in INH_CELLS
                post_is_inh = post_cellID in INH_CELLS

                # Base connection probability with adjustments to achieve target statistics
                if pre_is_exc and post_is_exc:  # E-E connections - increase probability
                    conn_prob = max(0.95, np.sqrt(200/(pre_size * post_size)))  # Increased from 10 to 15
                elif pre_is_exc and post_is_inh:  # E-I connections - keep high probability
                    conn_prob = max(0.5, np.sqrt(200/(pre_size * post_size)))  # Increased from 10 to 20
                elif pre_is_inh and post_is_exc:  # I-E connections - decrease probability
                    conn_prob = max(0.5, np.sqrt(200/(pre_size * post_size)))  # Decreased from 10 to 2
                elif pre_is_inh and post_is_inh:  # I-I connections - moderate probability
                    conn_prob = max(0.05, np.sqrt(200/(pre_size * post_size)))  # Increased from 5 to 8
                else:
                    conn_prob = min(0.6, np.sqrt(200/(pre_size * post_size)))
                
                size = min(pre_size, post_size, MAX_CONNECTIONS)
                try:
                    n_connections = max(1, int(float(size) * float(conn_prob)))
                except ValueError as e:
                    print(f"Conversion error: {e}")
                    n_connections = 1
                # Generate connections in pre-allocated arrays
                pre_idx = np.random.choice(pre_size, n_connections, replace=True)
                post_idx = np.random.choice(post_size, n_connections, replace=True)
                w = np.random.uniform(2, 3.0, n_connections)

                # Generate segment IDs using cell-type-specific rules (same as input placement logic)
                pre_segments = []
                post_segments = []
                pre_fractions = []
                post_fractions = []
                # Create population IDs for lookup
                pre_popID = f"{pre_region_id}_{pre_cellID}"
                post_popID = f"{post_region_id}_{post_cellID}"
                
                for _ in range(n_connections):
                    # Pre-synaptic segment selection
                    if is_cortical_cell(pre_popID):
                        # Cortical cells: avoid segment ID 1, allow segment 0 with 10% probability
                        if random.random() < 0.1:
                            pre_segments.append(0)
                        else:
                            max_segment = get_actual_segment_count(pre_popID)
                            pre_segments.append(random.randint(2, max(max_segment, 2)))
                    else:
                        # Thalamic or other cells: allow all segment IDs including 1
                        max_segment = get_actual_segment_count(pre_popID)
                        pre_segments.append(random.randint(0, max(max_segment, 0)))
                    
                    # Post-synaptic segment selection  
                    if is_cortical_cell(post_popID):
                        # Cortical cells: avoid segment ID 1, allow segment 0 with 10% probability
                        if random.random() < 0.1:
                            post_segments.append(0)
                        else:
                            max_segment = get_actual_segment_count(post_popID)
                            post_segments.append(random.randint(2, max(max_segment, 2)))
                    else:
                        # Thalamic or other cells: allow all segment IDs including 1
                        max_segment = get_actual_segment_count(post_popID)
                        post_segments.append(random.randint(0, max(max_segment, 0)))
                    
                    # Random fractions along the segments
                    pre_fractions.append(random.random())
                    post_fractions.append(random.random())


                Syn_proj = ContinuousProjection(
                    id=f"{pre_region_id}_{pre_cellID}_{post_region_id}_{post_cellID}_{DoubleSynID}",
                    presynaptic_population=f"{pre_region_id}_{pre_cellID}",
                    postsynaptic_population=f"{post_region_id}_{post_cellID}"
                )
                # Batch connection creation
                Syn_conn = [
                    ContinuousConnectionInstanceW(
                        id=k,
                        pre_cell=f"../{pre_region_id}_{pre_cellID}/{pre}/{pre_cellID}",
                        post_cell=f"../{post_region_id}_{post_cellID}/{post}/{post_cellID}",
                        pre_component=preSynID,
                        post_component=postSynID,
                        weight=float(weight),
                        pre_segment=str(pre_segments[k]),
                        post_segment=str(post_segments[k]),
                        pre_fraction_along=float(pre_fractions[k]),
                        post_fraction_along=float(post_fractions[k])
                    )
                    for k, (pre, post, weight) in enumerate(zip(pre_idx, post_idx, w))
                ]
                Syn_proj.continuous_connection_instance_ws.extend(Syn_conn)
                batch_projs.append(Syn_proj)
                update_synaptic_contact_stats(pre_cellID, post_cellID, n_connections)
            
            except Exception as e:
                print(f"Connection error in batch {i}: {e}")
                continue
        # Add batch to results
        proj_batch.extend(batch_projs)
    return proj_batch


print("Processing Continuous Projections...")
for Syn_proj in Continuous_Projections:
    regions, prePopList, postPopList, DoubleSynID, preSynID, postSynID = Syn_proj
    create_continuous_projections(
        net, regions, prePopList, postPopList, 
        DoubleSynID, preSynID, postSynID
    )

# ======================
# Electrical Projections 
# ======================
def create_electrical_projections(network, regions, prePopList, postPopList, GapJ_id):
    """High-performance electrical projections with GPU acceleration"""
    optimizer = NeuroMLQuantumOptimizer([network])
    MAX_CONNECTIONS = 1000
    BATCH_SIZE = 1000
    PROJ_BATCH_SIZE = 5
    
    # for pre_region, post_region in itertools.product(regions, repeat=2):
    #     if abs(regions.index(pre_region) - regions.index(post_region)) <= 1:
    valid_combinations = [
        (pre_region, post_region, pre_cell, post_cell)
        for pre_region, post_region in itertools.product(regions, repeat=2)
        if abs(regions.index(pre_region) - regions.index(post_region)) <= 1
        for pre_cell, post_cell in itertools.product(prePopList, postPopList)
        if pre_cell[0] == post_cell[0]  # Only process matching cell types
    ]
    total_items = len(valid_combinations)
    
    elec_proj_batch = []
    unique_counter = 0

    with tqdm(total=total_items, desc="Creating Electrical Projections") as pbar:
        for pre_region, post_region, pre_cell, post_cell in valid_combinations:
            pre_cellID, pre_size, pre_axonalNum = pre_cell
            post_cellID, post_size, post_axonalNum = post_cell
            # Ensure pre_size and post_size are positive
            if pre_cellID == post_cellID and pre_size <= 0 or post_size <= 0:
                print(f"Invalid population sizes: pre_size={pre_size}, post_size={post_size}")
                pbar.update(1)
                continue

            pre_popID = f"{pre_region.id}_{pre_cellID}"
            post_popID = f"{post_region.id}_{post_cellID}"
            
            try:
                if any(thalamic_prefix in pre_cellID for thalamic_prefix in ['TCR','TCRil','TCRc','TCRm', 'nRT', 'nRTil', 'nRTc', 'nRTm']):  # Thalamic connections
                    conn_prob = min(0.1, np.sqrt(20/(pre_size * post_size)))  # Increased from default
                else:  # Cortical connections
                    pre_is_exc = pre_cellID in EXC_CELLS
                    post_is_exc = post_cellID in EXC_CELLS
                    pre_is_inh = pre_cellID in INH_CELLS
                    post_is_inh = post_cellID in INH_CELLS
                    
                    if pre_is_exc and post_is_exc:  # E-E connections
                        conn_prob = min(0.1, np.sqrt(80/(pre_size * post_size)))  # Increased
                    elif pre_is_inh and post_is_inh:  # I-I connections
                        conn_prob = min(0.1, np.sqrt(5/(pre_size * post_size)))  # Increased
                    else:  # I-E or E-I connections
                        conn_prob = 0
                conn_indices = optimizer._quantum_connect(pre_size, post_size, conn_prob)

                if len(conn_indices[0]) > MAX_CONNECTIONS:
                    idx = np.random.choice(len(conn_indices[0]), MAX_CONNECTIONS, replace=False)
                    conn_indices = (conn_indices[0][idx], conn_indices[1][idx])
                if len(conn_indices[0]) > 0:
                    elec_proj = ElectricalProjection(
                        id=f"{pre_popID}_{GapJ_id}_{unique_counter}",
                        presynaptic_population=pre_popID,
                        postsynaptic_population=post_popID
                    )
                    
                    # Process all connections at once
                    num_connections = len(conn_indices[0])
                    weights = np.random.uniform(0.2, 0.6, size=num_connections)
                    pre_indices = conn_indices[0]
                    post_indices = conn_indices[1]
                    
                    # Generate segment IDs using cell-type-specific rules (same as input placement logic)
                    pre_segments = []
                    post_segments = []
                    for _ in range(num_connections):
                        if is_cortical_cell(pre_popID):
                            # Cortical cells: avoid segment ID 1, allow segment 0 with 10% probability
                            if random.random() < 0.1:
                                pre_segments.append(0)
                            else:
                                max_segment = get_actual_segment_count(pre_cellID)
                                pre_segments.append(random.randint(2, max(max_segment, 2)))
                        else:
                            # Thalamic or other cells: allow all segment IDs including 1
                            max_segment = get_actual_segment_count(pre_cellID)
                            pre_segments.append(random.randint(0, max(max_segment, 0)))
                        
                        if is_cortical_cell(post_popID):
                            # Cortical cells: avoid segment ID 1, allow segment 0 with 10% probability
                            if random.random() < 0.1:
                                post_segments.append(0)
                            else:
                                max_segment = get_actual_segment_count(post_cellID)
                                post_segments.append(random.randint(2, max(max_segment, 2)))
                        else:
                            # Thalamic or other cells: allow all segment IDs including 1
                            max_segment = get_actual_segment_count(post_cellID)
                            post_segments.append(random.randint(0, max(max_segment, 0)))
                    
                    pre_fractions = np.random.rand(num_connections)
                    post_fractions = np.random.rand(num_connections)
                    # Create ElectricalConnectionInstanceW objects
                    elec_conn = [
                        ElectricalConnectionInstanceW(
                            id=k,
                            pre_cell=f"../{pre_popID}/{pre}/{pre_cellID}",
                            post_cell=f"../{post_popID}/{post}/{post_cellID}",
                            pre_segment=str(pre_segments[k]),
                            post_segment=str(post_segments[k]),
                            pre_fraction_along=float(pre_fractions[k]),
                            post_fraction_along=float(post_fractions[k]),
                            synapse=GapJ_id,
                            weight=float(weights[k])
                        )
                        for k, (pre, post) in enumerate(zip(pre_indices, post_indices))
                    ]
                    elec_proj.electrical_connection_instance_ws.extend(elec_conn)
                    elec_proj_batch.append(elec_proj)
                    update_electrical_conductance_stats(pre_cellID, post_cellID, num_connections)
                    unique_counter += 1
            except Exception as e:
                print(f"Connection error: {e}")
                continue
            pbar.update(1)
        # Add remaining projections
        if elec_proj_batch:
            network.electrical_projections.extend(elec_proj_batch)
    print("Final electrical stats check:")
    # check_electrical_stats()
    with electrical_stats_lock:
        persistent_electrical_stats['E-E(ele)'] = electrical_conductance_stats['E-E(ele)']
        persistent_electrical_stats['E-I(ele)'] = electrical_conductance_stats['E-I(ele)']
        persistent_electrical_stats['I-E(ele)'] = electrical_conductance_stats['I-E(ele)']
        persistent_electrical_stats['I-I(ele)'] = electrical_conductance_stats['I-I(ele)']
        persistent_electrical_stats['total(ele)'] = electrical_conductance_stats['total(ele)']
        electrical_stats_finalized = True
    print(f"Persistent stats updated with final values: {persistent_electrical_stats}")
    print(f"Electrical stats finalized: {electrical_stats_finalized}")

print("Processing Electrical Projections...")
for elec_proj in Electrical_Projections:
    try:
        regions, prePopList, postPopList, GapJ_id = elec_proj
        create_electrical_projections(net, regions, prePopList, postPopList, GapJ_id)
    except Exception as e:
        print(f"Error processing electrical projection: {e}")
        continue

print_separate_connection_stats()

# ======================
# Quantum-Inspired Pulse Generation
# ======================
def create_quantum_pulse_generators(nml_doc,rng_seed=42):
    rng = np.random.default_rng(seed=rng_seed)  # Quantum-inspired reproducible randomness
    # rng = random.Random()
    # MAX_GENERATORS = 1000
    # INH_RATIO = 0.4
    # EXC_RATIO = 0.6
    base_configs = {
        "inh_PG": ("0.0s", "20.0s", lambda: f"{0.02 + 0.02 * random.random():.3f} nA"),
        "exc_PG": ("0.0s", "20.0s", lambda: f"{0.03 + 0.01 * random.random():.3f} nA")
    }

    pulse_generators = {}
    inh_count = 0
    exc_count = 0
    # MAX_INH = int(MAX_GENERATORS * INH_RATIO)
    # MAX_EXC = int(MAX_GENERATORS * EXC_RATIO)
    for regions, PopLists, PG_id, dest in InputLists:
        for region in regions:
            RegionID = region.id
            for cellID, pop_size, axonalNum in PopLists:
                Pop_id = f"{RegionID}_{cellID}"
                if cellID in INH_CELLS:
                    inh_pg_id = f'inh_PG_{Pop_id}'
                    # if inh_pg_id not in pulse_generators and inh_count < MAX_INH:
                    # if inh_pg_id not in pulse_generators:
                    Delay, Duration, amp_func = base_configs["inh_PG"]
                    pg = PulseGenerator(
                        id=inh_pg_id, 
                        delay=Delay, 
                        duration=Duration, 
                        amplitude=f"{0.02 + 0.02 * random.random():.3f} nA"
                    )
                    nml_doc.pulse_generators.append(pg)
                    pulse_generators[inh_pg_id] = pg
                    inh_count += 1
                elif cellID in EXC_CELLS:
                    exc_pg_id = f'exc_PG_{Pop_id}'
                    # if exc_pg_id not in pulse_generators and exc_count < MAX_EXC:
                    # if exc_pg_id not in pulse_generators:
                    Delay, Duration, axonalNum = base_configs["exc_PG"]
                    pg = PulseGenerator(
                        id=exc_pg_id,
                        delay=Delay,
                        duration=Duration,
                        amplitude=f"{0.03 + 0.01 * random.random():.3f} nA"
                    )
                    nml_doc.pulse_generators.append(pg)
                    pulse_generators[exc_pg_id] = pg
                    exc_count += 1
                # if inh_count >= MAX_INH and exc_count >= MAX_EXC:
                #     return pulse_generators
    return pulse_generators
# ======================
# Sparse Input List Generation
# ======================
def create_quantum_input_lists(nml_doc, pulse_generators, rng_seed=42):
    """Create input lists with quantum-inspired sparse connectivity"""
    BATCH_SIZE = 800
    INH_RATIO = 0.4
    EXC_RATIO = 0.6
    
    # # Calculate total available populations
    # total_pops = sum(len(PopLists) * len(regions) for regions, PopLists, _, _ in InputLists)
    
    # # Set target counts
    # MAX_INH = int(total_pops * INH_RATIO)
    # MAX_EXC = int(total_pops * EXC_RATIO)
    # inh_count = 0
    # exc_count = 0
    
    for regions, PopLists, PG_id, dest in tqdm(InputLists, desc="Creating Input Lists"):
        for region in regions:
            RegionID = region.id
            for cellID, pop_size, axonalNum in PopLists:
                Pop_id = f"{RegionID}_{cellID}"
                
                for pg_type in  ('inh_PG', 'exc_PG'):
                    pg_id = f'{pg_type}_{Pop_id}'
                    if pg_id not in pulse_generators:
                        continue
                        
                    
                    ratio = INH_RATIO if PG_id.startswith('inh_PG') else EXC_RATIO
                    n_connections = max(1, int(pop_size * ratio * 0.3))  # Connect to ~30% of neurons
                    if n_connections <= 0:
                        continue    
                    input_list = InputList(
                        id=f"input_{pg_id}",
                        component=pulse_generators[pg_id].id,
                        populations=Pop_id
                    )
                    
                    connected_cells = np.random.choice(pop_size, size=n_connections, replace=False)
                    input_distribution = np.random.poisson(1.5, n_connections)  # Most neurons get 1-2 inputs
                    input_distribution = np.maximum(input_distribution, 1)  # Ensure at least 1 input per selected neuron
                    weights = np.random.uniform(0.1, 0.5, size=len(connected_cells))
                    
                    for i in range(0, len(connected_cells), BATCH_SIZE):
                        end_idx = min(i+BATCH_SIZE, len(connected_cells))
                        batch_cells = connected_cells[i:end_idx]
                        batch_weights = weights[i:end_idx]
                        
                        # Determine segment ID assignment strategy based on cell type
                        if is_cortical_cell(Pop_id):
                            segment_ids = []
                            for _ in range(len(batch_cells)):
                                if random.random() < 0.1:  # 10% chance to use segment 0 (soma)
                                    segment_ids.append(0)
                                else:
                                    max_segment = get_actual_segment_count(Pop_id)
                                    segment_ids.append(random.randint(2, max(max_segment, 2)))  # Avoid segment 1
                        else:
                            # Thalamic cells (TC) or other cell types: allow all segment IDs including 1
                            max_segment = get_actual_segment_count(Pop_id)
                            segment_ids = [random.randint(0, max(max_segment, 0)) for _ in range(len(batch_cells))]
                            
                        inputs = [
                            InputW(
                                id=i+j,
                                target=f"../{Pop_id}/{cell_idx}/{cellID}",
                                segment_id=int(segment_ids[j]),
                                fraction_along=f"{random.uniform(0, 1):.2f}",
                                destination=dest,
                                weight=f"{float(weight):.3f}",
                            )
                            for j, (cell_idx, weight) in enumerate(zip(batch_cells, batch_weights))
                        ]
                        input_list.input.extend(inputs)
                    net.input_lists.append(input_list)

                        
# ======================
# Execution
# ======================
rng_seed = 42
# Create pulse generators (called once)
pulse_generators = create_quantum_pulse_generators(nml_doc, rng_seed)
# Create input lists (called once)
create_quantum_input_lists(nml_doc, pulse_generators, rng_seed)

# %%
# ======================
# Validation and Summary
# ======================
pg_list = getattr(nml_doc, "pulse_generators", []) or []
print("Pulse generators totals (exc, inh):",
      sum(1 for pg in pg_list if getattr(pg, "id", "").startswith("exc")),
      sum(1 for pg in pg_list if getattr(pg, "id", "").startswith("inh")))
print("Sample PG ids:", [pg.id for pg in pg_list[:30]])

created_pg_ids = {pg.id for pg in pg_list}
expected_pg_ids = set()
for regions, PopLists, PG_id, dest in InputLists:
    for region in regions:
        for cellID, _, _ in PopLists:
            expected_pg_ids.add(f"{PG_id}_{region.id}_{cellID}")

missing = sorted(list(expected_pg_ids - created_pg_ids))[:50]
extra = sorted(list(created_pg_ids - expected_pg_ids))[:50]
print("Missing expected PG ids (first 50):", missing)
print("Extra/Unexpected PG ids (first 50):", extra)

print("Number of net.input_lists:", len(getattr(net, "input_lists", [])))
print("Sample input_lists (id, component, populations, #inputs):")
for il in (getattr(net, "input_lists", []) or [])[:30]:
    print(f"  {il.id}  component={getattr(il,'component',None)} populations={getattr(il,'populations',None)} inputs={len(getattr(il,'input',[]) or [])}")

# Quick cross-check: any input_list referencing a PG that doesn't exist?
bad_refs = []
for il in (getattr(net, "input_lists", []) or []):
    comp = getattr(il, "component", "")
    if comp and comp not in created_pg_ids:
        bad_refs.append((il.id, comp))
print("InputLists referencing missing PG ids (first 50):", bad_refs[:50])
# %%
def optimized_write_and_validate(nml_doc, filename):
    """Memory-efficient NeuroML document writing and validation"""
    # print(nml_doc.summary())
    
    # 1. Write document in chunks
    print("Writing NeuroML document...")
    try:
        with tqdm(total=1, desc="Writing file") as pbar:
            # Write without pretty printing to reduce file size
            writers.NeuroMLWriter.write(nml_doc, filename)
            pbar.update(1)
    except Exception as e:
        print(f"Error writing file: {e}")
        return None

    # 2. Stream validation in chunks
    print("Validating NeuroML file...")
    try:
        CHUNK_SIZE = 4096  # 4KB chunks
        results = []
        
        with tqdm(desc="Validation progress") as pbar:
            # Split validation into regions for better memory management
            regions = [region.id for region in nml_doc.networks[0].regions]
            total_regions = len(regions)
            
            for i, region in enumerate(regions):
                result = validate_neuroml2(
                    filename,
                    max_memory='16G',  # Lower memory per region
                    # silent=True
                )
                results.append(result)
                pbar.update((i + 1) / total_regions * 100)
                
            # Check validation results
            validation_failed = False
            for i, result in enumerate(results):
                if result is None or (hasattr(result, 'valid') and not result.valid):
                    print(f"\nValidation failed for region {regions[i]}")
                    if hasattr(result, 'errors'):
                        for error in result.errors:
                            print(f"- {error}")
                    validation_failed = True
                    break

            if not validation_failed:
                print("\nNeuroML validation completed successfully")
                return filename
            return None

    except Exception as e:
        print(f"Critical validation error: {e}")
        return None

# %%

import os
def save_nml_summary(nml_doc, outdir="analysis_out", basename=None):
    """Save nml_doc.summary() as plain TXT.
       Creates outdir if it does not exist."""
    summary = nml_doc.summary()  # returns string (or text printed)
    if basename is None:
        basename = getattr(nml_doc, "id", "network")
    os.makedirs(outdir, exist_ok=True)

    txt_path = os.path.join(outdir, f"{basename}_summary.txt")
    with open(txt_path, "w") as fh:
        fh.write(summary)
    return txt_path

saved = save_nml_summary(nml_doc, outdir="analysis_out", basename=getattr(nml_doc, "id", "network"))
print(f"Saved NeuroML summary to: {saved}")

# Memory cleanup before validation
import gc
gc.collect()

nml_file = f"{net.id}.net.nml"
result = optimized_write_and_validate(nml_doc, nml_file)
print(f"Validation result: {'Success' if result else 'Failed'}")

# %%
def calculate_input_statistics(nml_doc):
    """
    Calculate the mean ± std of excitatory and inhibitory inputs per neuron
    and the Pearson correlation coefficient between E/I input ratios
    """
    
    # Dictionary to store input counts per cell
    cell_inputs = defaultdict(lambda: {'exc': 0, 'inh': 0})
    
    # Process input lists to count inputs per cell
    for input_list in nml_doc.networks[0].input_lists:
        component = input_list.component
        population = input_list.populations
        
        # Determine if this is excitatory or inhibitory input
        is_exc = component.startswith('exc')
        is_inh = component.startswith('inh')
        
        if not (is_exc or is_inh):
            continue
            
        # Count inputs for each cell in this input list
        for input_item in input_list.input:
            # Parse the target to get cell index
            # Target format: "../PopulationName/cell_index/cell_id"
            target_parts = input_item.target.split('/')
            if len(target_parts) >= 3:
                cell_index = int(target_parts[-2])  # Cell index in population
                
                # Update counts
                if is_exc:
                    cell_inputs[(population, cell_index)]['exc'] += 1
                elif is_inh:
                    cell_inputs[(population, cell_index)]['inh'] += 1
    
    # Convert to arrays for statistical analysis
    exc_counts = np.array([inputs['exc'] for inputs in cell_inputs.values()])
    inh_counts = np.array([inputs['inh'] for inputs in cell_inputs.values()])
    
    # Calculate statistics
    exc_mean = np.mean(exc_counts)
    exc_std = np.std(exc_counts)
    inh_mean = np.mean(inh_counts)
    inh_std = np.std(inh_counts)
    
    # Calculate total inputs
    total_exc_inputs = np.sum(exc_counts)
    total_inh_inputs = np.sum(inh_counts)
    total_inputs = exc_counts + inh_counts
    
    # Calculate E/I ratios for correlation analysis
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-8
    ei_ratios = exc_counts / (inh_counts + epsilon)
    
    # Remove any infinite values
    valid_indices = np.isfinite(ei_ratios)
    valid_ei_ratios = ei_ratios[valid_indices]
    valid_exc_counts = exc_counts[valid_indices]
    valid_inh_counts = inh_counts[valid_indices]
    
    # Debug information about data filtering
    print(f"DEBUG: Before filtering - exc_counts length: {len(exc_counts)}")
    print(f"DEBUG: Before filtering - inh_counts length: {len(inh_counts)}")
    print(f"DEBUG: Before filtering - ei_ratios length: {len(ei_ratios)}")
    print(f"DEBUG: After filtering - valid_exc_counts length: {len(valid_exc_counts)}")
    print(f"DEBUG: After filtering - valid_inh_counts length: {len(valid_inh_counts)}")
    print(f"DEBUG: After filtering - valid_ei_ratios length: {len(valid_ei_ratios)}")
    
    # Show some statistics about the filtering
    if len(inh_counts) > 0:
        zero_inh_count = np.sum(inh_counts == 0)
        print(f"DEBUG: Number of neurons with zero inhibitory inputs: {zero_inh_count}")
        if zero_inh_count > 0:
            print("DEBUG: These neurons will have infinite E/I ratios and be filtered out")
    
    # Remove any infinite values
    valid_indices = np.isfinite(ei_ratios)
    valid_ei_ratios = ei_ratios[valid_indices]
    valid_exc_counts = exc_counts[valid_indices]
    valid_inh_counts = inh_counts[valid_indices]
    
    # Calculate Pearson correlation between exc and inh inputs
    r_value = float('nan')
    p_value = float('nan')
    
    if len(valid_exc_counts) > 1 and len(valid_inh_counts) > 1:
        # Convert to NumPy arrays explicitly
        # Check if we're using cupy arrays (have .get method) or regular numpy arrays
        if hasattr(valid_exc_counts, 'get'):
            valid_exc_np = valid_exc_counts.get()
        else:
            valid_exc_np = np.asarray(valid_exc_counts)
            
        if hasattr(valid_inh_counts, 'get'):
            valid_inh_np = valid_inh_counts.get()
        else:
            valid_inh_np = np.asarray(valid_inh_counts)
            
        # Check if arrays have variance (needed for correlation calculation)
        exc_var = np.var(valid_exc_np)
        inh_var = np.var(valid_inh_np)
        
        if exc_var > 0 and inh_var > 0:
            try:
                r_value, p_value = pearsonr(valid_exc_np, valid_inh_np)
            except Exception:
                # In case of any computational error, set to NaN
                r_value, p_value = float('nan'), float('nan')
        else:
            # If no variance in one or both arrays, correlation is undefined
            r_value, p_value = float('nan'), float('nan')
        
        # Calculate detailed Pearson correlation components for display
        exc_mean_valid = np.mean(valid_exc_np)
        inh_mean_valid = np.mean(valid_inh_np)
        
        # Calculate numerator and denominator for Pearson correlation
        diff_exc = valid_exc_np - exc_mean_valid
        diff_inh = valid_inh_np - inh_mean_valid
        
        numerator = np.sum(diff_exc * diff_inh)
        sum_sq_exc = np.sum(diff_exc**2)
        sum_sq_inh = np.sum(diff_inh**2)
        denominator = np.sqrt(sum_sq_exc * sum_sq_inh)
    else:
        numerator, denominator = float('nan'), float('nan')
        exc_mean_valid, inh_mean_valid = float('nan'), float('nan')
        sum_sq_exc, sum_sq_inh = float('nan'), float('nan')
        
        # Print results
        print("=" * 40)
        print("Input Statistics per Neuron:")
        print("=" * 40)
        print(f"Total neurons (N): {len(exc_counts)}")

    print(f"Excitatory inputs: {exc_mean:.2f} ± {exc_std:.2f}")
    print(f"  Calculation: Mean = Σ(excitatory inputs) / N = {total_exc_inputs:.0f} / {len(exc_counts)} = {exc_mean:.2f}")
    print(f"  Standard deviation = √[Σ(excitatory_input_count - mean_exc)² / (N-1)] = {exc_std:.2f}")
    print(f"    where excitatory_input_count = individual neuron's excitatory input count (range: {np.min(exc_counts):.0f} to {np.max(exc_counts):.0f})")
    print(f"    where mean_exc = average excitatory inputs per neuron ({exc_mean:.2f})")
    print(f"Inhibitory inputs: {inh_mean:.2f} ± {inh_std:.2f}")
    print(f"  Calculation: Mean = Σ(inhibitory inputs) / N = {total_inh_inputs:.0f} / {len(inh_counts)} = {inh_mean:.2f}")
    print(f"  Standard deviation = √[Σ(inhibitory_input_count - mean_inh)² / (N-1)] = {inh_std:.2f}")
    print(f"    where inhibitory_input_count = individual neuron's inhibitory input count (range: {np.min(inh_counts):.0f} to {np.max(inh_counts):.0f})")
    print(f"    where mean_inh = average inhibitory inputs per neuron ({inh_mean:.2f})")
    print(f"E/I ratio: {exc_mean/(inh_mean + epsilon):.2f}")
    
    # Check if r_value is valid before printing
    if isinstance(r_value, (int, float)) and not np.isnan(r_value):
        print(f"Pearson correlation (r): {r_value:.4f}")
    else:
        print("Pearson correlation (r): NaN")
    
    # Handle case where r_value might be a tuple (scipy version compatibility)
    if isinstance(r_value, tuple):
        r_val = r_value[0]
    else:
        r_val = r_value
        
    # Check if r_val is a valid number before checking for NaN
    if isinstance(r_val, (int, float)) and not np.isnan(r_val):
        print(f"  Calculation: r = Σ[(Xi - X̄)(Yi - Ȳ)] / √[Σ(Xi - X̄)² * Σ(Yi - Ȳ)²]")
        print(f"    Numerator = Σ[(exc_input - {exc_mean_valid:.2f})(inh_input - {inh_mean_valid:.2f})] = {numerator:.2f}")
        print(f"    Denominator = √[{sum_sq_exc:.2f} * {sum_sq_inh:.2f}] = {denominator:.2f}")
        print(f"    r = {numerator:.2f} / {denominator:.2f} = {r_val:.4f}")
        print(f"  Where Xi = excitatory inputs, Yi = inhibitory inputs")
        print(f"  X̄ = mean excitatory inputs ({exc_mean_valid:.2f}), Ȳ = mean inhibitory inputs ({inh_mean_valid:.2f})")
        print(f"  N = {len(valid_exc_counts)} neuron pairs with both excitatory and inhibitory inputs")
    else:
        # Check if we have enough data but zero variance
        if len(valid_exc_counts) > 1 and len(valid_inh_counts) > 1:
            # These variables should be defined from the earlier processing
            if 'exc_var' in locals() and 'inh_var' in locals():
                if exc_var == 0 and inh_var == 0:
                    print("  Correlation calculation: Skipped (both excitatory and inhibitory inputs have zero variance)")
                elif exc_var == 0:
                    print("  Correlation calculation: Skipped (excitatory inputs have zero variance)")
                elif inh_var == 0:
                    print("  Correlation calculation: Skipped (inhibitory inputs have zero variance)")
                else:
                    print("  Correlation calculation: Skipped (computational error)")
            else:
                print("  Correlation calculation: Skipped (computational error)")
        else:
            print("  Correlation calculation: Skipped (insufficient data)")
        
    if isinstance(p_value, (int, float)) and not np.isnan(p_value):
        print(f"P-value: {p_value:.4f}")
    else:
        print("P-value: NaN")
        
    print(f"Total inputs: {np.mean(total_inputs):.2f} ± {np.std(total_inputs):.2f}")
    print(f"  Calculation: Mean = Σ(total inputs) / N = {np.sum(total_inputs):.0f} / {len(total_inputs)} = {np.mean(total_inputs):.2f}")
    print("-" * 40)
    print("Total Input Counts:")
    print(f"  Total excitatory inputs: {total_exc_inputs:.0f}")
    print(f"  Total inhibitory inputs: {total_inh_inputs:.0f}")
    print(f"  Overall total inputs: {total_exc_inputs + total_inh_inputs:.0f}")
    
    # Show examples of individual neuron input counts
    print("-" * 40)
    print("Sample of individual neuron input counts:")
    print("Neuron ID\t\tExc\tInh\tTotal")
    sample_count = min(10, len(exc_counts))
    cell_inputs_list = list(cell_inputs.items())
    for i in range(sample_count):
        neuron_id, inputs = cell_inputs_list[i]
        # neuron_id is a tuple of (population_name, cell_index)
        print(f"{neuron_id[0]}_{neuron_id[1]}\t\t{inputs['exc']}\t{inputs['inh']}\t{inputs['exc'] + inputs['inh']}")
    
    return {
        'exc_mean': exc_mean,
        'exc_std': exc_std,
        'inh_mean': inh_mean,
        'inh_std': inh_std,
        'ei_ratio': exc_mean/(inh_mean + epsilon),
        'pearson_r': r_value,
        'p_value': p_value,
        'exc_counts': exc_counts,
        'inh_counts': inh_counts,
        'total_exc_inputs': total_exc_inputs,
        'total_inh_inputs': total_inh_inputs
    }

stats = calculate_input_statistics(nml_doc)

def export_connection_statistics(syn_ee, syn_ei, syn_ie, syn_ii, ele_ee, ele_ii, correlation, detailed_stats=None):
    """Export connection statistics to a JSON file for use by plotting scripts"""
    import json
    import os
    
    # Properly handle NaN values for JSON serialization
    correlation_value = None
    if correlation is not None and not (isinstance(correlation, float) and np.isnan(correlation)):
        correlation_value = float(correlation)
    
    stats = {
        'syn_ee_contacts': int(syn_ee),
        'syn_ei_contacts': int(syn_ei),
        'syn_ie_contacts': int(syn_ie),
        'syn_ii_contacts': int(syn_ii),
        'ele_ee_conductances': int(ele_ee),
        'ele_ii_conductances': int(ele_ii),
        'correlation': correlation_value,
        'network_name': net.id,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add detailed statistics if provided
    if detailed_stats:
        # Add connection type percentages
        stats['connection_percentages'] = {
            'excitatory': {
                'count': detailed_stats['percentages']['excitatory'][0],
                'percentage': detailed_stats['percentages']['excitatory'][1]
            },
            'inhibitory': {
                'count': detailed_stats['percentages']['inhibitory'][0],
                'percentage': detailed_stats['percentages']['inhibitory'][1]
            }
        }
        
        # Add incoming connections by target cell type
        stats['incoming_connections'] = detailed_stats['incoming']
        
        # Add outgoing connections by source cell type
        stats['outgoing_connections'] = detailed_stats['outgoing']
        
        # Add synaptic conductance metrics
        stats['synaptic_conductance'] = {
            'ampa': detailed_stats['synaptic_conductance']['ampa'],
            'nmda': detailed_stats['synaptic_conductance']['nmda'],
            'gabaa': detailed_stats['synaptic_conductance']['gabaa'],
            'gabab': detailed_stats['synaptic_conductance']['gabab'],
            'total': detailed_stats['synaptic_conductance']['total'],
            'average_per_connection': detailed_stats['synaptic_conductance']['average']
        }
        
        # Add electrical conductance
        stats['electrical_conductance'] = {
            'total': detailed_stats['synaptic_conductance']['total'] / detailed_stats['synaptic_conductance']['average'] if detailed_stats['synaptic_conductance']['average'] > 0 else 0,
            'peak': 3.0 * ele_ee + 3.0 * ele_ii  # Assuming 3.0 nS per electrical connection
        }
        
        # Add total peak conductance
        stats['total_peak_conductance'] = (
            detailed_stats['synaptic_conductance']['total'] + 
            (3.0 * ele_ee + 3.0 * ele_ii)
        )
    
    # Save to the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define base_name from the network name in stats
    base_name = stats['network_name']  # Use the network name from stats
    # Create analysis_out directory if it doesn't exist
    analysis_dir = os.path.join(script_dir, 'analysis_out')
    os.makedirs(analysis_dir, exist_ok=True)
    # Save connection stats to analysis_out directory with base_name prefix
    stats_file = os.path.join(analysis_dir, f'{base_name}_connection_stats.json')
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nConnection statistics exported to {stats_file}")
    if correlation_value is not None:
        print(f"Correlation coefficient: {correlation_value:.6f}")
    else:
        print("Correlation coefficient: None (insufficient data or no variance)")

def calculate_detailed_connection_statistics():
    # Get final electrical stats
    elec_stats = get_final_electrical_stats()
    
    # Synaptic contacts
    syn_ee = synaptic_contact_stats['E-E(syn)']
    syn_ei = synaptic_contact_stats['E-I(syn)']
    syn_ie = synaptic_contact_stats['I-E(syn)']
    syn_ii = synaptic_contact_stats['I-I(syn)']
    total_syn = synaptic_contact_stats['total(syn)']
    
    # Electrical conductances
    ele_ee = elec_stats['E-E(ele)']
    ele_ei = elec_stats['E-I(ele)']
    ele_ie = elec_stats['I-E(ele)']
    ele_ii = elec_stats['I-I(ele)']
    total_ele = elec_stats['total(ele)']
    
    # Total connections
    total_connections = total_syn + total_ele
    
    print("\n" + "="*60)
    print("DETAILED CONNECTION STATISTICS")
    print("="*60)
    
    # 1. Percentage of excitatory and inhibitory connections
    print("\n1. CONNECTION TYPE PERCENTAGES:")
    print("-" * 40)
    if total_connections > 0:
        # Fix: Make consistent with extract_network_dataset which counts outgoing connections
        exc_connections = syn_ee + syn_ei + ele_ee + ele_ei  # Excitatory outgoing
        inh_connections = syn_ie + syn_ii + ele_ii + ele_ie  # Inhibitory outgoing
        print(f"Excitatory connections: {exc_connections} ({exc_connections/total_connections*100:.2f}%)")
        print(f"Inhibitory connections: {inh_connections} ({inh_connections/total_connections*100:.2f}%)")
    else:
        print("No connections found")
    
    # 2. Incoming E/I connections for excitatory and inhibitory cells
    print("\n2. INCOMING CONNECTIONS BY TARGET CELL TYPE:")
    print("-" * 40)
    # Incoming to excitatory cells - Fix to be consistent with extract_network_dataset
    incoming_to_exc = (syn_ee + syn_ie + ele_ee + ele_ie) #  
    incoming_exc_to_exc = (syn_ee + ele_ee)
    incoming_inh_to_exc =   (syn_ie + ele_ie)
    
    print(f"To Excitatory cells:")
    print(f"  Total: {incoming_to_exc}")
    if incoming_to_exc > 0:
        print(f"  From Excitatory: {incoming_exc_to_exc} ({incoming_exc_to_exc/incoming_to_exc*100:.2f}%)")
        print(f"  From Inhibitory: {incoming_inh_to_exc} ({incoming_inh_to_exc/incoming_to_exc*100:.2f}%)")
    
    # Incoming to inhibitory cells - Fix to be consistent with extract_network_dataset
    incoming_to_inh = (syn_ei + syn_ii + ele_ii) #  + ele_ei
    incoming_exc_to_inh = syn_ei # ( + ele_ei)
    incoming_inh_to_inh = (syn_ii + ele_ii)
    
    print(f"To Inhibitory cells:")
    print(f"  Total: {incoming_to_inh}")
    if incoming_to_inh > 0:
        print(f"  From Excitatory: {incoming_exc_to_inh} ({incoming_exc_to_inh/incoming_to_inh*100:.2f}%)")
        print(f"  From Inhibitory: {incoming_inh_to_inh} ({incoming_inh_to_inh/incoming_to_inh*100:.2f}%)")
    
    # 3. E/I connections ratio distribution
    print("\n3. CONNECTION RATIOS BY SOURCE CELL TYPE:")
    print("-" * 40)
    # From excitatory cells
    from_exc_total = (syn_ee + syn_ei + ele_ee) #  + ele_ei
    exc_to_exc = (syn_ee + ele_ee)
    exc_to_inh = syn_ei # (syn_ei + ele_ei)
    
    print(f"From Excitatory cells:")
    print(f"  Total: {from_exc_total}")
    if from_exc_total > 0:
        print(f"  To Excitatory: {exc_to_exc} ({exc_to_exc/from_exc_total*100:.2f}%)")
        print(f"  To Inhibitory: {exc_to_inh} ({exc_to_inh/from_exc_total*100:.2f}%)")
        print(f"  E/I Ratio: {exc_to_exc/exc_to_inh if exc_to_inh > 0 else float('inf'):.2f}")
    
    # From inhibitory cells
    from_inh_total = (syn_ie + syn_ii + ele_ie + ele_ii)
    inh_to_exc = (syn_ie + ele_ie)
    inh_to_inh = (syn_ii + ele_ii)
    
    print(f"From Inhibitory cells:")
    print(f"  Total: {from_inh_total}")
    if from_inh_total > 0:
        print(f"  To Excitatory: {inh_to_exc} ({inh_to_exc/from_inh_total*100:.2f}%)")
        print(f"  To Inhibitory: {inh_to_inh} ({inh_to_inh/from_inh_total*100:.2f}%)")
        print(f"  E/I Ratio: {inh_to_exc/inh_to_inh if inh_to_inh > 0 else float('inf'):.2f}")
    
    # 4. Total synaptic contacts and peak conductance
    print("\n4. SYNAPTIC AND ELECTRICAL CONDUCTANCE METRICS:")
    print("-" * 40)
    print(f"Total synaptic contacts: {total_syn}")
    print(f"Total electrical conductances: {total_ele}")
    
    # Calculate actual synaptic contact based on real connection data and synapse parameters
    # Using the actual synaptic contact values:
    # AMPA: 0.367879 nS
    # NMDA: 0.8 nS 
    # GABAa: 0.057 nS
    # GABAb: 0.0169139 nS
    # GapJ: 3 nS
    
    # For AMPA_NMDA double synapses, we assume 60% AMPA and 40% NMDA contribution
    # For GABA double synapses, we assume 85% GABAa and 15% GABAb contribution
    
    # Calculate total conductance contribution from each synapse type
    # AMPA contribution from excitatory connections (EE and EI)
    amp_a_contrib = 0.367879 * (syn_ee + syn_ei) * 0.60
    # NMDA contribution from excitatory connections (EE and EI)  
    nmda_contrib = 0.8 * (syn_ee + syn_ei) * 0.40
    # GABAa contribution from inhibitory connections (IE and II)
    gabaa_contrib = 0.057 * (syn_ie + syn_ii) * 0.85
    # GABAb contribution from inhibitory connections (IE and II)
    gabab_contrib = 0.0169139 * (syn_ie + syn_ii) * 0.15
    
    # Total synaptic contact is the sum of all contributions
    total_synaptic_conductance = amp_a_contrib + nmda_contrib + gabaa_contrib + gabab_contrib
    
    # Calculate average synaptic contact per connection
    total_synaptic_contacts = syn_ee + syn_ei + syn_ie + syn_ii
    if total_synaptic_contacts > 0:
        avg_synaptic_conductance = total_synaptic_conductance / total_synaptic_contacts
    else:
        avg_synaptic_conductance = 0.0
    
    # Electrical conductance based on actual GapJunction parameter:
    # GapJ: 3 nS
    avg_electrical_conductance = 3.0  # nS - based on GapJ conductance parameter
    
    total_peak_synaptic_conductance = total_synaptic_conductance
    total_peak_electrical_conductance = total_ele * avg_electrical_conductance
    
    print(f"  AMPA contribution: {amp_a_contrib:.2f} nS")
    print(f"  NMDA contribution: {nmda_contrib:.2f} nS") 
    print(f"  GABAa contribution: {gabaa_contrib:.2f} nS")
    print(f"  GABAb contribution: {gabab_contrib:.2f} nS")
    print(f"  Total synaptic contact: {total_synaptic_conductance:.2f} nS")
    print(f"  Average synaptic contact per connection: {avg_synaptic_conductance:.4f} nS")
    print(f"  Estimated peak synaptic contact: {total_peak_synaptic_conductance:.2f} nS")
    print(f"  Estimated peak electrical conductance: {total_peak_electrical_conductance:.2f} nS")
    print(f"  Total peak conductance: {total_peak_synaptic_conductance + total_peak_electrical_conductance:.2f} nS")
    
    return {
        'percentages': {
            'excitatory': (exc_connections, exc_connections/total_connections*100 if total_connections > 0 else 0),
            'inhibitory': (inh_connections, inh_connections/total_connections*100 if total_connections > 0 else 0)
        },
        'incoming': {
            'to_excitatory': {
                'total': incoming_to_exc,
                'from_excitatory': (incoming_exc_to_exc, incoming_exc_to_exc/incoming_to_exc*100 if incoming_to_exc > 0 else 0),
                'from_inhibitory': (incoming_inh_to_exc, incoming_inh_to_exc/incoming_to_exc*100 if incoming_to_exc > 0 else 0)
            },
            'to_inhibitory': {
                'total': incoming_to_inh,
                'from_excitatory': (incoming_exc_to_inh, incoming_exc_to_inh/incoming_to_inh*100 if incoming_to_inh > 0 else 0),
                'from_inhibitory': (incoming_inh_to_inh, incoming_inh_to_inh/incoming_to_inh*100 if incoming_to_inh > 0 else 0)
            }
        },
        'outgoing': {
            'from_excitatory': {
                'total': from_exc_total,
                'to_excitatory': (exc_to_exc, exc_to_exc/from_exc_total*100 if from_exc_total > 0 else 0),
                'to_inhibitory': (exc_to_inh, exc_to_inh/from_exc_total*100 if from_exc_total > 0 else 0)
            },
            'from_inhibitory': {
                'total': from_inh_total,
                'to_excitatory': (inh_to_exc, inh_to_exc/from_inh_total*100 if from_inh_total > 0 else 0),
                'to_inhibitory': (inh_to_inh, inh_to_inh/from_inh_total*100 if from_inh_total > 0 else 0)
            }
        },
        'synaptic_conductance': {
            'ampa': amp_a_contrib,
            'nmda': nmda_contrib,
            'gabaa': gabaa_contrib,
            'gabab': gabab_contrib,
            'total': total_synaptic_conductance,
            'average': avg_synaptic_conductance
        }
    }

def analyze_connection_distribution():
    """Analyze the distribution of connections among populations"""
    
    print("\n" + "="*60)
    print("POPULATION CONNECTION DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Analyze cell type distribution
    exc_cell_count = len(EXC_CELLS)
    inh_cell_count = len(INH_CELLS)
    total_cells = exc_cell_count + inh_cell_count
    
    print(f"Cell Population:")
    print(f"  Excitatory cell types: {exc_cell_count}")
    print(f"  Inhibitory cell types: {inh_cell_count}")
    print(f"  Total cell types: {total_cells}")
    
    if total_cells > 0:
        print(f"  Excitatory ratio: {exc_cell_count/total_cells*100:.2f}%")
        print(f"  Inhibitory ratio: {inh_cell_count/total_cells*100:.2f}%")
    
    # Connection density analysis
    syn_total = synaptic_contact_stats['total(syn)']
    elec_stats = get_final_electrical_stats()
    ele_total = elec_stats['total(ele)']
    
    if total_cells > 0:
        syn_density = syn_total / total_cells if total_cells > 0 else 0
        elec_density = ele_total / total_cells if total_cells > 0 else 0
        
        print(f"\nConnection Density (per cell type):")
        print(f"  Synaptic contacts: {syn_density:.2f}")
        print(f"  Electrical conductances: {elec_density:.2f}")
        
    # E/I balance analysis
    syn_ee = synaptic_contact_stats['E-E(syn)']
    syn_ei = synaptic_contact_stats['E-I(syn)']
    syn_ie = synaptic_contact_stats['I-E(syn)']
    syn_ii = synaptic_contact_stats['I-I(syn)']
    
    ele_ee = elec_stats['E-E(ele)']
    ele_ei = elec_stats['E-I(ele)']
    ele_ie = elec_stats['I-E(ele)']
    ele_ii = elec_stats['I-I(ele)']
    
    total_ee = syn_ee + ele_ee
    total_ei = syn_ei + ele_ei
    total_ie = syn_ie + ele_ie
    total_ii = syn_ii + ele_ii
    
    print(f"\nE/I Balance Analysis:")
    if total_ie > 0:
        ei_ratio_exc = total_ee / total_ie if total_ie > 0 else float('inf')
        print(f"  E/I ratio for excitatory targets: {ei_ratio_exc:.2f}")
    if total_ii > 0:
        ei_ratio_inh = total_ei / total_ii if total_ii > 0 else float('inf')
        print(f"  E/I ratio for inhibitory targets: {ei_ratio_inh:.2f}")

# Add this after your existing connection processing code
print("\n" + "="*60)
print("FINAL CONNECTION ANALYSIS")
print("="*60)

# Calculate and display detailed statistics
detailed_stats = calculate_detailed_connection_statistics()
# analyze_connection_distribution()  # This is already called above

stats = calculate_input_statistics(nml_doc)

# Export statistics to a file for use by plotting scripts
export_connection_statistics(
    synaptic_contact_stats['E-E(syn)'], 
    synaptic_contact_stats['E-I(syn)'], 
    synaptic_contact_stats['I-E(syn)'], 
    synaptic_contact_stats['I-I(syn)'], 
    get_final_electrical_stats()['E-E(ele)'], 
    get_final_electrical_stats()['I-I(ele)'], 
    stats['pearson_r'], 
    detailed_stats
)


# %%
from collections import defaultdict
from pathlib import Path
import json
import pandas as pd

def _cellid_set_from_tuples(tuple_list):
    return {t[0] for t in tuple_list}

def extract_network_dataset(net, nml_doc):
    """
    Extract network statistics.

    Definitions:
      - cont_* : continuous synaptic contacts (counts of ContinuousConnectionInstanceWs)
         - EE, EI, IE, II : global counts where pre_type -> post_type (Exc/Inh)
         - cont_syn_exc_total : synaptic contacts whose presynaptic population is excitatory (EE + EI)
         - cont_syn_inh_total : synaptic contacts whose presynaptic population is inhibitory (IE + II)
         - per-population fields:
             * cont_out / cont_in : total synaptic contacts outgoing/incoming for that population
             * cont_exc_out / cont_inh_out : outgoing contacts where the presynaptic population is excitatory/inhibitory
             * cont_exc_in / cont_inh_in : incoming contacts originating from excitatory/inhibitory presynaptic populations
         - densities: cont_*_density = cont_count / pop_size

      - elec_* : electrical connections (counts of ElectricalConnectionInstanceWs)
         - elec_EE, elec_II : global counts (only same-type for electrical)
         - elec_exc_total, elec_inh_total : totals for excitatory/inhibitory electrical
         - per-population fields:
             * elec_out / elec_in : total electrical outgoing/incoming for that population
             * elec_exc_out / elec_inh_out, elec_exc_in / elec_inh_in similarly if desired

      - overall_* : combine synaptic presynaptic-type totals and electrical totals:
         - overall_exc_connections = cont_syn_exc_total + elec_exc_total
         - overall_inh_connections = cont_syn_inh_total + elec_inh_total

    Returns:
      dict with keys: pops_df, pop_stats_df, summary
    """
    exc_set = _cellid_set_from_tuples(PTs) | _cellid_set_from_tuples(ITs_c) | _cellid_set_from_tuples(CTs_cSTUT) | _cellid_set_from_tuples(TCs_matrix)
    inh_set = _cellid_set_from_tuples(ITs_b) | _cellid_set_from_tuples(ITs_d) | _cellid_set_from_tuples(CTs_bSTUT) | _cellid_set_from_tuples(CTs_dSTUT) | _cellid_set_from_tuples(TCs_intralaminar)

    # build populations table
    pops = []
    for pop in getattr(net, "populations", []):
        comp = getattr(pop, "component", None)
        try:
            size = int(getattr(pop, "size", 0) or 0)
        except (ValueError, TypeError):
            size = 0
        if comp in exc_set:
            typ = "exc"
        elif comp in inh_set:
            typ = "inh"
        else:
            typ = "unknown"
        region, _, cell = pop.id.partition('_')
        # # typ = "exc" if comp in exc_set else ("inh" if comp in inh_set else "unknown")
        typ = "exc" if comp in exc_set else "inh"
        pops.append({
            "pop_id": pop.id,
            "component": comp,
            "region": region,
            "size": size,
            "type": typ
        })
    pops_df = pd.DataFrame(pops).set_index("pop_id")
    def _get_type(pop_id):
        if pop_id in pops_df.index:
            return pops_df.at[pop_id, "type"]
        logging.warning(f"Population not found in pops_df: {pop_id}")
        return "unknown"

    # prepare per-pop containers
    cont_stats = defaultdict(lambda: {"out":0,"in":0,"exc_out":0,"inh_out":0,"exc_in":0,"inh_in":0})
    elec_stats = defaultdict(lambda: {"out":0,"in":0,"exc_out":0,"inh_out":0,"exc_in":0,"inh_in":0})

    # global synaptic totals by pre/post type
    EE = EI = IE = II = 0

    # continuous synaptic projections
    for proj in getattr(net, "continuous_projections", []) or []:
        pre_pop = getattr(proj, "presynaptic_population", None)
        post_pop = getattr(proj, "postsynaptic_population", None)
        instances = getattr(proj, "continuous_connection_instance_ws", []) or []
        nconn = len(instances)
        if nconn == 0:
            continue

        # pre_typ = pops_df.loc[pre_pop]["type"] 
        # post_typ = pops_df.loc[post_pop]["type"] 
        pre_typ = _get_type(pre_pop)
        post_typ = _get_type(post_pop)
        # accumulate per-pop totals regardless of type (but only if populations exist)
        if pre_pop in pops_df.index:
            cont_stats[pre_pop]["out"] += nconn
            if pre_typ == "exc":
                cont_stats[pre_pop]["exc_out"] += nconn
            elif pre_typ == "inh":
                cont_stats[pre_pop]["inh_out"] += nconn
        if post_pop in pops_df.index:
            cont_stats[post_pop]["in"] += nconn
            if pre_typ == "exc":
                cont_stats[post_pop]["exc_in"] += nconn
            elif pre_typ == "inh":
                cont_stats[post_pop]["inh_in"] += nconn
        # # accumulate per-pop totals
        # cont_stats[pre_pop]["out"] += nconn
        # cont_stats[post_pop]["in"] += nconn
        # global category counts only for known exc/inh
        if pre_typ in ("exc","inh") and post_typ in ("exc","inh"):
            if pre_typ == "exc" and post_typ == "exc":
                EE += nconn
            elif pre_typ == "exc" and post_typ == "inh":
                EI += nconn
            elif pre_typ == "inh" and post_typ == "exc":
                IE += nconn
            elif pre_typ == "inh" and post_typ == "inh":
                II += nconn
        else:
            logging.debug(f"Skipped global categorization for projection {getattr(proj,'id',None)} (pre={pre_pop}:{pre_typ}, post={post_pop}:{post_typ})")

        # # outgoing broken down by presynaptic population type
        # if pre_typ == "exc":
        #     cont_stats[pre_pop]["exc_out"] += nconn
        # elif pre_typ == "inh":
        #     cont_stats[pre_pop]["inh_out"] += nconn

        # # incoming broken down by presynaptic population type (i.e., whether source was exc/inh)
        # if pre_typ == "exc":
        #     cont_stats[post_pop]["exc_in"] += nconn
        # elif pre_typ == "inh":
        #     cont_stats[post_pop]["inh_in"] += nconn

        # # global category counts (pre -> post)
        # if pre_typ == "exc" and post_typ == "exc":
        #     EE += nconn
        # elif pre_typ == "exc" and post_typ == "inh":
        #     EI += nconn
        # elif pre_typ == "inh" and post_typ == "exc":
        #     IE += nconn
        # elif pre_typ == "inh" and post_typ == "inh":
        #     II += nconn

    # electrical projections (gap junctions)
    elec_EE = elec_II = 0
    for proj in getattr(net, "electrical_projections", []) or []:
        pre_pop = getattr(proj, "presynaptic_population", None)
        post_pop = getattr(proj, "postsynaptic_population", None)
        instances = getattr(proj, "electrical_connection_instance_ws", []) or []
        nconn = len(instances)
        if nconn == 0:
            continue
        
        pre_typ = _get_type(pre_pop)
        post_typ = _get_type(post_pop)

        if pre_pop in pops_df.index:
            elec_stats[pre_pop]["out"] += nconn
        if post_pop in pops_df.index:
            elec_stats[post_pop]["in"] += nconn

        if pre_typ == "exc" and post_typ == "exc":
            elec_EE += nconn
            if pre_pop in pops_df.index: elec_stats[pre_pop]["exc_out"] += nconn
            if post_pop in pops_df.index: elec_stats[post_pop]["exc_in"] += nconn
        elif pre_typ == "inh" and post_typ == "inh":
            elec_II += nconn
            if pre_pop in pops_df.index: elec_stats[pre_pop]["inh_out"] += nconn
            if post_pop in pops_df.index: elec_stats[post_pop]["inh_in"] += nconn
        else:
            logging.debug(f"Electrical proj with unknown types: {getattr(proj,'id',None)}")
        # pre_typ = pops_df.loc[pre_pop]["type"] 
        # post_typ = pops_df.loc[post_pop]["type"] 
        # elec_stats[pre_pop]["out"] += nconn
        # elec_stats[post_pop]["in"] += nconn

        # # mark as exc/inh electrical based on the populations involved (should be same-type)
        # if pre_typ == "exc" and post_typ == "exc":
        #     elec_EE += nconn
        #     elec_stats[pre_pop]["exc_out"] += nconn
        #     elec_stats[post_pop]["exc_in"] += nconn
        # elif pre_typ == "inh" and post_typ == "inh":
        #     elec_II += nconn
        #     elec_stats[pre_pop]["inh_out"] += nconn
        #     elec_stats[post_pop]["inh_in"] += nconn

    # inputs and pulse generators
    input_lists = getattr(net, "input_lists", []) or []
    total_inputs = 0
    exc_inputs = 0
    inh_inputs = 0
    input_per_pop_exc = defaultdict(int)
    input_per_pop_inh = defaultdict(int)
    for il in input_lists:
        comp = getattr(il, "component", "")
        is_inh = comp.startswith("inh")
        inputs = getattr(il, "input", []) or []
        n = len(inputs)
        total_inputs += n
        # target_pop = getattr(il, "populations", None) or getattr(il, "populations", "")
        # normalize populations field to string key
        target_pop = getattr(il, "populations", None)
        if isinstance(target_pop, (list, tuple)):
            target_pop = target_pop[0] if target_pop else ""
        target_pop = str(target_pop)

        if is_inh:
            inh_inputs += n
            input_per_pop_inh[target_pop] += n
        else:
            exc_inputs += n
            input_per_pop_exc[target_pop] += n
        target_pop = getattr(il, "populations", None) or getattr(il, "populations", "")

    # pulse generators in document
    pg_list = getattr(nml_doc, "pulse_generators", []) or []
    exc_pgs = sum(1 for pg in pg_list if getattr(pg, "id", "").startswith("exc"))
    inh_pgs = sum(1 for pg in pg_list if getattr(pg, "id", "").startswith("inh"))

    # build population-level rows (including detailed cont/elec breakdowns)
    rows = []
    for pid, r in pops_df.iterrows():
        size = max(1, int(r["size"]))
        cont = cont_stats[pid]
        elec = elec_stats[pid]
        exc_inputs_pop = input_per_pop_exc.get(pid, 0)
        inh_inputs_pop = input_per_pop_inh.get(pid, 0)
        total_out = cont["out"] + elec["out"]
        total_in = cont["in"] + elec["in"]

        # cont_out = cont_stats[pid]["out"]
        # cont_in = cont_stats[pid]["in"]
        # cont_exc_out = cont_stats[pid]["exc_out"]
        # cont_inh_out = cont_stats[pid]["inh_out"]
        # cont_exc_in = cont_stats[pid]["exc_in"]
        # cont_inh_in = cont_stats[pid]["inh_in"]

        # elec_out = elec_stats[pid]["out"]
        # elec_in = elec_stats[pid]["in"]
        # elec_exc_out = elec_stats[pid]["exc_out"]
        # elec_inh_out = elec_stats[pid]["inh_out"]
        # elec_exc_in = elec_stats[pid]["exc_in"]
        # elec_inh_in = elec_stats[pid]["inh_in"]

        # total_out = cont_out + elec_out
        # total_in = cont_in + elec_in
        # exc_inputs_pop = input_per_pop_exc.get(pid, 0)
        # inh_inputs_pop = input_per_pop_inh.get(pid, 0)
        # inputs = exc_inputs_pop + inh_inputs_pop
        # size = max(1, int(r["size"]))

        # rows.append({
        #     "pop_id": pid,
        #     "component": r["component"],
        #     "region": r["region"],
        #     "type": r["type"],
        #     "size": size,
        #     # synaptic breakdown
        #     "cont_out": cont_out,
        #     "cont_in": cont_in,
        #     "cont_exc_out": cont_exc_out,
        #     "cont_inh_out": cont_inh_out,
        #     "cont_exc_in": cont_exc_in,
        #     "cont_inh_in": cont_inh_in,
        #     # electrical breakdown
        #     "elec_out": elec_out,
        #     "elec_in": elec_in,
        #     "elec_exc_out": elec_exc_out,
        #     "elec_inh_out": elec_inh_out,
        #     "elec_exc_in": elec_exc_in,
        #     "elec_inh_in": elec_inh_in,
        #     # totals / inputs
        #     "total_out": total_out,
        #     "total_in": total_in,
        #     "inputs": inputs,
        #     "exc_inputs": exc_inputs_pop,
        #     "inh_inputs": inh_inputs_pop,
        #     # densities (per neuron)
        #     "cont_out_density": cont_out/size,
        #     "cont_in_density": cont_in/size,
        #     "elec_out_density": elec_out/size,
        #     "elec_in_density": elec_in/size,
        #     "input_density": inputs/size,
        #     "exc_input_density": (exc_inputs_pop/size),
        #     "inh_input_density": (inh_inputs_pop/size)
        # })
        rows.append({
            "pop_id": pid,
            "component": r["component"],
            "region": r["region"],
            "type": r["type"],
            "size": size,
            "cont_out": cont["out"],
            "cont_in": cont["in"],
            "cont_exc_out": cont["exc_out"],
            "cont_inh_out": cont["inh_out"],
            "cont_exc_in": cont["exc_in"],
            "cont_inh_in": cont["inh_in"],
            "elec_out": elec["out"],
            "elec_in": elec["in"],
            "elec_exc_out": elec["exc_out"],
            "elec_inh_out": elec["inh_out"],
            "elec_exc_in": elec["exc_in"],
            "elec_inh_in": elec["inh_in"],
            "total_out": total_out,
            "total_in": total_in,
            "inputs": exc_inputs_pop + inh_inputs_pop,
            "exc_inputs": exc_inputs_pop,
            "inh_inputs": inh_inputs_pop,
            "cont_out_density": cont["out"]/size,
            "cont_in_density": cont["in"]/size,
            "elec_out_density": elec["out"]/size,
            "elec_in_density": elec["in"]/size,
            "input_density": (exc_inputs_pop + inh_inputs_pop)/size,
            "exc_input_density": exc_inputs_pop/size,
            "inh_input_density": inh_inputs_pop/size
        })

    pop_stats_df = pd.DataFrame(rows).set_index("pop_id")

    # aggregate summaries and percentages
    total_syn_contacts = EE + EI + IE + II
    total_elec = elec_EE + elec_II
    cont_syn_exc_total = EE + EI     # presynaptic excitatory synaptic contacts
    cont_syn_inh_total = IE + II     # presynaptic inhibitory synaptic contacts
    # elec_exc_total = elec_EE
    # elec_inh_total = elec_II

    # overall_exc_connections = cont_syn_exc_total + elec_exc_total
    # overall_inh_connections = cont_syn_inh_total + elec_inh_total
    # percentages for synaptic categories if totals non-zero
    def pct(x, tot):
        return (100.0 * x / tot) if tot else 0.0

    summary = {
        "population_counts": {
            "total_populations": len(pops_df),
            "exc_populations": int((pops_df.type=="exc").sum()),
            "inh_populations": int((pops_df.type=="inh").sum()),
            "unknown_populations": int((pops_df.type=="unknown").sum())
        },
        "cell_counts": {
            "total_cells": int(pops_df['size'].sum()) if "size" in pops_df.columns else 0,
            "exc_cells": int(pops_df[pops_df.type=="exc"]["size"].sum()) if "size" in pops_df.columns and (pops_df.type=="exc").any() else 0,
            "inh_cells": int(pops_df[pops_df.type=="inh"]["size"].sum()) if "size" in pops_df.columns and (pops_df.type=="inh").any() else 0
        },
        "synaptic_contacts": {
            "EE": int(EE), "EI": int(EI), "IE": int(IE), "II": int(II),
            "total_syn_contacts": int(total_syn_contacts),
            "pct_EE": pct(EE, total_syn_contacts),
            "pct_EI": pct(EI, total_syn_contacts),
            "pct_IE": pct(IE, total_syn_contacts),
            "pct_II": pct(II, total_syn_contacts),
            "cont_syn_exc_total": int(cont_syn_exc_total),
            "cont_syn_inh_total": int(cont_syn_inh_total),
            "pct_cont_syn_exc": pct(cont_syn_exc_total, total_syn_contacts),
            "pct_cont_syn_inh": pct(cont_syn_inh_total, total_syn_contacts)
        },
        "electrical_conductances": {
            "EE": int(elec_EE), "II": int(elec_II),
            "total_electrical": int(total_elec),
            "pct_elec_EE": pct(elec_EE, total_elec),
            "pct_elec_II": pct(elec_II, total_elec)
        },
        "overall_connections": {
            "overall_exc_connections": int(cont_syn_exc_total + elec_EE),
            "overall_inh_connections": int(cont_syn_inh_total + elec_II),
            "pct_overall_exc": pct(cont_syn_exc_total + elec_EE, cont_syn_exc_total + elec_EE + cont_syn_inh_total + elec_II),
            "pct_overall_inh": pct(cont_syn_inh_total + elec_II, cont_syn_exc_total + elec_EE + cont_syn_inh_total + elec_II)
        },
        "pulse_generators": {"exc": exc_pgs, "inh": inh_pgs},
        "inputs": {"total": total_inputs, "exc": int(exc_inputs), "inh": int(inh_inputs)},
        "per_population_table": pop_stats_df.to_dict(orient="index")
    }

    dataset = {
        "pops_df": pops_df,
        "pop_stats_df": pop_stats_df,
        "summary": summary
    }
    return dataset


def save_dataset(dataset, outdir="analysis_out", basename="network_dataset"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # CSVs
    dataset["pops_df"].to_csv(outdir / f"{basename}_populations.csv")
    dataset["pop_stats_df"].to_csv(outdir / f"{basename}_population_stats.csv")
    # JSON summary
    with open(outdir / f"{basename}_summary.json", "w") as fh:
        json.dump(dataset["summary"], fh, indent=2)
        
    return outdir

# Automatically extract and save network data to analysis_out
print("Extracting network data...")
try:
    # Extract data from the generated network
    basename = getattr(nml_doc, "id", "network")
    dataset = extract_network_dataset(net, nml_doc)
    
    # Save to analysis_out directory
    outdir = save_dataset(dataset, outdir="analysis_out", basename=basename)
    print(f"Network data saved to {outdir}")
    
except Exception as e:
    print(f"Warning: Could not extract and save network data: {e}")
    print("Network file was generated successfully, but data extraction failed.")

print("Network generation complete. Data saved to analysis_out/")
print("To run analysis, use: python analysis.py")