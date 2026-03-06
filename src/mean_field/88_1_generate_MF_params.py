#!/usr/bin/env python3
"""
Generate network params using physiology values weighted by per-population connectivity.
This script matches pathway keys from `net_params/*pathways_physiology.json` to
components in `analysis_out/{network}_summary.json`'s `per_population_table` by
matching the suffix after ':' (e.g., 'L23_PC') to component names and uses
'elec_in' (or 'inputs') as weights.

Writes YAMLs into `yaml/`.
"""
import os, glob, json, statistics, re
from copy import deepcopy

try:
    import yaml
except Exception:
    raise RuntimeError('PyYAML required: pip install pyyaml')

# Updated network IDs to match spike file naming convention
network_ids = [
    'spike_TC2PT','spike_TC2CT','spike_TC2IT4_IT2CT','spike_TC2IT2PTCT','spike_max_CTC_plus',
    'spike_M1a_max_plus','spike_M1_max_plus','spike_M2_max_plus','spike_M2aM1aS1a_max_plus',
    'spike_S1bM1bM2b_max_plus','spike_M2M1S1_max_plus',
    'TC2PT','TC2CT','TC2IT4_IT2CT','TC2IT2PTCT','max_CTC_plus',
    'M1a_max_plus','M1_max_plus','M2_max_plus','M2aM1aS1a_max_plus',
    'S1bM1bM2b_max_plus','M2M1S1_max_plus',
]

def weighted_median(values, weights):
    if not values:
        return None
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)
    if total <= 0:
        return None
    cum = 0.0
    for v,w in pairs:
        cum += w
        if cum >= total/2.0:
            return v
    return pairs[-1][0]

def load_template():
    cands = glob.glob('*_network_params.yaml')
    pref = [c for c in cands if 'optimus' in c or 'max' in c]
    pick = pref[0] if pref else (cands[0] if cands else None)
    if pick:
        with open(pick,'r') as fh:
            return yaml.safe_load(fh)
    # minimal
    return {
        'network_id': 'TEMPLATE', 'populations': ['E','I'], 'N':[100,100],
        'p':0.02,'g':5.0,'connection_rule':'fixed_indegree',
        'neuron_type':'iaf_psc_delta','multapses':False,
        'C':{'val':1.0,'unit':'pF'}, 'tau_m':{'val':20.0,'unit':'ms'},
        'tau_r':{'val':2.0,'unit':'ms'}, 'V_0_rel':{'val':0.0,'unit':'mV'},
        'V_th_rel':{'val':15.0,'unit':'mV'}, 'E_L':{'val':0.0,'unit':'mV'},
        'V_m':{'val':0.0,'unit':'mV'}, 'tau_s':{'val':0.0,'unit':'ms'},
        'd':{'val':1.0,'unit':'ms'}, 'delay_dist':'none',
        'j':{'val':0.8,'unit':'mV'}, 'gaussianize':True, 'j_std':0.2,
        'nu_ext':{'val':[300.0,10.0],'unit':'Hz'}, 'I_ext':{'val':1.0,'unit':'pA'}
    }

def compute_p_estimate(total_syn_contacts, N_exc, N_inh):
    N_total = N_exc + N_inh
    if N_total <= 0:
        return 0.0
    p = float(total_syn_contacts) / float(N_total ** 2)
    p = max(min(p,1.0),1e-8)
    return float('{:.6g}'.format(p))

def compute_p_from_anatomy(anatomy_file):
    """
    Compute average connection probability from anatomy file.
    Extracts actual connection probabilities from the anatomy JSON file
    which contains realistic connection probabilities for different pathways.
    """
    if not os.path.exists(anatomy_file):
        print(f'  Warning: Anatomy file {anatomy_file} not found')
        return None
        
    try:
        with open(anatomy_file, 'r') as fh:
            anatomy_data = json.load(fh)
        
        # Extract all connection probabilities
        connection_probs = []
        for connection_key, connection_data in anatomy_data.items():
            if isinstance(connection_data, dict) and 'connection_probability' in connection_data:
                try:
                    prob = float(connection_data['connection_probability'])
                    # Only consider biologically reasonable probabilities (0.001 to 0.5)
                    if 0.001 <= prob <= 0.5:
                        connection_probs.append(prob)
                except (ValueError, TypeError):
                    continue
        
        # Return average if we have any valid probabilities
        if connection_probs:
            avg_prob = sum(connection_probs) / len(connection_probs)
            # Clamp to reasonable biological range
            avg_prob = max(min(avg_prob, 0.2), 0.01)
            print(f'  Using anatomy-based connection probability: {avg_prob:.4f}')
            return avg_prob
        
        print(f'  Warning: No valid connection probabilities found in {anatomy_file}')
        return None
    except Exception as e:
        print(f'  Error reading anatomy file {anatomy_file}: {e}')
        return None

def parse_external_rates_from_spike_file(network_id):
    """
    Parse actual external input rates from spike_*.net.nml files containing 
    TransientPoissonFiringSynapse elements.
    
    Parameters:
    - network_id: network identifier (e.g., 'spike_max_CTC_plus', 'spike_TC2PT')
    
    Returns:
    - [nu_ext_E, nu_ext_I] in Hz, or None if file not found
    """
    # Map network_id to spike filename
    nml_filename_map = {
        'spike_max_CTC_plus': 'spike_max_CTC_plus.net.nml',
        'spike_TC2PT': 'spike_TC2PT.net.nml',
        'spike_TC2CT': 'spike_TC2CT.net.nml',
        'spike_TC2IT4_IT2CT': 'spike_TC2IT4_IT2CT.net.nml',
        'spike_TC2IT2PTCT': 'spike_TC2IT2PTCT.net.nml',
        'spike_M1a_max_plus': 'spike_M1a_max_plus.net.nml',
        'spike_M1_max_plus': 'spike_M1_max_plus.net.nml',
        'spike_M2_max_plus': 'spike_M2_max_plus.net.nml',
        'spike_M2aM1aS1a_max_plus': 'spike_M2aM1aS1a_max_plus.net.nml',
        'spike_M2M1S1_max_plus': 'spike_M2M1S1_max_plus.net.nml',
        'spike_S1bM1bM2b_max_plus': 'spike_S1bM1bM2b_max_plus.net.nml',
        'max_CTC_plus': 'max_CTC_plus.net.nml',
        'TC2PT': 'TC2PT.net.nml',
        'TC2CT': 'TC2CT.net.nml',
        'TC2IT4_IT2CT': 'TC2IT4_IT2CT.net.nml',
        'TC2IT2PTCT': 'TC2IT2PTCT.net.nml',
        'M1a_max_plus': 'M1a_max_plus.net.nml',
        'M1_max_plus': 'M1_max_plus.net.nml',
        'M2_max_plus': 'M2_max_plus.net.nml',
        'M2aM1aS1a_max_plus': 'M2aM1aS1a_max_plus.net.nml',
        'M2M1S1_max_plus': 'M2M1S1_max_plus.net.nml',
        'S1bM1bM2b_max_plus': 'S1bM1bM2b_max_plus.net.nml'
    }
    
    filename = nml_filename_map.get(network_id)
    if not filename or not os.path.exists(filename):
        print(f'  Warning: NML file {filename} not found for network {network_id}')
        return None
    
    try:
        # Parse the NML file to extract TransientPoissonFiringSynapse rates
        exc_rates = []
        inh_rates = []
        
        with open(filename, 'r') as f:
            content = f.read()
            
        # Find all TransientPoissonFiringSynapse elements
        # Updated regex patterns to handle attributes in any order
        # Look for id and averageRate attributes in any order within the element
        pattern_id_rate = r'<transientPoissonFiringSynapse[^>]*id="([^"]*)"[^>]*averageRate="([^"]*)"[^>]*/>'
        pattern_rate_id = r'<transientPoissonFiringSynapse[^>]*averageRate="([^"]*)"[^>]*id="([^"]*)"[^>]*/>'
        
        # Find matches with id first, then averageRate
        matches = re.findall(pattern_id_rate, content, re.IGNORECASE)
        
        # If no matches, try with averageRate first, then id
        if not matches:
            matches_alt = re.findall(pattern_rate_id, content, re.IGNORECASE)
            # Swap to maintain consistent (id, rate) order
            matches = [(match[1], match[0]) for match in matches_alt]
        
        if not matches:
            print(f'  Warning: No TransientPoissonFiringSynapse elements found in {filename}')
            return None
            
        for synapse_id, rate_str in matches:
            # Extract numeric value from rate string (e.g., "300 Hz" -> 300.0)
            rate_match = re.search(r'([\d.]+)', rate_str)
            if not rate_match:
                continue
            rate_value = float(rate_match.group(1))
            
            # Determine if this is excitatory or inhibitory based on synapse ID
            if 'exc_' in synapse_id.lower():
                exc_rates.append(rate_value)
            elif 'inh_' in synapse_id.lower():
                inh_rates.append(rate_value)
            else:
                # If we can't determine, assume it's excitatory (most common case)
                exc_rates.append(rate_value)
        
        if not exc_rates and not inh_rates:
            print(f'  Warning: No valid TransientPoissonFiringSynapse elements found in {filename}')
            return None
            
        # Use the first found rates of each type (they should all be the same)
        avg_exc_rate = exc_rates[0] if exc_rates else statistics.mean(exc_rates[1:]) if len(exc_rates) > 1 else 0.0
        avg_inh_rate = inh_rates[0] if inh_rates else statistics.mean(inh_rates[1:]) if len(inh_rates) > 1 else 0.0
        
        # Ensure we have values for both rates
        if avg_exc_rate == 0.0 and avg_inh_rate > 0.0:
            avg_exc_rate = avg_inh_rate * 1.25
        elif avg_inh_rate == 0.0 and avg_exc_rate > 0.0:
            avg_inh_rate = avg_exc_rate * 0.8
            
        # Handle case where both are zero
        if avg_exc_rate == 0.0 and avg_inh_rate == 0.0:
            return None
            
        print(f'  Parsed external rates from {filename}: E={avg_exc_rate:.2f} Hz, I={avg_inh_rate:.2f} Hz')
        return [avg_exc_rate, avg_inh_rate]
        
    except Exception as e:
        print(f'  Error parsing {filename}: {e}')
        return None

os.makedirs('yaml', exist_ok=True)
template = load_template()

for nid in network_ids:
    print('Processing', nid)
    # Use the full network ID (with spike_ prefix) for summary and anatomy file lookup
    # since the actual files in analysis_out/ include the spike_ prefix
    summary_f = os.path.join('analysis_out', f'{nid}_summary.json')
    anatomy_file = os.path.join('net_params', f'extracted_{nid}_anatomy.json')
    print(f'  Looking for summary file: {summary_f}')
    print(f'  File exists: {os.path.exists(summary_f)}')
    print(f'  Looking for anatomy file: {anatomy_file}')
    print(f'  Anatomy file exists: {os.path.exists(anatomy_file)}')
    
    physio_files = glob.glob(os.path.join('net_params', f'extracted_{nid}_*pathways_physiology.json'))
    # fallback to any extracted_{nid}_*.json
    if not physio_files:
        physio_files = glob.glob(os.path.join('net_params', f'extracted_{nid}_*.json'))

    if not os.path.exists(summary_f):
        print('  missing summary, skipping')
        continue
    with open(summary_f,'r') as fh:
        summary = json.load(fh)

    cell_counts = summary.get('cell_counts',{})
    exc = int(cell_counts.get('exc_cells', cell_counts.get('exc',0) or 0))
    inh = int(cell_counts.get('inh_cells', cell_counts.get('inh',0) or 0))
    
    # Compute p from anatomy file instead of synaptic contacts
    p_est = compute_p_from_anatomy(anatomy_file)
    if p_est is None:
        # Fallback to original method if anatomy file is not available or has no valid data
        syn = summary.get('synaptic_contacts',{})
        total_syn = int(syn.get('total_syn_contacts') or syn.get('total',0) or 0)
        p_est = compute_p_estimate(total_syn, exc, inh)
        # Apply biological constraint to fallback value
        p_est = max(min(p_est, 0.2), 0.01)
        print(f'  Using fallback connection probability: {p_est:.4f}')

    perpop = summary.get('per_population_table',{})
    # build list of component names and their available weights
    comp_weights = {}
    for comp, v in perpop.items():
        # prefer elec_in, fallback to inputs or total_in
        w = v.get('elec_in') or v.get('inputs') or v.get('total_in') or 0
        comp_weights[comp] = w

    # collect pathway values and per-pathway aggregated weight
    pathway_vals = []
    pathway_weights = []
    pathway_cv_vals = []
    pathway_cv_weights = []

    for pf in physio_files:
        try:
            with open(pf,'r') as fh:
                phys = json.load(fh)
        except Exception as e:
            print('  could not read', pf, e)
            continue
        for path_key, info in phys.items():
            if not isinstance(info, dict):
                continue
            # get epsp and cv
            epsp = info.get('epsp_mean')
            cv = info.get('cv_psp_amplitude_mean')
            if epsp is None and cv is None:
                continue
            # compute weight by matching suffix of path_key to components
            suffix = path_key.split(':')[-1]
            wsum = 0
            for comp, w in comp_weights.items():
                if suffix in comp:
                    try:
                        wsum += float(w)
                    except Exception:
                        pass
            if wsum <= 0:
                # fallback to non-zero uniform weight 1
                wsum = 1.0
            if epsp is not None:
                try:
                    pathway_vals.append(float(epsp))
                    pathway_weights.append(wsum)
                except Exception:
                    pass
            if cv is not None:
                try:
                    pathway_cv_vals.append(float(cv))
                    pathway_cv_weights.append(wsum)
                except Exception:
                    pass

    # compute weighted medians
    j_val = weighted_median(pathway_vals, pathway_weights) if pathway_vals else None
    cv_val = weighted_median(pathway_cv_vals, pathway_cv_weights) if pathway_cv_vals else None

    conf = deepcopy(template)
    conf['network_id'] = nid
    conf['N'] = [exc, inh]
    conf['p'] = p_est
    if j_val is not None:
        conf['j'] = {'val': float('{:.6g}'.format(j_val)), 'unit': 'mV'}
    if cv_val is not None:
        conf['j_std'] = float('{:.6g}'.format(cv_val))

    # Compute nu_ext from spike files instead of using heuristic
    nu_ext_vals = parse_external_rates_from_spike_file(nid)
    if nu_ext_vals is not None:
        conf['nu_ext'] = {'val': nu_ext_vals, 'unit': 'Hz'}
        print(f'  Using actual spike file rates: E={nu_ext_vals[0]:.2f} Hz, I={nu_ext_vals[1]:.2f} Hz')
    else:
        # Use standard rates [300.0, 10.0] Hz as fallback instead of heuristic rates
        conf['nu_ext'] = {'val': [300.0, 10.0], 'unit': 'Hz'}
        print(f'  Using standard fallback rates: E=300.00 Hz, I=10.00 Hz')

    outp = os.path.join('yaml', f'{nid}_network_params.yaml')
    with open(outp,'w') as fh:
        yaml.safe_dump(conf, fh, sort_keys=False)
    print(f'  wrote {outp} N={exc}+{inh} p={conf["p"]} j={conf.get("j")} j_std={conf.get("j_std")}')