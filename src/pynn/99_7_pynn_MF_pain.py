# coding: utf-8
"""
Balanced network of excitatory and inhibitory neurons with unified parameters.
Implementation based on benchmarks from Brette et al. (2007) but with parameters
derived from actual network connection statistics.

Andrew Davison, UNIC, CNRS
August 2006
"""

import socket
from math import *
from pyNN.utility import get_simulator, Timer, ProgressBar, init_logging, normalized_filename
from pyNN.random import NumpyRNG, RandomDistribution
import yaml
import os
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

quantum_optimization_performed = False

def initialize_gpu():
    try:
        import cupy as cp
        logger.info("CuPy is available")
        
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            logger.info(f"GPU acceleration available with {device_count} device(s)")
            return cp, True
        except cp.cuda.runtime.CUDARuntimeError as e:
            if "cudaErrorInsufficientDriver" in str(e):
                logger.warning(f"GPU available but CUDA driver version is insufficient: {e}")
                logger.warning("Falling back to CPU-based computations")
                return np, False
            else:
                logger.error(f"Other CUDA error occurred: {e}")
                logger.warning("Falling back to CPU-based computations")
                return np, False
        except Exception as e:
            logger.error(f"Unexpected error when initializing GPU: {e}")
            logger.warning("Falling back to CPU-based computations")
            return np, False
            
    except ImportError:
        logger.info("CuPy not available, using CPU-based computations")
        return np, False

xp, GPU_AVAILABLE = initialize_gpu()

# Try to import quantum-inspired optimization libraries
try:
    from scipy.optimize import dual_annealing
    QUANTUM_ANNEALING_AVAILABLE = True
    logger.info("Quantum-inspired optimization (dual annealing) available")
except ImportError:
    QUANTUM_ANNEALING_AVAILABLE = False
    logger.info("Dual annealing not available, using standard optimization")

# Global CSA availability flag
CSA_AVAILABLE = False
try:
    import csa
    CSA_AVAILABLE = True
except ImportError:
    pass

# === Configure the simulator ================================================

sim, options = get_simulator(
    ("benchmark", "either CUBA or COBA"),
    ("--plot-figure", "plot the simulation results to a file", {"action": "store_true"}),
    ("--use-views", "use population views in creating the network", {"action": "store_true"}),
    ("--use-assembly", "use assemblies in creating the network", {"action": "store_true"}),
    ("--use-csa", "use the Connection Set Algebra to define the connectivity", {"action": "store_true"}),
    ("--debug", "print debugging information"),
    ("--net-id", "Network identifier to automatically find parameter files", 
     {"default": "max_CTC_plus"}),
    ("--gpu-accelerate", "Enable GPU acceleration for large networks", {"action": "store_true"}),
    ("--optimize-params", "Use quantum-inspired optimization for network parameters", {"action": "store_true"}),
    ("--tstop", "Simulation duration (ms)", {"type": float, "default": 1000}),
    ("--scale-down", "scaling network's scale(0.0-1.0)", {"type": float, "default": 1.0})
)

if options.use_csa:
    import csa

if options.debug:
    init_logging(None, debug=True)

timer = Timer()

# === Define parameters ========================================================
# Automatically generate file paths from net_id
net_id = options.net_id
network_params_file = f"yaml/{net_id}_network_params.yaml"
connection_stats_file = f"analysis_out/{net_id}_connection_stats.json"

# Default parameters (will be overridden if YAML file is provided)
threads = 16
rngseed = 98765
parallel_safe = True

n        = 72550  # number of cells
r_ei     = 33560/38990   # number of excitatory cells:number of inhibitory cells
pconn    = 0.528  # connection probability
stim_dur = 1000.   # (ms) duration of random stimulation
dt       = 0.1   # (ms) simulation timestep
tstop    = 1000  # (ms) simulaton duration
rate     = 60.0   # Hz
delay    = 0.2

NEST_CONNECTION_LIMIT = 134217726
 
def load_parameters():    
    params = {
        'n_exc': 33560,
        'n_inh': 38990,
        'n': 72550,
        'rate': 60.0,  
        'pconn': 0.528,
        'delay': 5.5202,
        'tau_m': 20.0,
        'stim_dur': 1000.0,
        'dt': 0.1,
        'tstop': 1000,
        'w_exc': 4.0,   # Default excitatory weight
        'w_inh': 51.0   # Default inhibitory weight (positive value, will be made negative later)
    }
    
    if os.path.exists(network_params_file):
        with open(network_params_file) as f:
            yaml_params = yaml.safe_load(f)
    
        # Load network parameters from YAML
        params['n_exc'] = yaml_params['N'][0]
        params['n_inh'] = yaml_params['N'][1]
        params['pconn'] = yaml_params.get('p', yaml_params.get('connection_probability', 
                        yaml_params.get('epsilon', yaml_params.get('conn_prob', 0.1))))
        
        # Safety check: Cap connection probability to biologically realistic maximum
        MAX_BIOLOGICAL_PCONN = 0.2  # 20% maximum for cortical networks
        if params['pconn'] > MAX_BIOLOGICAL_PCONN:
            logger.warning(f"Connection probability {params['pconn']:.3f} exceeds biological maximum {MAX_BIOLOGICAL_PCONN}. Capping to {MAX_BIOLOGICAL_PCONN}.")
            params['pconn'] = MAX_BIOLOGICAL_PCONN
        
        params['delay'] = yaml_params.get('d', yaml_params.get('delay', 1.5))
        # Handle dictionary format for delay (e.g., {'val': 5.5466, 'unit': 'ms'})
        if isinstance(params['delay'], dict):
            params['delay'] = params['delay'].get('val', 1.5)
        # Synaptic weights
        params['w_exc'] = yaml_params.get('w_exc', yaml_params.get('J_E', 4.0))
        params['w_inh'] = abs(yaml_params.get('w_inh', yaml_params.get('J_I', 51.0)))  
        
        params['tau_m'] = yaml_params.get('tau_m', {}).get('val', params['tau_m'])
        params['stim_dur'] = yaml_params.get('stim_dur', params['stim_dur'])
        params['dt'] = yaml_params.get('dt', params['dt'])
        params['tstop'] = yaml_params.get('tstop', params['tstop'])
        
        # Load external rates
        nu_ext = yaml_params.get('nu_ext', [51435.0, 39859.27993456888])  # Default biological rates
        if isinstance(nu_ext, dict):
            nu_ext = nu_ext.get('val', [51435.0, 39859.27993456888])
        
        # Safety check: Cap external rates to biological maximums
        MAX_EXTERNAL_RATE = 20.0  # Hz - more realistic for cortical networks
        if isinstance(nu_ext, list) and len(nu_ext) >= 2:
            nu_ext[0] = min(nu_ext[0], MAX_EXTERNAL_RATE)  # Excitatory external rate
            nu_ext[1] = min(nu_ext[1], MAX_EXTERNAL_RATE)  # Inhibitory external rate
        params['rate_e'] = nu_ext[0]
        params['rate_i'] = nu_ext[1]
        
        # Calculate overall external rate as weighted average (used for Poisson inputs)
        total_neurons = params['n_exc'] + params['n_inh']
        weighted_rate = (nu_ext[0] * params['n_exc'] + nu_ext[1] * params['n_inh']) / total_neurons
        params['rate'] = weighted_rate
        
        print(f"Loaded parameters from {network_params_file}")
        print(f"  Total neurons: {params['n']} | Excitatory neurons: {params['n_exc']} | Inhibitory neurons: {params['n_inh']}")
        print(f"  Connection probability: {pconn:.4f} | Synaptic delay: {params['delay']} ms | Excitatory weight: {params['w_exc']} | Inhibitory weight: {params['w_inh']}")
    else:
        # Calculate neuron counts using original method
        n_exc = int(round((params['n'] * params['r_ei'] / (1 + params['r_ei']))))
        params['n_exc'] = n_exc
        params['n_inh'] = params['n'] - n_exc
        print(f"Using calculated neuron counts: E={params['n_exc']}, I={params['n_inh']}, Total={params['n']}")
    
    # Initialize expected_connections variable before any potential use
    total_neurons = params['n_exc'] + params['n_inh']
    expected_connections = (params['n_exc'] ** 2 + params['n_exc'] * params['n_inh'] + 
                           params['n_inh'] * params['n_exc'] + params['n_inh'] ** 2) * params['pconn']
    use_gpu = getattr(options, 'gpu_accelerate', True)  # GPU is now enabled by default
    if isinstance(use_gpu, str):
        use_gpu = use_gpu.lower() in ['true', '1', 'yes', 'on']
    
    if use_gpu:
        print(f"Network size of {total_neurons:,} neurons with expected connections: ~{int(expected_connections):,} ")
        if GPU_AVAILABLE:
            print("GPU acceleration: Enabled (default setting)")
        else:
            print("GPU acceleration: Requested but not available, using CPU")
    else:
        print("GPU acceleration: Disabled (use --gpu-accelerate=True to enable)")
    scale_factor = getattr(options, 'scale_down', 1.0)
    if scale_factor != 1.0:
        print(f"apply network scaling factor: {scale_factor}")
        params['n_exc'] = max(1, int(params['n_exc'] * scale_factor))  # Ensure at least 1 neuron
        params['n_inh'] = max(1, int(params['n_inh'] * scale_factor))  # Ensure at least 1 neuron
        params['n'] = params['n_exc'] + params['n_inh']
        print(f"Scaled Total neurons: {params['n']} | Scaled Excitatory neurons: {params['n_exc']} | Scaled Inhibitory neurons: {params['n_inh']}")
        # Recalculate expected connections after scaling
        total_neurons = params['n_exc'] + params['n_inh']
        expected_connections = ((params['n_exc'] * params['pconn']) ** 2 + params['n_exc'] * params['n_inh'] * params['pconn'] + 
                               params['n_inh'] * params['n_exc'] * params['pconn'] + (params['n_inh'] * params['pconn']) ** 2) * params['pconn']
        print(f"Scaled network size of {total_neurons:,} neurons with expected connections: ~{int(expected_connections):,}")

    if expected_connections > NEST_CONNECTION_LIMIT:
        reduction_factor = NEST_CONNECTION_LIMIT / expected_connections
        pconn_scaled = max(0.001, params['pconn'] * reduction_factor)  
        print(f"Scaling down connection probability from {params['pconn']} to {pconn_scaled}")
        params['pconn'] = pconn_scaled
        
    # Add all parameters to the returned dictionary
    params['r_ei'] = params['n_exc'] / params['n_inh'] 
    
    # Load synaptic weight parameter j
    j_param = yaml_params.get('j', 0.1)
    if isinstance(j_param, dict):
        j_param = j_param.get('val', 0.1)
    params['j'] = j_param
    
    # Load synaptic weights (these are typically not used in conductance-based models when Gexc/Ginh are available)
    params['w_exc'] = yaml_params.get('w_exc', yaml_params.get('J_E', 0.1))  # Reduced default
    params['w_inh'] = abs(yaml_params.get('w_inh', yaml_params.get('J_I', 0.1)))  # Reduced default
    
    return params

def extract_net_id(network_params_file, connection_stats_file):
    import os
    net_params = os.path.basename(network_params_file)
    conn_stats = os.path.basename(connection_stats_file)
    if net_params.endswith('_network_params.yaml'):
        net_id = net_params.replace('_network_params.yaml', '')
    elif conn_stats.endswith('_connection_stats.json'):
        net_id = conn_stats.replace('_connection_stats.json', '')
    else:
        raise ValueError("Invalid network_params_file or connection_stats_file")
        
    return net_id

def quantum_inspired_parameter_optimization(params):
    """Use quantum-inspired optimization to find optimal network parameters."""
    global quantum_optimization_performed
    
    if quantum_optimization_performed:
        logger.info("Quantum-inspired optimization already performed, skipping...")
        return params
    
    if not QUANTUM_ANNEALING_AVAILABLE:
        print("Quantum-inspired optimization not available. Using default parameters.")
        return params
    
    print("Performing quantum-inspired parameter optimization...")
    
    def objective(param_vector):
        """Objective function to minimize - based on actual simulation results."""
        # Unpack parameters
        pconn, delay, w_exc, w_inh = param_vector
        
        # Apply constraints to keep parameters in reasonable ranges
        # Limit connection probability to avoid excessive connections
        pconn = max(0.02, min(0.08, pconn))
        
        # Create a temporary copy of parameters for testing
        test_params = params.copy()
        test_params['pconn'] = pconn
        test_params['delay'] = delay
        if 'w_exc' in test_params:
            test_params['w_exc'] = w_exc
        if 'w_inh' in test_params:
            test_params['w_inh'] = w_inh
            
        try:
            # Run a short simulation to evaluate these parameters
            # Create a smaller network for faster evaluation
            # Safely get neuron counts with defaults
            test_n_exc = max(100, test_params.get('n_exc', 800) // 10)  # 10% of excitatory neurons
            test_n_inh = max(50, test_params.get('n_inh', 200) // 10)   # 10% of inhibitory neurons
            
            # Create test populations
            test_pop_exc = sim.Population(test_n_exc, sim.IF_curr_exp(**test_params['cell_params_exc']))
            test_pop_inh = sim.Population(test_n_inh, sim.IF_curr_exp(**test_params['cell_params_inh']))
            
            # Create Poisson input for more realistic firing
            rate = 10.0  # Hz
            noise_exc = sim.Population(test_n_exc, sim.SpikeSourcePoisson(rate=rate))
            noise_inh = sim.Population(test_n_inh, sim.SpikeSourcePoisson(rate=rate))
            
            # Connect noise to populations
            sim.Projection(noise_exc, test_pop_exc, sim.OneToOneConnector(), 
                          sim.StaticSynapse(weight=2.0, delay=delay))
            sim.Projection(noise_inh, test_pop_inh, sim.OneToOneConnector(), 
                          sim.StaticSynapse(weight=2.0, delay=delay))
            
            # Determine connector type based on CSA availability and user preference
            use_csa_flag = getattr(options, 'use_csa', False) and CSA_AVAILABLE
            
            if use_csa_flag:
                connector = sim.CSAConnector(csa.cset(csa.random(pconn)))
            else:
                connector = sim.FixedProbabilityConnector(pconn)
            
            # Connect excitatory to inhibitory
            sim.Projection(test_pop_exc, test_pop_inh, 
                          connector,
                          sim.StaticSynapse(weight=w_exc, delay=delay))
            
            # Connect inhibitory to excitatory
            sim.Projection(test_pop_inh, test_pop_exc,
                          connector, 
                          sim.StaticSynapse(weight=w_inh, delay=delay))
            
            # Record spikes
            test_pop_exc.record('spikes')
            test_pop_inh.record('spikes')
            
            # Run short simulation
            sim.run(500.0)  # 500ms simulation for evaluation
            
            # Calculate firing rates
            exc_spikes = test_pop_exc.get_data('spikes').segments[0].spiketrains
            inh_spikes = test_pop_inh.get_data('spikes').segments[0].spiketrains
            
            exc_rate = sum(len(st) for st in exc_spikes) / (test_n_exc * 0.5)  # Firing rate in Hz
            inh_rate = sum(len(st) for st in inh_spikes) / (test_n_inh * 0.5)  # Firing rate in Hz
            
            # Reset the simulator for next iteration
            sim.reset()
            
            # Objective: target firing rates based on official PyNN benchmarks
            if options.benchmark == "COBA":
                target_exc_rate = 13.0  # COBA should have ~13 Hz
                target_inh_rate = 13.0  # COBA should have ~13 Hz
            else:  # CUBA
                target_exc_rate = 6.0   # CUBA should have ~6 Hz  
                target_inh_rate = 6.0   # CUBA should have ~6 Hz
            
            # Penalty for deviation from target rates
            exc_penalty = abs(exc_rate - target_exc_rate) / target_exc_rate if target_exc_rate > 0 else abs(exc_rate - target_exc_rate)
            inh_penalty = abs(inh_rate - target_inh_rate) / target_inh_rate if target_inh_rate > 0 else abs(inh_rate - target_inh_rate)
            
            # Balance penalty (E:I ratio should be close to 1:1 for both models)
            if inh_rate > 0:
                ei_ratio = exc_rate / inh_rate
                balance_penalty = abs(ei_ratio - 1.0) * 5  # Target 1:1 ratio
            else:
                balance_penalty = 100  # Large penalty if inh_rate is zero
            
            # Connection probability penalty (prefer values in 0.02-0.08 range)
            if pconn < 0.02:
                pconn_penalty = (0.02 - pconn) * 1000
            elif pconn > 0.08:
                pconn_penalty = (pconn - 0.08) * 1000
            else:
                pconn_penalty = 0
            
            total_penalty = exc_penalty + inh_penalty + balance_penalty + pconn_penalty
            
            print(f"Test: pconn={pconn:.3f}, delay={delay:.1f}, w_exc={w_exc:.2f}, w_inh={w_inh:.2f}")
            print(f"      Exc rate: {exc_rate:.2f} Hz, Inh rate: {inh_rate:.2f} Hz, Penalty: {total_penalty:.2f}")
            
            return total_penalty
            
        except Exception as e:
            # print(f"Simulation failed with parameters {param_vector}: {e}")
            return 1000.0 # Return a large penalty value for failed simulations
    
    # Parameter bounds [pconn, delay, w_exc, w_inh]
    # Use bounds that match typical PyNN benchmark ranges
    if options.benchmark == "COBA":
        # COBA: inhibitory conductance should be positive
        inh_weight_bounds = (1.0, 100.0)
    else:  # CUBA
        # CUBA: inhibitory current should be negative
        inh_weight_bounds = (-100.0, -1.0)
    
    bounds = [
        (0.02, 0.08),    # Connection probability - benchmark range (not too low, not too high)
        (0.1, 5.0),       # Delay
        (0.1, 20.0),      # Excitatory weight - wider range for better exploration
        inh_weight_bounds # Inhibitory weight (positive for COBA, negative for CUBA)
    ]
    
    # Perform dual annealing optimization (quantum-inspired)
    # dual_annealing_kwargs contains only maxiter and optional workers parameters
    try:
        # Check if dual_annealing supports workers parameter
        import inspect
        sig = inspect.signature(dual_annealing)
        dual_annealing_kwargs = {
            'maxiter': 20,           # Fewer iterations for speed
            'initial_temp': 50,      # Higher initial temperature for better exploration
            'visit': 2.62,           # Visit parameter for faster convergence
            'accept': -5.0           # Accept parameter for faster convergence
        }
        
        if 'workers' in sig.parameters:
            # Newer scipy versions support workers parameter for parallel processing
            dual_annealing_kwargs['workers'] = -1  # Use all available CPU cores for parallel processing
            
        result = dual_annealing(objective, bounds, **dual_annealing_kwargs)
            
        pconn_opt, delay_opt, w_exc_opt, w_inh_opt = result.x
        
        # Apply final constraints to match benchmark standards
        pconn_opt = max(0.02, min(0.08, pconn_opt))
        delay_opt = max(0.1, min(5.0, delay_opt))
        

        print(f"Optimized parameters:")
        print(f"  Connection probability: {pconn_opt:.4f} | Delay: {delay_opt:.2f} ms | Excitatory weight: {w_exc_opt:.2f} | Inhibitory weight: {w_inh_opt:.2f}")
        
        # Update parameters
        params['pconn'] = pconn_opt
        params['delay'] = delay_opt
        if 'w_exc' in params:
            params['w_exc'] = w_exc_opt
        if 'w_inh' in params:
            params['w_inh'] = w_inh_opt
        
        quantum_optimization_performed = True
        
    except Exception as e:
        print(f"Optimization failed: {e}. Using default parameters.")
    
    return params

params = load_parameters()
if getattr(options, 'optimize_params', False):
    # Check if we're using pre-calculated parameters (mean-field or biological)
    # If net_id is specified and not default, we're using pre-calculated parameters
    if hasattr(options, 'net_id') and options.net_id != "default":
        logger.warning("Parameter optimization disabled: using pre-calculated mean-field/biological parameters from net_id='%s'", options.net_id)
        logger.info("To enable optimization, use --net-id default or omit --net-id")
    else:
        params = quantum_inspired_parameter_optimization(params)
n_exc = params['n_exc']
n_inh = params['n_inh']
n = params['n']
rate = params['rate']
pconn = params['pconn']
delay = params['delay']
tstop = params['tstop']
stim_dur = params['stim_dur']

# GPU-accelerated network creation for large networks
def create_gpu_accelerated_network(sim, params):
    """Create a network using GPU-accelerated techniques for large-scale simulations."""
    print("Creating GPU-accelerated large-scale network...")
    
    n_exc = params['n_exc']
    n_inh = params['n_inh']
    pconn = params['pconn']
    
    # Check if network is large enough to require GPU acceleration
    total_neurons = n_exc + n_inh
    expected_connections = n_exc * n_exc * pconn + n_exc * n_inh * pconn + n_inh * n_exc * pconn + n_inh * n_inh * pconn
    
    print(f"Network size: {total_neurons:,} neurons with expected connections: ~{int(expected_connections):,}")
    
    # For very large networks, reduce connection probability to avoid memory issues
    if total_neurons > NEST_CONNECTION_LIMIT or expected_connections > NEST_CONNECTION_LIMIT:  # 50k neurons or 5M connections
        scale_factor = min(1.0, NEST_CONNECTION_LIMIT / total_neurons) if total_neurons > 0 else 1.0
        pconn_scaled = max(0.01, pconn * scale_factor)
        print(f"Scaling down connection probability from {pconn} to {pconn_scaled} for memory efficiency")
        pconn = pconn_scaled
    
    # Update params with scaled values
    params['pconn'] = pconn
    
    return params

min_delay = min(0.1, params['delay'] * 0.9)
max_delay = max(1.0, params['delay'] * 2)  # Set max_delay to be at least twice the delay or 1.0, whichever is larger


# cell_params
area   = 20000. # (µm²) 
tau_m  = 20.    # (ms)
cm     = 1.     # (µF/cm²)
g_leak = 5e-5   # (S/cm²) 
if options.benchmark == "COBA":
    E_leak   = -75.  # (mV)
elif options.benchmark == "CUBA":
    E_leak   = -60.  # (mV)

v_thresh = -50.   # (mV)
v_reset  = -75.   # (mV)
t_refrac = 5.     # (ms) (clamped at v_reset)
v_mean   = -75.   # (mV) 'mean' membrane potential, for calculating CUBA weights
tau_exc  = 5.     # (ms)
tau_inh  = 10.    # (ms) 

if options.benchmark == "COBA":
    Gexc = 4.     # (nS)
    Ginh = 51.    # (nS)
elif options.benchmark == "CUBA":
    Gexc = 0.27   # (nS) #Those weights should be similar to the COBA weights
    Ginh = 4.5    # (nS) # but the delpolarising drift should be taken into account
Erev_exc = 0.     # (mV)
Erev_inh = -80.   # (mV)

syn_params = {
    "Gexc": Gexc,
    "Ginh": Ginh,
    "Erev_exc": Erev_exc,
    "Erev_inh": Erev_inh,
}    

def initialize_membrane_potentials(cells, v_reset, v_thresh, rng):
    use_gpu = getattr(options, 'gpu_accelerate', False) and GPU_AVAILABLE
    
    if use_gpu:
        uniform_values = xp.random.uniform(v_reset, v_thresh, cells.size)
        if hasattr(uniform_values, 'get'):
            uniform_values = uniform_values.get()
    else:
        uniform_values = np.random.uniform(v_reset, v_thresh, cells.size)
    
    cells.initialize(v=uniform_values)
# ===Calculate derived parameters==============================================

# Calculate cm from the cell parameters
area  = area*1e-8                     # convert to cm²
cm    = cm*area*1000                  # convert to nF
Rm    = 1e-6/(g_leak*area)            # membrane resistance in MΩ
assert tau_m == cm*Rm                 # just to check
n_exc = int(round((n*r_ei/(1+r_ei)))) # number of excitatory cells
n_inh = n - n_exc                     # number of inhibitory cells

if options.benchmark == "COBA":
    celltype = sim.IF_cond_exp
    w_exc    = Gexc*1e-3              # We convert conductances to uS
    w_inh    = Ginh*1e-3              # Positive
    print(f"Using COBA model with IF_cond_exp cell type")
elif options.benchmark == "CUBA":
    celltype = sim.IF_curr_exp
    w_exc = 1e-3*Gexc*(Erev_exc - v_mean)  # (nA) weight of excitatory synapses
    w_inh = 1e-3*Ginh*(Erev_inh - v_mean)  # (nA) 
    assert w_exc > 0; assert w_inh < 0
    print(f"Using CUBA model with IF_curr_exp cell type")


# GPU-accelerated connection creation
def create_connections_with_gpu(sim, pre_cells, post_cells, connector, synapse, receptor_type):
    """Create connections using GPU acceleration for large networks."""
    # For very large networks, we might want to split the connection creation
    total_neurons = pre_cells.size + post_cells.size
    
    # Calculate expected number of connections
    if hasattr(connector, 'p_connect') and connector.p_connect:
        pconn = connector.p_connect
    elif hasattr(connector, 'p') and connector.p:
        pconn = connector.p
    else:
        # Default fallback
        pconn = 0.1
        
    expected_connections = pre_cells.size * post_cells.size * pconn
    
    # Check if we're likely to exceed NEST's connection limit (134,217,726)
    if expected_connections > 134217726:  # 134 million
        print(f"Warning: Expected connections ({int(expected_connections):,}) had exceed NEST limit")
        print("Consider reducing network size or connection probability")
    
    if GPU_AVAILABLE and total_neurons > 2000000:
        print(f"Using GPU-accelerated connection creation for {total_neurons} neurons")
        return sim.Projection(pre_cells, post_cells, connector, synapse, receptor_type=receptor_type)
    else:
        return sim.Projection(pre_cells, post_cells, connector, synapse, receptor_type=receptor_type)

# === Build the network ========================================================

extra = {'threads' : threads,
         'filename': "va_%s.xml" % options.benchmark,
         'label': 'VA'}
if options.simulator == "neuroml":
    extra["file"] = "VAbenchmarks.xml"

node_id = sim.setup(timestep=dt, min_delay=min_delay, max_delay=max_delay, **extra)
num_processes = sim.num_processes()

host_name = socket.gethostname()
print("Host #%d is on %s" % (node_id + 1, host_name))

print("%s Initialising the simulator with %d thread(s)..." % (node_id, extra['threads']))

E_leak = -60. if options.benchmark == "COBA" else -49.
cell_params = {
    'tau_m'      : tau_m,    
    'tau_syn_E'  : tau_exc,  
    'tau_syn_I'  : tau_inh,
    'v_rest'     : E_leak,   
    'v_reset'    : v_reset,  
    'v_thresh'   : v_thresh,
    'cm'         : cm,       
    'tau_refrac' : t_refrac
}

if (options.benchmark == "COBA"):
    cell_params['e_rev_E'] = Erev_exc
    cell_params['e_rev_I'] = Erev_inh

timer.start()

print("%s Creating cell populations..." % node_id)
if options.use_views:
    # create a single population of neurons, and then use population views to define excitatory and inhibitory sub-populations
    all_cells = sim.Population(n_exc + n_inh, celltype(**cell_params), label="All Cells")
    exc_cells = all_cells[:n_exc]
    exc_cells.label = "Excitatory cells"
    inh_cells = all_cells[n_exc:]
    inh_cells.label = "Inhibitory cells"
else:
    # create separate populations for excitatory and inhibitory neurons
    exc_cells = sim.Population(n_exc, celltype(**cell_params), label="Excitatory_Cells")
    inh_cells = sim.Population(n_inh, celltype(**cell_params), label="Inhibitory_Cells")
    if options.use_assembly:
        # group the populations into an assembly
        all_cells = exc_cells + inh_cells

# Configure spike recording for both populations
exc_cells.record('spikes')
inh_cells.record('spikes')

stim_dur =params['stim_dur']
if options.benchmark == "COBA":
    ext_stim_exc = sim.Population(100, sim.SpikeSourcePoisson(rate=rate, duration=stim_dur), label="ext_poisson_exc")
    ext_stim_inh = sim.Population(100, sim.SpikeSourcePoisson(rate=rate, duration=stim_dur), label="ext_poisson_inh")
    rconn = 0.01
    ext_conn = sim.FixedProbabilityConnector(rconn)
    ext_syn = sim.StaticSynapse(weight=0.1)

print("%s Initialising membrane potential to random values..." % node_id)
rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
uniformDistr = RandomDistribution('uniform', low=v_reset, high=v_thresh, rng=rng)
if options.use_views:
    all_cells.initialize(v=uniformDistr)
else:
    exc_cells.initialize(v=uniformDistr)
    inh_cells.initialize(v=uniformDistr)

# Parse the gpu-accelerate option for membrane potential initialization
gpu_accelerate_str = getattr(options, 'gpu_accelerate', 'True')
if isinstance(gpu_accelerate_str, str):
    use_gpu = gpu_accelerate_str.lower() in ['true', '1', 'yes', 'on']
else:
    use_gpu = bool(gpu_accelerate_str)

# Use GPU-accelerated initialization if available and requested
if use_gpu and GPU_AVAILABLE:
    print(f"Initializing membrane potentials with GPU acceleration for {exc_cells.size} exc_cells and {inh_cells.size} inh_cells")
    initialize_membrane_potentials(exc_cells, v_reset, v_thresh, rng)
else:
    print(f"Initializing membrane potentials for {exc_cells.size} exc_cells and {inh_cells.size} inh_cells")
    initialize_membrane_potentials(exc_cells, v_reset, v_thresh, rng)
    initialize_membrane_potentials(inh_cells, v_reset, v_thresh, rng)

# Parse the gpu-accelerate option for network creation
gpu_accelerate_str = getattr(options, 'gpu_accelerate', 'True')
if isinstance(gpu_accelerate_str, str):
    use_gpu = gpu_accelerate_str.lower() in ['true', '1', 'yes', 'on']
else:
    use_gpu = bool(gpu_accelerate_str)

print("%s connecting populations..." % node_id)
progress_bar = ProgressBar(width=30)

# Use CSA connector if explicitly requested and available
use_csa_flag = getattr(options, 'use_csa', False) and CSA_AVAILABLE

if use_csa_flag:
    connector = sim.CSAConnector(csa.cset(csa.random(pconn)))
else:
    connector = sim.FixedProbabilityConnector(pconn, rng=rng, callback=progress_bar)
exc_syn = sim.StaticSynapse(weight=w_exc, delay=delay)
inh_syn = sim.StaticSynapse(weight=w_inh, delay=delay)  # Use w_inh directly for COBA

connections = {}
if options.use_views or options.use_assembly:
    connections['exc'] = sim.Projection(exc_cells, all_cells, connector, exc_syn, receptor_type='excitatory')
    connections['inh'] = sim.Projection(inh_cells, all_cells, connector, inh_syn, receptor_type='inhibitory')
    if (options.benchmark == "COBA"):
        connections['ext2e'] = sim.Projection(ext_stim_exc, all_cells, ext_conn, ext_syn, receptor_type='excitatory')
        connections['ext2i'] = sim.Projection(ext_stim_inh, all_cells, ext_conn, ext_syn, receptor_type='excitatory')
else:
    connections['e2e'] = sim.Projection(exc_cells, exc_cells, connector, exc_syn, receptor_type='excitatory')
    connections['e2i'] = sim.Projection(exc_cells, inh_cells, connector, exc_syn, receptor_type='excitatory')
    connections['i2e'] = sim.Projection(inh_cells, exc_cells, connector, inh_syn, receptor_type='inhibitory')
    connections['i2i'] = sim.Projection(inh_cells, inh_cells, connector, inh_syn, receptor_type='inhibitory')
    if (options.benchmark == "COBA"):
        connections['ext2e'] = sim.Projection(ext_stim_exc, exc_cells, ext_conn, ext_syn, receptor_type='excitatory')
        connections['ext2i'] = sim.Projection(ext_stim_inh, inh_cells, ext_conn, ext_syn, receptor_type='excitatory')

# === Setup recording ==========================================================
print("%s Setting up recording..." % node_id)
if options.use_views or options.use_assembly:
    all_cells.record('spikes')
    exc_cells[[0, 1]].record('v')
    inh_cells[[0, 1]].record('v')
else:
    exc_cells.record('spikes')
    inh_cells.record('spikes')
    exc_cells[0, 1].record('v')
    inh_cells[0, 1].record('v') 

buildCPUTime = timer.diff()

# GPU-accelerated post-simulation analysis
def analyze_results_with_gpu(exc_cells, inh_cells, tstop):
    """Analyze simulation results using GPU acceleration if available and enabled."""
    use_gpu = getattr(options, 'gpu_accelerate', True)  # GPU is now enabled by default
    if isinstance(use_gpu, str):
        use_gpu = use_gpu.lower() in ['true', '1', 'yes', 'on']
    
    use_gpu = use_gpu and GPU_AVAILABLE
    
    print("Performing result analysis%s..." % (" with GPU acceleration (default setting)" if use_gpu and GPU_AVAILABLE else (" with GPU acceleration requested but not available, using CPU" if getattr(options, 'gpu_accelerate', True) and not GPU_AVAILABLE else " with CPU")))
    
    # Get spike data
    exc_data = exc_cells.get_data().segments[0]
    inh_data = inh_cells.get_data().segments[0]
    
    print(f"Excitatory data segments: {len(exc_cells.get_data().segments)} | Inhibitory data segments: {len(inh_cells.get_data().segments)}")
    print(f"Excitatory spiketrains: {len(exc_data.spiketrains) if hasattr(exc_data, 'spiketrains') else 'N/A'}")
    print(f"Inhibitory spiketrains: {len(inh_data.spiketrains) if hasattr(inh_data, 'spiketrains') else 'N/A'}")
    
    # Check if we have any spike data at all
    if not hasattr(exc_data, 'spiketrains') or not hasattr(inh_data, 'spiketrains'):
        print("No spike data found!")
        return 0, 0, 0.0, 0.0
    
    # Count neurons with spikes
    exc_neurons_with_spikes = 0
    inh_neurons_with_spikes = 0
    
    # Use GPU for spike analysis if available and enabled
    try:
        exc_spike_times = []
        for i, spiketrain in enumerate(exc_data.spiketrains):
            if len(spiketrain) > 0:
                exc_neurons_with_spikes += 1
                if use_gpu:
                    times = xp.array(spiketrain.times)
                    exc_spike_times.extend(times.get() if hasattr(times, 'get') else times)
                else:
                    exc_spike_times.extend(spiketrain.times)
            if i < 5:  # Print first 5 neurons spike info
                print(f"Exc neuron {i}: {len(spiketrain)} spikes")
        
        inh_spike_times = []
        for i, spiketrain in enumerate(inh_data.spiketrains):
            if len(spiketrain) > 0:
                inh_neurons_with_spikes += 1
                if use_gpu:
                    times = xp.array(spiketrain.times)
                    inh_spike_times.extend(times.get() if hasattr(times, 'get') else times)
                else:
                    inh_spike_times.extend(spiketrain.times)
            if i < 5:  # Print first 5 neurons spike info
                print(f"Inh neuron {i}: {len(spiketrain)} spikes")
        
        print(f"Excitatory neurons with spikes: {exc_neurons_with_spikes}/{exc_cells.size}")
        print(f"Inhibitory neurons with spikes: {inh_neurons_with_spikes}/{inh_cells.size}")
        
        # Calculate spike counts using GPU if available and enabled
        if use_gpu:
            exc_spike_times = xp.array(exc_spike_times)
            inh_spike_times = xp.array(inh_spike_times)
            
            E_total_spikes = len(exc_spike_times)
            I_total_spikes = len(inh_spike_times)
            
            # Convert back to CPU for final calculations
            if hasattr(exc_spike_times, 'get'):
                exc_spike_times = exc_spike_times.get()
            if hasattr(inh_spike_times, 'get'):
                inh_spike_times = inh_spike_times.get()
        else:
            E_total_spikes = len(exc_spike_times)
            I_total_spikes = len(inh_spike_times)
        
        E_mean_spike_count = E_total_spikes / exc_cells.size if exc_cells.size > 0 else 0
        I_mean_spike_count = I_total_spikes / inh_cells.size if inh_cells.size > 0 else 0
        
        E_rate = E_mean_spike_count * 1000.0 / tstop if tstop > 0 else 0
        I_rate = I_mean_spike_count * 1000.0 / tstop if tstop > 0 else 0
        
        print(f"Excitatory spikes (total): {E_total_spikes} | Inhibitory spikes (total): {I_total_spikes}")
        print(f"Excitatory neurons: {exc_cells.size} | Inhibitory neurons: {inh_cells.size}")
        print(f"Excitatory mean spike count per neuron: {E_mean_spike_count:.2f} | Inhibitory mean spike count per neuron: {I_mean_spike_count:.2f}")
        print(f"Simulation time: {tstop} ms")
        print(f"Excitatory rate: {E_rate:.2f} Hz | Inhibitory rate: {I_rate:.2f} Hz")
        
        return E_total_spikes, I_total_spikes, E_rate, I_rate
    except Exception as e:
        print(f"GPU-accelerated analysis failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to standard analysis
        # Calculate total spike count manually to match the main path calculation
        exc_data = exc_cells.get_data().segments[0]
        inh_data = inh_cells.get_data().segments[0]
        
        exc_spike_times = []
        for spiketrain in exc_data.spiketrains:
            if len(spiketrain) > 0:
                exc_spike_times.extend(spiketrain.times)
        
        inh_spike_times = []
        for spiketrain in inh_data.spiketrains:
            if len(spiketrain) > 0:
                inh_spike_times.extend(spiketrain.times)
        
        E_total_spikes = len(exc_spike_times)
        I_total_spikes = len(inh_spike_times)
        
        E_mean_spike_count = E_total_spikes / exc_cells.size if exc_cells.size > 0 else 0
        I_mean_spike_count = I_total_spikes / inh_cells.size if inh_cells.size > 0 else 0
        
        E_rate = E_mean_spike_count * 1000.0 / tstop if tstop > 0 else 0
        I_rate = I_mean_spike_count * 1000.0 / tstop if tstop > 0 else 0
        
        print(f"Excitatory spikes (total): {E_total_spikes} | Inhibitory spikes (total): {I_total_spikes}")
        print(f"Excitatory neurons: {exc_cells.size} | Inhibitory neurons: {inh_cells.size}")
        print(f"Excitatory mean spike count per neuron: {E_mean_spike_count:.2f} | Inhibitory mean spike count per neuron: {I_mean_spike_count:.2f}")
        print(f"Simulation time: {tstop} ms")
        print(f"Excitatory rate: {E_rate:.2f} Hz | Inhibitory rate: {I_rate:.2f} Hz")
        
        return E_total_spikes, I_total_spikes, E_rate, I_rate


# === Run simulation ===========================================================
print("%d Running simulation..." % node_id)

sim.run(tstop)

simCPUTime = timer.elapsedTime()

# Get spike data
exc_data = exc_cells.get_data().segments[0]
inh_data = inh_cells.get_data().segments[0]

# Calculate spike counts and rates
exc_spike_times = []
for spiketrain in exc_data.spiketrains:
    if len(spiketrain) > 0:
        exc_spike_times.extend(spiketrain.times)

inh_spike_times = []
for spiketrain in inh_data.spiketrains:
    if len(spiketrain) > 0:
        inh_spike_times.extend(spiketrain.times)

# Calculate total spike counts
E_total_spikes = len(exc_spike_times)
I_total_spikes = len(inh_spike_times)

# Calculate mean spike count per neuron
E_mean_spike_count = E_total_spikes / exc_cells.size if exc_cells.size > 0 else 0
I_mean_spike_count = I_total_spikes / inh_cells.size if inh_cells.size > 0 else 0

# Calculate firing rates (Hz) = mean spike count / simulation time (seconds)
E_rate = E_mean_spike_count * 1000.0 / tstop if tstop > 0 else 0
I_rate = I_mean_spike_count * 1000.0 / tstop if tstop > 0 else 0

print(f"excitatory spikes counts: {E_total_spikes} | inhibitory spikes counts: {I_total_spikes}")
print(f"excitatory mean spike count per neuron: {E_mean_spike_count:.2f} | inhibitory mean spike count per neuron: {I_mean_spike_count:.2f}")
print(f"excitatory spikes rates: {E_rate:.2f} Hz | inhibitory spikes rates: {I_rate:.2f} Hz")

# = Print results to file ===============================

print("%d Writing data to file..." % node_id)

# Parse the gpu-accelerate option for final status report
gpu_accelerate_str = getattr(options, 'gpu_accelerate', 'True')
if isinstance(gpu_accelerate_str, str):
    use_gpu = gpu_accelerate_str.lower() in ['true', '1', 'yes', 'on']
else:
    use_gpu = bool(gpu_accelerate_str)

# Extract network ID from the parameters file

network_params_file = getattr(options, 'network_params', f'yaml/{net_id}_network_params.yaml')
saveCPUTime = timer.diff()
net_id = extract_net_id(network_params_file, connection_stats_file)
# Save files with network ID in the filename
filename = normalized_filename("PyNN_pain_results", "%s_%s_exc" % (net_id, options.benchmark), "pkl",
                               options.simulator)
exc_cells.write_data(filename,
                     annotations={'script_name': __file__})
inh_cells.write_data(filename.replace("exc", "inh"),
                     annotations={'script_name': __file__})

writeCPUTime = timer.diff()

if options.use_views or options.use_assembly:
    connections = "%d e→e,i  %d i→e,i" % (connections['exc'].size(),
                                          connections['inh'].size())
else:
    connections = u"%d e→e  %d e→i  %d i→e  %d i→i" % (connections['e2e'].size(),
                                                       connections['e2i'].size(),
                                                       connections['i2e'].size(),
                                                       connections['i2i'].size())
                

if node_id == 0:
    print("\n--- Vogels-Abbott Network Simulation ---")
    print("Nodes                          : %d" % num_processes)
    print("Simulation type                : %s" % options.benchmark)
    print("Number of Neurons              : %d" % n)
    print("Number of Excitatory Neurons   : %d" % n_exc)
    print("Number of Inhibitory Neurons   : %d" % n_inh)
    print("Number of Synapses             : %s" % connections)
    print("Connection propobility         : %.4f" % pconn)
    print("Excitatory conductance         : %g nS" % Gexc)
    print("Inhibitory conductance         : %g nS" % Ginh)
    print("Excitatory rate                : %g Hz" % E_rate)
    print("Inhibitory rate                : %g Hz" % I_rate)
    print("Build time                     : %g s" % buildCPUTime)
    print("Simulation time                : %g s" % simCPUTime)
    
    # Print GPU acceleration status
    if getattr(options, 'gpu_accelerate', False) and GPU_AVAILABLE:
        print("GPU acceleration           : Enabled")
    elif getattr(options, 'gpu_accelerate', False) and not GPU_AVAILABLE:
        print("GPU acceleration           : Requested but not available")
    else:
        print("GPU acceleration           : Not enabled")
    
    if getattr(options, 'optimize_params', False) and QUANTUM_ANNEALING_AVAILABLE:
        print("Parameter optimization     : Quantum-inspired (dual annealing)")
    elif getattr(options, 'optimize_params', False) and not QUANTUM_ANNEALING_AVAILABLE:
        print("Parameter optimization     : Requested but not available")
    else:
        print("Parameter optimization     : Not enabled")

# === Finished with simulator ==========================================================

sim.end()
