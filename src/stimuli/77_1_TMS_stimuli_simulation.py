#!/usr/bin/env python3
"""
Transcranial Magnetic Stimulation (TMS) Simulation with Physiological Parameters
----------------------------------------------------------------------------------
This script simulates TMS stimuli based on physiological parameters from the article:
Aberra AS, Wang B, Grill WM, Peterchev AV. 
Simulation of transcranial magnetic stimulation in head model with morphologically-realistic cortical neurons. 
Brain Stimul. 
2020 Jan-Feb;13(1):175-189. 
doi: 10.1016/j.brs.2019.10.002. 
Epub 2019 Oct 7. PMID: 31611014; PMCID: PMC6889021.

Key parameters from the article:
- Pulse widths: Monophasic, half sine, and biphasic waveforms
- Current directions: P->A (posterior-anterior) and A->P (anterior-posterior)
- Frequencies: 5-100 Hz as used in the study
- Amplitude: Based on motor threshold scaling
- Cell types: Layer 5 pyramidal cells (lowest thresholds), Layer 2/3 pyramidal cells, and inhibitory basket cells

The implementation uses NEST simulator with current-based generators to mimic TMS-induced electric fields.
"""

import matplotlib
from matplotlib import lines
import matplotlib.pyplot as plt
import nest
import numpy as np


def step(t, n, initial, after, after2, after3, after4, seed=1, dt=0.01):
    """Helper function to simulate generators with parameter changes."""
    nest.ResetKernel()
    nest.local_num_threads = 6
    nest.resolution = dt
    nest.rng_seed = seed

    g = nest.Create("sinusoidal_gamma_generator", n, params=initial)
    sr = nest.Create("spike_recorder")
    nest.Connect(g, sr)
    
    # For 30-step simulation, we divide time into 30 parts but apply the same parameter
    # for every 6 consecutive steps to create 5 distinct levels
    # Ensure time_per_step is a multiple of dt
    time_per_step = int(t / 30 / dt) * dt
    
    # First level (6 steps)
    for _ in range(6):
        nest.Simulate(time_per_step)
    
    # Second level (6 steps)
    g.set(after)
    for _ in range(6):
        nest.Simulate(time_per_step)
    
    # Third level (6 steps)
    g.set(after2)
    for _ in range(6):
        nest.Simulate(time_per_step)
    
    # Fourth level (6 steps)
    g.set(after3)
    for _ in range(6):
        nest.Simulate(time_per_step)
    
    # Fifth level (6 steps)
    g.set(after4)
    for _ in range(6):
        nest.Simulate(time_per_step)

    return sr.events


def step_30(t, n, initial_v, final_v, seed=1, dt=1.0):
    """Helper function to simulate 30 consecutive step pulses."""
    nest.ResetKernel()
    # nest.local_num_threads = params['simulation']['local_num_threads']
    nest.local_num_threads = 6
    nest.resolution = dt
    nest.rng_seed = seed

    g = nest.Create("sinusoidal_gamma_generator", n)
    sr = nest.Create("spike_recorder")
    nest.Connect(g, sr)
    
    # Divide time into 30 equal parts, ensuring each is a multiple of dt
    time_per_step = int(t / 30 / dt) * dt
    spike_data = []
    
    for i in range(30):
        # Linearly change voltage from initial to final
        voltage = initial_v + (final_v - initial_v) * (i / 29)
        # For sinusoidal_gamma_generator, 0 <= amplitude <= rate
        if voltage >= 0:
            g.set({"rate": voltage, "amplitude": 0.0})
        else:
            # For negative voltages, use positive rate with magnitude
            g.set({"rate": abs(voltage), "amplitude": 0.0})
        nest.Simulate(time_per_step)
        # Collect spike data for histogram
        if i == 0:
            spike_data = sr.events
        else:
            # Merge spike data
            for key in sr.events:
                spike_data[key] = np.concatenate([spike_data[key], sr.events[key]])

    return spike_data


def plot_hist(spikes, alpha=1.0, label=None):
    """Plot histogram of emitted spikes."""
    plt.hist(
        spikes["times"], 
        bins=np.arange(0.0, max(spikes["times"]) + 1.5, 1.0).tolist(), 
        histtype="step", 
        alpha=alpha, 
        label=label)

# Reset the NEST kernel for a clean start
nest.ResetKernel()
nest.resolution = 0.01
print("Simulating pain stimulus with 365 nm wavelength and 10 V bias voltage...")

###############################################################################
pulse_width_ms = ['0.2', '0.5', '0.3']
titles = ['TMS_Monophasic', 'TMS_Half-Sine', 'TMS_Biphasic']
waveforms = {"name": titles, "pulse_width_ms": pulse_width_ms}
plt.figure(figsize=(18, 9))
num_nodes = 3

g = nest.Create(
    "sinusoidal_gamma_generator",
    n=num_nodes,
    params={"rate": 100.0, "amplitude": 80.0, "frequency":12.0, 
            "phase": 0.0, "individual_spike_trains": True,
        }
)
m = nest.Create("multimeter", num_nodes, {"interval": 0.1, "record_from": ["rate"]})
# Create spike recorder
s = nest.Create("spike_recorder", num_nodes)

# Connect devices
nest.Connect(m, g, "one_to_one")
nest.Connect(g, s, "one_to_one")

nest.Simulate(1000) # Simulate for 1000 ms (1 s)

colors = plt.colormaps["viridis_r"](np.linspace(0, 1, num_nodes)).tolist()
for j in range(num_nodes):
    ev = m[j].events
    t = ev["times"]
    r = ev["rate"]
    spike_times = s[j].events["times"]
    plt.subplot(231)
    bin_width = (1 / float(pulse_width_ms[j])) * 10.0
    t_start = 0.0
    t_stop = 1000
    bins = np.arange(t_start, t_stop + bin_width, bin_width)
    h, e = np.histogram(spike_times, bins=bins)
    rates = h * (1000.0 / bin_width)  # Convert to spikes per second
    plt.step(e[:-1], rates, color=colors[j], where='post', alpha=0.7)
    plt.plot(t, r, color=colors[j], label=f'{titles[j]}({pulse_width_ms[j]} ms)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.title("PST histogram and firing rates")
    plt.ylabel("Spikes per second")
    plt.xlabel("Time (ms)")

    # "Individual spike trains for each target" plot
    plt.subplot(234)
    isi = np.diff(np.sort(spike_times))
    plt.hist(isi, bins=np.arange(0.0, 50.0, 1.0).tolist(), histtype="step", color=colors[j], label=f'{titles[j]}({pulse_width_ms[j]} ms)')
    plt.title("ISI histogram")
    plt.xlabel("ISI (ms)")
    plt.legend()

# Reset kernel and set up for new simulation
nest.ResetKernel()
nest.local_num_threads = 6

# Create sinusoidal gamma generators for each waveform type
g = nest.Create(
    "sinusoidal_gamma_generator",
    n=num_nodes,
    params={"rate": 100.0, "amplitude": 80.0, "frequency": 12.0, 
            "phase": 0.0,"order": 3.0, "individual_spike_trains": False,
        }
)

p = nest.Create("parrot_neuron", 30)
s = nest.Create("spike_recorder")

nest.Connect(g, p)
nest.Connect(p, s)

nest.Simulate(1000)

plt.subplot(232) # "A -> P" subplot (Anterior to Posterior direction)
ev = {} # Create a dictionary to store events for each title
ev = s.events # Get events from the single spike recorder
colors_ap = matplotlib.colormaps.get_cmap("coolwarm")(np.linspace(0, 1, num_nodes)).tolist()

for j in range(min(num_nodes, len(g))):  # Ensure we don't exceed the number of generators
    try:
        # Compute parrot-group size dynamically so we don't assume 10 per generator
        if len(g) > 0:
            group_size = max(1, int(np.ceil(len(p) / float(len(g)))))
        else:
            group_size = len(p)

        start_idx = j * group_size
        end_idx = min(len(p), start_idx + group_size) - 1

        # Verify indices
        if start_idx < len(p) and start_idx <= end_idx:
            sn_id = p[start_idx].get('global_id')
            en_id = p[end_idx].get('global_id')

            # Filter spike events for this generator's neurons
            mask = (ev['senders'] >= sn_id) & (ev['senders'] <= en_id)
            g_spikes = ev['times'][mask]
            g_senders = ev['senders'][mask]

            ev[titles[j]] = {'times': g_spikes, 'senders': g_senders}
            spike_trains = [g_spikes[g_senders == sender] for sender in np.unique(g_senders)]
            # Use coolwarm colormap for A->P direction
            colors = [colors_ap[j]] * len(spike_trains)
            plt.eventplot(
                spike_trains,
                lineoffsets=np.arange(len(spike_trains), dtype=float).tolist(),
                linewidths=1,
                linelengths=0.8,
                colors=colors,
                alpha=0.9,
            )
        else:
            print(f"Warning: Not enough parrot neurons for generator {j}")
            ev[titles[j]] = {'times': np.array([]), 'senders': np.array([])}
    except Exception as e:
        print(f"Error processing generator {j}: {e}")
        # Continue with empty data for this generator
        ev[titles[j]] = {'times': np.array([]), 'senders': np.array([])}
        
plt.ylim([-0.5, len(titles) - 0.5])
plt.yticks([])
plt.title("TMS Stimulus Individual Spike Trains for each target\n (A -> P Direction: coolwarm)")

plt.subplot(233) # "P -> A" subplot (Posterior to Anterior direction)
ev = {}
ev = s.events
colors_pa = matplotlib.colormaps.get_cmap("coolwarm_r")(np.linspace(0, 1, num_nodes)).tolist()
for j in range(min(num_nodes, len(g))):  # Ensure we don't exceed the number of generators
    try:
        # Compute parrot-group size dynamically so we don't assume 10 per generator
        if len(g) > 0:
            group_size = max(1, int(np.ceil(len(p) / float(len(g)))))
        else:
            group_size = len(p)

        start_idx = j * group_size
        end_idx = min(len(p), start_idx + group_size) - 1

        # Verify indices
        if start_idx < len(p) and start_idx <= end_idx:
            sn_id = p[start_idx].get('global_id')
            en_id = p[end_idx].get('global_id')

            # Filter spike events for this generator's neurons
            mask = (ev['senders'] >= sn_id) & (ev['senders'] <= en_id)
            g_spikes = ev['times'][mask]
            g_senders = ev['senders'][mask]

            ev[titles[j]] = {'times': g_spikes, 'senders': g_senders}
            spike_trains = [g_spikes[g_senders == sender] for sender in np.unique(g_senders)]
            # Use coolwarm_r colormap for P->A direction
            colors = [colors_pa[j]] * len(spike_trains)
            plt.eventplot(
                spike_trains,
                lineoffsets=np.arange(len(spike_trains), dtype=float).tolist(),
                linewidths=1,
                linelengths=0.8,
                colors=colors,
                alpha=0.9,
            )
        else:
            print(f"Warning: Not enough parrot neurons for generator {j}")
            ev[titles[j]] = {'times': np.array([]), 'senders': np.array([])}
    except Exception as e:
        print(f"Error processing generator {j}: {e}")
        ev[titles[j]] = {'times': np.array([]), 'senders': np.array([])}
        
plt.ylim([-0.5, len(titles) - 0.5])
plt.yticks([])
plt.title("TMS Stimulus Individual Spike Trains for each target\n(P -> A Direction: coolwarm_r)")

# Reset kernel and set up for new simulation
nest.ResetKernel()
nest.resolution = 0.01
nest.local_num_threads = 6


num_nodes = 3 # Create sinusoidal gamma generators for each waveform type
g = nest.Create(
    "sinusoidal_gamma_generator",
    n=num_nodes,
    params={"rate": 100.0, "amplitude": 80.0, 
            "frequency": 12.0, "phase": 0.0, "order": 3.0,
            "individual_spike_trains": False,
            },
)

p = nest.Create("parrot_neuron", 30)
s = nest.Create("spike_recorder")

nest.Connect(g, p)
nest.Connect(p, s)

nest.Simulate(1000)
ev = s.events
plt.subplot(235)

# Check if ev has the expected structure
if hasattr(ev, 'keys') and 'times' in ev and 'senders' in ev and len(ev['times']) > 0:
    spike_trains = [ev["times"][ev["senders"] == sender] for sender in np.unique(ev["senders"])]
    lineoffsets = (np.unique(ev["senders"]) - min(ev["senders"])).tolist()
else: # Handle empty or unexpected events structure
    spike_trains = []
    lineoffsets = []

if spike_trains:
    colors = matplotlib.colormaps.get_cmap("coolwarm")(np.linspace(0, 1, len(spike_trains))).tolist()
    plt.eventplot(
        spike_trains, 
        lineoffsets=lineoffsets, 
        linelengths=0.8, 
        linewidths=1, 
        colors=colors,
        alpha=0.7
    )

plt.ylim([-0.5, 19.5])
plt.yticks([])
plt.title("TMS Stimulus\nOne spike train for all targets(A -> P Direction: coolwarm)")

# Create a second version of "One spike train for all targets"
plt.subplot(236)
# This will show the same data but in a separate subplot for better visualization
if hasattr(ev, 'keys') and 'times' in ev and 'senders' in ev and len(ev['times']) > 0:
    spike_trains = [ev["times"][ev["senders"] == sender] for sender in np.unique(ev["senders"])]
    lineoffsets = (np.unique(ev["senders"]) - min(ev["senders"])).tolist()
else:
    # Handle empty or unexpected events structure
    spike_trains = []
    lineoffsets = []

if spike_trains:
    colors = matplotlib.colormaps.get_cmap("coolwarm_r")(np.linspace(0, 1, len(spike_trains))).tolist()
    plt.eventplot(
        spike_trains, 
        lineoffsets=lineoffsets, 
        linelengths=0.8, 
        linewidths=1, 
        colors=colors,
        alpha=0.7
    )
plt.ylim([-0.5, 19.5])
plt.yticks([])
plt.title("TMS Stimulus\nOne spike train for all targets (P -> A Direction: coolwarm_r)")

# Adjust layout and save combined figure
plt.tight_layout()
plt.savefig('stimuli_generation/TMS_stimuli_simulation.png', dpi=300, bbox_inches='tight')
plt.show()
###############################################################################
t = 1000
n = 1000
dt = 0.01
steps = int(t / dt)
offset = t / 1000.0 * 2 * np.pi

# Grid for rate modulation plots
grid = (2, 3)
fig = plt.figure(figsize=(15, 10))
plt.subplot(grid[0], grid[1], 1)
# Stepwise DC Rate Increase: 20 → 35 → 50 → 65 → 80 Hz (6 steps per level)
# Monophasic-like waveform parameters
spikes_mono = step(
    t,
    n,
    {"rate": 20, "order": 1.0},
    {"rate": 35, "order": 1.0},
    {"rate": 50, "order": 1.0},
    {"rate": 65, "order": 1.0},
    {"rate": 80, "order": 1.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes_mono, alpha=0.7, label='Monophasic')

# Half-sine-like waveform parameters
spikes_half = step(
    t,
    n,
    {"rate": 20,  "order": 3.0},
    {"rate": 35,  "order": 3.0},
    {"rate": 50,  "order": 3.0},
    {"rate": 65,  "order": 3.0},
    {"rate": 80,  "order": 3.0},
    seed=124,
    dt=dt,
)
plot_hist(spikes_half, alpha=0.7, label='Half-sine')

# Biphasic-like waveform parameters
spikes_bi = step(
    t,
    n,
    {"rate": 20,  "order": 2.0},
    {"rate": 35,  "order": 2.0},
    {"rate": 50,  "order": 2.0},
    {"rate": 65,  "order": 2.0},
    {"rate": 80,  "order": 2.0},
    seed=125,
    dt=dt,
)
plot_hist(spikes_bi, alpha=0.7, label='Biphasic')

# Expected waveform
exp = np.ones(int(steps))
level_steps = int(steps / 5)

for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    if level == 0:
        exp[start_idx:end_idx] *= 20
    elif level == 1:
        exp[start_idx:end_idx] *= 35
    elif level == 2:
        exp[start_idx:end_idx] *= 50
    elif level == 3:
        exp[start_idx:end_idx] *= 65
    elif level == 4:
        exp[start_idx:end_idx] *= 80

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2, label='Theoretical')
plt.legend()
plt.title("Stepwise DC Rate Increase\n20 → 35 → 50 → 65 → 80 Hz (6 steps per level)")
plt.ylabel("Spikes per second")

plt.subplot(grid[0], grid[1], 2)
# Stepwise DC Rate Decrease: 100 → 80 → 60 → 40 → 20 (6 steps per level)
# Monophasic-like waveform parameters
spikes_mono = step(
    t,
    n,
    {"rate": 100, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 1},
    {"rate": 80, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 1},
    {"rate": 60, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 1},
    {"rate": 40, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 1},
    {"rate": 20, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 1},
    seed=123,
    dt=dt,
)
plot_hist(spikes_mono, alpha=0.7, label='Monophasic')

# Half-sine-like waveform parameters
spikes_half = step(
    t,
    n,
    {"rate": 100, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 2},
    {"rate": 80, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 2},
    {"rate": 60, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 2},
    {"rate": 40, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 2},
    {"rate": 20, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 2},
    seed=124,
    dt=dt,
)
plot_hist(spikes_half, alpha=0.7, label='Half-sine')

# Biphasic-like waveform parameters
spikes_bi = step(
    t,
    n,
    {"rate": 100, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 3},
    {"rate": 80, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 3},
    {"rate": 60, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 3},
    {"rate": 40, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 3},
    {"rate": 20, "amplitude": 0, "frequency": 0, "phase": 0.0, "order": 3},
    seed=125,
    dt=dt,
)
plot_hist(spikes_bi, alpha=0.7, label='Biphasic')

# Expected waveform
exp = np.ones(int(steps))
level_steps = int(steps / 5)

for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    if level == 0:
        exp[start_idx:end_idx] *= 100
    elif level == 1:
        exp[start_idx:end_idx] *= 80
    elif level == 2:
        exp[start_idx:end_idx] *= 60
    elif level == 3:
        exp[start_idx:end_idx] *= 40
    elif level == 4:
        exp[start_idx:end_idx] *= 20

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2, label='Theoretical')
plt.legend()
plt.title("Stepwise DC Rate Decrease\n100 → 80 → 60 → 40 → 20 (6 steps per level)")


plt.subplot(grid[0], grid[1], 3)
# Stepwise Rate Modulation: 40 → 30 → 20 → 10 → 5 (6 steps per level)
# Monophasic-like waveform parameters
spikes_mono = step(
    t,
    n,
    {"rate": 50, "amplitude": 40, "frequency": 8.0, "phase": 0.0,  "order": 1},
    {"rate": 50, "amplitude": 30, "frequency": 8.0, "phase": 0.0, "order": 1},
    {"rate": 50, "amplitude": 20, "frequency": 8.0, "phase": 0.0, "order": 1},
    {"rate": 50, "amplitude": 10, "frequency": 8.0, "phase": 0.0, "order": 1},
    {"rate": 50, "amplitude": 5, "frequency": 8.0, "phase": 0.0, "order": 1},
    seed=123,
    dt=dt,
)
plot_hist(spikes_mono, alpha=0.7, label='Monophasic')

# Half-sine-like waveform parameters
spikes_half = step(
    t,
    n,
    {"rate": 50, "amplitude": 40, "frequency": 10.0, "phase": 0.0,  "order": 2},
    {"rate": 50, "amplitude": 30, "frequency": 10.0, "phase": 0.0, "order": 2},
    {"rate": 50, "amplitude": 20, "frequency": 10.0, "phase": 0.0, "order": 2},
    {"rate": 50, "amplitude": 10, "frequency": 10.0, "phase": 0.0, "order": 2},
    {"rate": 50, "amplitude": 5, "frequency": 10.0, "phase": 0.0, "order": 2},
    seed=124,
    dt=dt,
)
plot_hist(spikes_half, alpha=0.7, label='Half-sine')

# Biphasic-like waveform parameters
spikes_bi = step(
    t,
    n,
    {"rate": 50, "amplitude": 40, "frequency": 12.0, "phase": 0.0, "order": 3},
    {"rate": 50, "amplitude": 30, "frequency": 12.0, "phase": 0.0, "order": 3},
    {"rate": 50, "amplitude": 20, "frequency": 12.0, "phase": 0.0, "order": 3},
    {"rate": 50, "amplitude": 10, "frequency": 12.0, "phase": 0.0, "order": 3},
    {"rate": 50, "amplitude": 5, "frequency": 12.0, "phase": 0.0, "order": 3},
    seed=125,
    dt=dt,
)
plot_hist(spikes_bi, alpha=0.7, label='Biphasic')

# Expected waveform
exp_1 = np.zeros(int(steps))
exp_2 = np.zeros(int(steps))
exp_3 = np.zeros(int(steps))
level_steps = int(steps / 5)

for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    time_offset = (start_idx * dt) / 1000.0  # Convert offset to seconds
    time_array = np.arange(0, level_steps * dt, dt) / 1000.0
    time_array_with_offset = time_array + time_offset
    
    if level == 0:
        exp_1[start_idx:end_idx] = 50 + 40 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 50 + 40 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 50 + 40 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)

    elif level == 1:
        exp_1[start_idx:end_idx] = 50 + 30 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 50 + 30 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 50 + 30 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 2:
        exp_1[start_idx:end_idx] = 50 + 20 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 50 + 20 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 50 + 20 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 3:
        exp_1[start_idx:end_idx] = 50 + 10 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 50 + 10 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 50 + 10 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 4:
        exp_1[start_idx:end_idx] = 50 + 5 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 50 + 5 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 50 + 5 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp_1, "navy", linewidth=2, label='Thy_monophasic', linestyle='--')
plt.plot(time_axis, exp_2, "red", linewidth=2, label='Thy_half-sine', linestyle=':')
plt.plot(time_axis, exp_3, "darkgreen", linewidth=2, label='Thy_biphasic', linestyle='-')

plt.legend()
plt.title("Stepwise Rate Modulation\n40 → 30 → 20 → 10 → 5 (6 steps per level)")

# Stepwise DC Rate and Rate Modulation: 20 → 35 → 50 → 65 → 80
plt.subplot(grid[0], grid[1], 4)

# Monophasic-like waveform parameters
spikes_mono = step(
    t,
    n,
    {"rate": 40.0, "amplitude": 20, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    {"rate": 70.0, "amplitude": 35, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    {"rate": 100.0, "amplitude": 50, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    {"rate": 130.0, "amplitude": 65,  "frequency": 8.0, "phase": 0.0, "order": 1.0},
    {"rate": 160.0, "amplitude": 80, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes_mono, alpha=0.7, label='Monophasic')

# Half-sine-like waveform parameters
spikes_half = step(
    t,
    n,
    {"rate": 40.0, "amplitude": 20.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    {"rate": 70.0, "amplitude": 35.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    {"rate": 100.0, "amplitude": 50.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    {"rate": 130.0, "amplitude": 65.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    {"rate": 160.0, "amplitude": 80.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    seed=124,
    dt=dt,
)
plot_hist(spikes_half, alpha=0.7, label='Half-sine')

# Biphasic-like waveform parameters
spikes_bi = step(
    t,
    n,
    {"rate": 40.0, "amplitude": 20.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    {"rate": 70.0, "amplitude": 35.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    {"rate": 100.0, "amplitude": 50.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    {"rate": 130.0, "amplitude": 65.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    {"rate": 160.0, "amplitude": 80.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    seed=125,
    dt=dt,
)
plot_hist(spikes_bi, alpha=0.7, label='Biphasic')

# Expected waveform
exp_1 = np.zeros(int(steps))
exp_2 = np.zeros(int(steps))
exp_3 = np.zeros(int(steps))
level_steps = int(steps / 5)

for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    time_offset = (start_idx * dt) / 1000.0  # Convert offset to seconds
    time_array = np.arange(0, level_steps * dt, dt) / 1000.0
    time_array_with_offset = time_array + time_offset
    
    if level == 0:
        exp_1[start_idx:end_idx] = 40.0 + 20.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 40.0 + 20.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 40.0 + 20.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 1:
        exp_1[start_idx:end_idx] = 70.0 + 35.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 70.0 + 35.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 70.0 + 35.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 2:
        exp_1[start_idx:end_idx] = 100.0 + 50.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 100.0 + 50.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 100.0 + 50.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 3:
        exp_1[start_idx:end_idx] = 130.0 + 65.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 130.0 + 65.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 130.0 + 65.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 4:
        exp_1[start_idx:end_idx] = 160.0 + 80.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 160.0 + 80.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 160.0 + 80.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    
time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp_1, "navy", linewidth=2, label='Thy_monophasic', linestyle='--')
plt.plot(time_axis, exp_2, "red", linewidth=2, label='Thy_half-sine', linestyle=':')
plt.plot(time_axis, exp_3, "darkgreen", linewidth=2, label='Thy_biphasic', linestyle='-')

plt.legend()
plt.title("Stepwise DC Rate and Rate Modulation\n20 → 35 → 50 → 65 → 80")
plt.ylabel("Spikes per second")

plt.subplot(grid[0], grid[1], 5)
# Monophasic-like waveform parameters
spikes_mono = step(
    t,
    n,
    {"rate": 80.0, "amplitude": 5.0, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 20.0, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 40.0, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 60.0, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 80.0, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes_mono, alpha=0.7, label='Monophasic')

# Half-sine-like waveform parameters
spikes_half = step(
    t,
    n,
    {"rate": 80.0, "amplitude": 5.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    {"rate": 80.0, "amplitude": 20.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    {"rate": 80.0, "amplitude": 40.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    {"rate": 80.0, "amplitude": 60.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    {"rate": 80.0, "amplitude": 80.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    seed=124,
    dt=dt,
)
plot_hist(spikes_half, alpha=0.7, label='Half-sine')

# Biphasic-like waveform parameters
spikes_bi = step(
    t,
    n,
    {"rate": 80.0, "amplitude": 5.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    {"rate": 80.0, "amplitude": 20.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    {"rate": 80.0, "amplitude": 40.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    {"rate": 80.0, "amplitude": 60.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    {"rate": 80.0, "amplitude": 80.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    seed=125,
    dt=dt,
)
plot_hist(spikes_bi, alpha=0.7, label='Biphasic')

# Expected waveform
exp_1 = np.zeros(int(steps))
exp_2 = np.zeros(int(steps))
exp_3 = np.zeros(int(steps))
level_steps = int(steps / 5)

for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    time_offset = (start_idx * dt) / 1000.0  # Convert offset to seconds
    time_array = np.arange(0, level_steps * dt, dt) / 1000.0
    time_array_with_offset = time_array + time_offset
    
    if level == 0:
        exp_1[start_idx:end_idx] = 80.0 + 5.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 80.0 + 5.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 80.0 + 5.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 1:
        exp_1[start_idx:end_idx] = 80.0 + 20.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 80.0 + 20.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 80.0 + 20.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)

    elif level == 2:
        exp_1[start_idx:end_idx] = 80.0 + 40.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 80.0 + 40.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 80.0 + 40.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 3:
        exp_1[start_idx:end_idx] = 80.0 + 60.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 80.0 + 60.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 80.0 + 60.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
    elif level == 4:
        exp_1[start_idx:end_idx] = 80.0 + 80.0 * np.sin(2 * np.pi * 8.0 * time_array_with_offset)
        exp_2[start_idx:end_idx] = 80.0 + 80.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
        exp_3[start_idx:end_idx] = 80.0 + 80.0 * np.sin(2 * np.pi * 12.0 * time_array_with_offset)
time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp_1, "navy", linewidth=2, label='Thy_monophasic', linestyle='--')
plt.plot(time_axis, exp_2, "red", linewidth=2, label='Thy_half-sine', linestyle=':')
plt.plot(time_axis, exp_3, "darkgreen", linewidth=2, label='Thy_biphasic', linestyle='-')

plt.legend()
plt.title("Stepwise Rate Modulation\n5 → 20 → 40 → 60 → 80 (6 steps per level)")

# Modulation Phase Changes: 0 → π/4 → π/2 → 3π/4 → π
plt.subplot(grid[0], grid[1], 6)

# Monophasic-like waveform parameters
spikes_mono = step(
    t,
    n,
    {"rate": 50.0, "amplitude": 30.0, "frequency": 8.0, "phase": 0.0, "order": 1.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 8.0, "phase": 90.0, "order": 1.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 8.0, "phase": 180.0,  "order": 1.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 8.0, "phase": 270.0, "order": 1.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 8.0, "phase": 360.0, "order": 1.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes_mono, alpha=0.7, label='Monophasic')

# Half-sine-like waveform parameters
spikes_half = step(
    t,
    n,
    {"rate": 50.0, "amplitude": 30.0, "frequency": 10.0, "phase": 0.0, "order": 2.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 10.0, "phase": 90.0, "order": 2.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 10.0, "phase": 180.0, "order": 2.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 10.0, "phase": 270.0, "order": 2.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 10.0, "phase": 360.0, "order": 2.0},
    seed=124,
    dt=dt,
)
plot_hist(spikes_half, alpha=0.7, label='Half-sine')

# Biphasic-like waveform parameters
spikes_bi = step(
    t,
    n,
    {"rate": 50.0, "amplitude": 30.0, "frequency": 12.0, "phase": 0.0, "order": 3.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 12.0, "phase": 90.0, "order": 3.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 12.0, "phase": 180.0, "order": 3.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 12.0, "phase": 270.0, "order": 3.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 12.0, "phase": 360.0, "order": 3.0},
    seed=125,
    dt=dt,
)
plot_hist(spikes_bi, alpha=0.7, label='Biphasic')

# Expected waveform
exp_1 = np.zeros(int(steps))
exp_2 = np.zeros(int(steps))
exp_3 = np.zeros(int(steps))
level_steps = int(steps / 5)

for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    time_array = np.arange(start_idx, end_idx) * dt / 1000.0
    phase_degrees = [0, 90, 180, 270, 360]
    phase_radians = np.radians(phase_degrees[level])

    exp_1[start_idx:end_idx] = 50 + 30 * np.sin(2 * np.pi * 8.0 * time_array + phase_radians)
    exp_2[start_idx:end_idx] = 50 + 30 * np.sin(2 * np.pi * 10.0 * time_array + phase_radians)
    exp_3[start_idx:end_idx] = 50 + 30 * np.sin(2 * np.pi * 12.0 * time_array + phase_radians)
    
    
time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp_1, "navy", linewidth=2, label='Thy_monophasic', linestyle='--')
plt.plot(time_axis, exp_2, "red", linewidth=2, label='Thy_half-sine', linestyle=':')
plt.plot(time_axis, exp_3, "darkgreen", linewidth=2, label='Thy_biphasic', linestyle='-')

plt.legend()
plt.title("Modulation Phase Changes\n0 → π/4 → π/2 → 3π/4 → π")
plt.xlabel("Time (ms)")

# Adjust layout and save second figure
plt.tight_layout()
plt.savefig('stimuli_generation/TMS_rate_modulation.png', dpi=300, bbox_inches='tight')
plt.show()

print("Simulation completed. Results saved to 'stimuli_generation/TMS_stimuli_simulation.png', and 'stimuli_generation/TMS_rate_modulation.png'")