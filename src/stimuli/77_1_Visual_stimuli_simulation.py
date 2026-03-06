#!/usr/bin/env python3
"""
Sinusoidal Visual Stimuli Simulation with Specific Parameters
-----------------------------------------------

This script simulates sinusoidal visual stimuli with the following parameters:
- Pulse width: 0.2-1 ms
- Frequency: 5-100 Hz
- Pulse-train duration: 0.2-1 s
- Amplitude: 0.2-5 mA

And examines variations in transfer curves with:
- 365 nm wavelength (P365nm)
- 10 V bias voltage (E10V)
- Single pulse of 1 s duration

The implementation uses NEST simulator with sinusoidal_gamma_generator
to generate and record the stimuli.

Based on concepts from:
Jieun Kim, Jung Wook Lim, Han Seul Kim,
"Synaptic devices for simulating brain processes in visual-information perception to persisting memory through attention mechanisms,"
Materials Today Advances,
Volume 20,
2023,
100421,
ISSN 2590-0498,
https://doi.org/10.1016/j.mtadv.2023.100421.
"""

import matplotlib.pyplot as plt
import matplotlib
import nest
import numpy as np

def step(t, n, initial, post1, post2, post3, post4, seed=1, dt=0.01):
    """Helper function to simulate generators with parameter changes."""
    nest.ResetKernel()
    nest.local_num_threads = 6
    nest.resolution = dt
    nest.rng_seed = seed

    g = nest.Create("sinusoidal_gamma_generator", n, params=initial)
    sr = nest.Create("spike_recorder")
    nest.Connect(g, sr)
    
    # For 25-step simulation: divide time into 25 parts but apply the same parameter
    # for every 5 consecutive steps to create 5 distinct levels
    # Ensure time_per_step is a multiple of dt
    time_per_step = int(t / 25 / dt) * dt
    
    # First level (5 steps)
    for _ in range(5):
        nest.Simulate(time_per_step)
    
    # Second level (5 steps)
    g.set(post1)
    for _ in range(5):
        nest.Simulate(time_per_step)
    
    # Third level (5 steps)
    g.set(post2)
    for _ in range(5):
        nest.Simulate(time_per_step)
    
    # Fourth level (5 steps)
    g.set(post3)
    for _ in range(5):
        nest.Simulate(time_per_step)
    
    # Fifth level (5 steps)
    g.set(post4)
    for _ in range(5):
        nest.Simulate(time_per_step)

    return sr.events

def step_25(t, n, initial_v, final_v, seed=1, dt=1.0):
    """Helper function to simulate 30 consecutive step pulses."""
    nest.ResetKernel()
    nest.local_num_threads = 6
    nest.resolution = dt
    nest.rng_seed = seed

    g = nest.Create("sinusoidal_gamma_generator", n)
    sr = nest.Create("spike_recorder")
    nest.Connect(g, sr)
    
    # Divide time into 25 equal parts, ensuring each is a multiple of dt
    time_per_step = int(t / 25 / dt) * dt
    spike_data = []
    
    for i in range(25):
        
        voltage = initial_v + (final_v - initial_v) * (i / 24)  # Linearly change voltage from initial to final
        # For sinusoidal_gamma_generator, amplitude must be <= rate
        # We'll set amplitude to 0 to make it a constant rate
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


def plot_hist(spikes):
    """Plot histogram of emitted spikes."""
    plt.hist(
        spikes["times"], 
        bins=np.arange(0.0, max(spikes["times"]) + 1.5, 1.0).tolist(), 
        histtype="step",
    )

# Reset the NEST kernel for a clean start
nest.ResetKernel()
nest.resolution = 0.01
print("Simulating visual stimulus with 365 nm wavelength and 10 V bias voltage...")

###############################################################################
# Set default parameters for sinusoidal gamma generator
plt.figure(figsize=(10, 8))
num_nodes = 4
g = nest.Create(
    "sinusoidal_gamma_generator",
    n=num_nodes,
    params={"rate": 60.0, "amplitude": 20.0, 
            "frequency": 10.0, "phase": 0.0,
            "individual_spike_trains": False,
        },
)

# Create multimeter to record the instantaneous rate
m = nest.Create("multimeter", num_nodes, {"interval": 0.1, "record_from": ["rate"]})
s = nest.Create("spike_recorder", num_nodes)

# Connect devices
nest.Connect(m, g, "one_to_one")
nest.Connect(g, s, "one_to_one")

nest.Simulate(1000) # Simulate for 1000 ms (1 s)

# Generate colors using magma colormap
colors = matplotlib.colormaps.get_cmap("viridis_r")(np.linspace(0, 1, num_nodes)).tolist()
for j in range(num_nodes):
    ev = m[j].events
    t = ev["times"]
    r = ev["rate"]
    spike_times = s[j].events["times"]
    plt.subplot(221)
    # Create PST histogram with the same time base as the firing rate
    bin_width = 25.0
    t_start = 0.0
    t_stop = 1000 
    bins = np.arange(t_start, t_stop + bin_width, bin_width)
    h, e = np.histogram(spike_times, bins=bins)
    plt.plot(t, r, color=colors[j])
    rates = h * (1000.0 / bin_width)  # Convert to spikes per second
    plt.step(e[:-1], rates, color=colors[j], where='post', linestyle='--')
    
    plt.legend()
    plt.title("PST histogram and firing rates")
    plt.ylabel("Spikes per second")

    # ISI histogram (subplot 2)
    plt.subplot(223)
    isi = np.diff(np.sort(spike_times))
    plt.hist(isi, bins=np.arange(0.0, 50.0, 2.0).tolist(), histtype="step", color=colors[j])
    plt.title("ISI histogram")

# Reset kernel for next simulation
nest.ResetKernel()
nest.local_num_threads = 6

# Simulate visual stimulus with same parameters but individual spike trains
g = nest.Create(
    "sinusoidal_gamma_generator",
    params={"rate": 60.0, "amplitude": 22.0, "frequency": 10.0,"phase": 0.0, "individual_spike_trains": True,
        }
)

p = nest.Create("parrot_neuron", 20)
s = nest.Create("spike_recorder")

nest.Connect(g, p)
nest.Connect(p, s)

nest.Simulate(1000)
ev = s.events

# Plot spike raster for P365nm + E10V stimulus with individual spike trains (subplot 3)
plt.subplot(222)
spike_trains = [ev["times"][ev["senders"] == sender] for sender in np.unique(ev["senders"])]
colors = matplotlib.colormaps.get_cmap("coolwarm")(np.linspace(0, 1, len(spike_trains))).tolist()
plt.eventplot(
    spike_trains, 
    lineoffsets=np.unique(ev["senders"]) - min(ev["senders"]), 
    linelengths=0.8, 
    linewidths=1.5, 
    colors=colors,
    alpha=0.7
)
plt.ylim([-0.5, 19.5])
plt.yticks([])
plt.title("Visual Stimulus (365 nm, 10 V)\nIndividual spike trains for target")

# Reset kernel for next simulation
nest.ResetKernel()
nest.local_num_threads = 6

# Simulate visual stimulus with same parameters but one spike train for all targets
g = nest.Create(
    "sinusoidal_gamma_generator",
    params={"rate": 60.0, "amplitude": 18.0, "frequency": 10.0, "phase": 0.0, "individual_spike_trains": False,
        },
)

p = nest.Create("parrot_neuron", 20)
s = nest.Create("spike_recorder")

nest.Connect(g, p)
nest.Connect(p, s)

nest.Simulate(1000)
ev = s.events

# Plot spike raster for one spike train for all targets (subplot 4)
plt.subplot(224)
spike_trains = [ev["times"][ev["senders"] == sender] for sender in np.unique(ev["senders"])]
colors = matplotlib.colormaps.get_cmap("coolwarm")(np.linspace(0, 1, len(spike_trains))).tolist()
plt.eventplot(
    spike_trains, 
    lineoffsets=np.unique(ev["senders"]) - min(ev["senders"]), 
    linelengths=0.8, 
    linewidths=1.5, 
    colors=colors,
    alpha=0.7
)
plt.ylim([-0.5, 19.5])
plt.yticks([])
plt.title("Visual Stimulus (365 nm, 10 V)\nOne spike train for all targets")

# Adjust layout and save first figure
plt.tight_layout()
plt.savefig('stimuli_generation/visual_stimuli_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
# Create the rate modulation plot (2x3 layout)
plt.figure(figsize=(15, 10))

t = 1000
n = 1000
dt = 0.01
steps = int(t / dt)
offset = t / 1000.0 * 2 * np.pi

# Grid for rate modulation plots
grid = (2, 3)

# DC rate steps: 20 → 35 → 50 → 65 → 80 Hz (subplot 1)
plt.subplot(grid[0], grid[1], 1)
spikes = step(
    t,
    n,
    {"rate": 20.0},
    {"rate": 35.0},
    {"rate": 50.0},
    {"rate": 65.0},
    {"rate": 80.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.ones(int(steps))
level_steps = int(steps / 5)
for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    if level == 0:
        exp[start_idx:end_idx] *= 20.0
    elif level == 1:
        exp[start_idx:end_idx] *= 35.0
    elif level == 2:
        exp[start_idx:end_idx] *= 50.0
    elif level == 3:
        exp[start_idx:end_idx] *= 65.0
    elif level == 4:
        exp[start_idx:end_idx] *= 80.0

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("DC Rate \n20 → 35 → 50 → 65 → 80")
plt.ylabel("Spikes per second")
plt.xlabel("Time [ms]")

# DC rate steps: 100 → 80 → 60 → 40 → 20 Hz (subplot 2)
plt.subplot(grid[0], grid[1], 2)
spikes = step(
    t,
    n,
    {"rate": 100.0},
    {"rate": 80.0},
    {"rate": 60.0},
    {"rate": 40.0},
    {"rate": 20.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.ones(int(steps))
level_steps = int(steps / 5)
for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    if level == 0:
        exp[start_idx:end_idx] *= 100.0
    elif level == 1:
        exp[start_idx:end_idx] *= 80.0
    elif level == 2:
        exp[start_idx:end_idx] *= 60.0
    elif level == 3:
        exp[start_idx:end_idx] *= 40.0
    elif level == 4:
        exp[start_idx:end_idx] *= 20.0

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("DC Rate \n100 → 80 → 60 → 40 → 20")
plt.xlabel("Time [ms]")

# Amplitude steps: 40 → 30 → 20 → 10 → 5 Hz (subplot 3)
plt.subplot(grid[0], grid[1], 3)
spikes = step(
    t,
    n,
    {"rate": 50.0, "amplitude": 40.0, "frequency": 10.0, "phase": 0.0, "order": 1.0},
    {"rate": 50.0, "amplitude": 30.0, "frequency": 10.0, "phase": 0.0, "order": 1.0},
    {"rate": 50.0, "amplitude": 20.0, "frequency": 10.0, "phase": 0.0, "order": 1.0},
    {"rate": 50.0, "amplitude": 10.0, "frequency": 10.0, "phase": 0.0, "order": 1.0},
    {"rate": 50.0, "amplitude": 5.0, "frequency": 10.0, "phase": 0.0, "order": 1.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.zeros(int(steps))
level_steps = int(steps / 5)

for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    time_offset = (start_idx * dt) / 1000.0  # Convert offset to seconds
    time_array = np.arange(0, level_steps * dt, dt) / 1000.0  # Convert to seconds
    time_array_with_offset = time_array + time_offset
    
    if level == 0:
        exp[start_idx:end_idx] = 50.0 + 40.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
    elif level == 1:    
        exp[start_idx:end_idx] = 50.0 + 30.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
    elif level == 2:
        exp[start_idx:end_idx] = 50.0 + 20.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
    elif level == 3:
        exp[start_idx:end_idx] = 50.0 + 10.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
    elif level == 4:
        exp[start_idx:end_idx] = 50.0 + 5.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("Rate Modulation\n40 → 30 → 20 → 10 → 5")
plt.xlabel("Time [ms]")

# DC rate and amplitude steps: 15 → 25 → 35 → 40 → 50 Hz (subplot 4)
plt.subplot(grid[0], grid[1], 4)
spikes = step(
    t,
    n,
    {"rate": 30.0, "amplitude": 15.0, "frequency": 20.0, "phase": 0.0, "order": 1.0},
    {"rate": 50.0, "amplitude": 25.0, "frequency": 20.0, "phase": 0.0, "order": 1.0},
    {"rate": 70.0, "amplitude": 35.0, "frequency": 20.0, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 40.0, "frequency": 20.0, "phase": 0.0, "order": 1.0},
    {"rate": 100.0, "amplitude": 50.0, "frequency": 20.0, "phase": 0.0, "order": 1.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.zeros(int(steps))

level_steps = int(steps / 5)
for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    time_offset = (start_idx * dt) / 1000.0  # Convert offset to seconds
    time_array = np.arange(0, level_steps * dt, dt) / 1000.0  # Convert to seconds
    time_array_with_offset = time_array + time_offset
    
    if level == 0:
        exp[start_idx:end_idx] = 30.0 + 15.0 * np.sin(2 * np.pi * 20.0 * time_array_with_offset)
    elif level == 1:    
        exp[start_idx:end_idx] = 50.0 + 25.0 * np.sin(2 * np.pi * 20.0 * time_array_with_offset)
    elif level == 2:
        exp[start_idx:end_idx] = 70.0 + 35.0 * np.sin(2 * np.pi * 20.0 * time_array_with_offset)
    elif level == 3:    
        exp[start_idx:end_idx] = 80.0 + 40.0 * np.sin(2 * np.pi * 20.0 * time_array_with_offset)
    elif level == 4:
        exp[start_idx:end_idx] = 100.0 + 50.0 * np.sin(2 * np.pi * 20.0 * time_array_with_offset)

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("DC Rate & Rate Modulation\n15 → 25 → 35 → 40 → 50")
plt.ylabel("Spikes per second")
plt.xlabel("Time [ms]")

# Modulation introduction steps: 5 → 10 → 20 → 40 → 60 Hz (subplot 5)
plt.subplot(grid[0], grid[1], 5)
spikes = step(
    t,
    n,
    {"rate": 80.0, "amplitude": 5.0,"frequency": 10.0,"phase": 0.0,"order": 1.0},
    {"rate": 80.0,"amplitude": 10.0,"frequency": 10.0,"phase": 0.0,"order": 1.0},
    {"rate": 80.0,"amplitude": 20.0,"frequency": 10.0,"phase": 0.0,"order": 1.0},
    {"rate": 80.0,"amplitude": 40.0,"frequency": 10.0,"phase": 0.0, "order": 1.0},
    {"rate": 80.0,"amplitude": 60.0,"frequency": 10.0,"phase": 0.0, "order": 1.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.zeros(int(steps))
level_steps = int(steps / 5)

# Ensure amplitude <= rate for all steps
for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    time_offset = (start_idx * dt) / 1000.0  # Convert offset to seconds
    time_array = np.arange(0, level_steps * dt, dt) / 1000.0  # Convert to seconds
    time_array_with_offset = time_array + time_offset
    
    if level == 0:
        exp[start_idx:end_idx] = 80.0 + 5.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
    elif level == 1:
        exp[start_idx:end_idx] = 80.0 + 10.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
    elif level == 2:
        exp[start_idx:end_idx] = 80.0 + 20.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
    elif level == 3:
        exp[start_idx:end_idx] = 80.0 + 40.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
    elif level == 4:
        exp[start_idx:end_idx] = 80.0 + 60.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("Rate Modulation \n5 → 10 → 20 → 400 → 60")
plt.xlabel("Time [ms]")

# Phase progression steps: 0 → π/4 → π/2 → 3π/4 → π (subplot 6)
plt.subplot(grid[0], grid[1], 6)
spikes = step(
    t,
    n,
    {"rate": 20.0, "amplitude": 15.0, "frequency": 15.0, "phase": 0.0, "order": 6.0},
    {"rate": 20.0, "amplitude": 15.0, "frequency": 15.0, "phase": 90.0, "order": 6.0},
    {"rate": 20.0, "amplitude": 15.0, "frequency": 15.0, "phase": 180.0, "order": 6.0},
    {"rate": 20.0, "amplitude": 15.0, "frequency": 15.0, "phase": 270.0, "order": 6.0},
    {"rate": 20.0, "amplitude": 15.0, "frequency": 15.0, "phase": 360.0, "order": 6.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.zeros(int(steps))

level_steps = int(steps / 5)
for level in range(5):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    time_array = np.arange(start_idx, end_idx) * dt / 1000.0  # Convert to seconds
    phase_degrees = [0, 90, 180, 270, 360]
    phase_radians = np.radians(phase_degrees[level])
    exp[start_idx:end_idx] = 20 + 5.0 * np.sin(2 * np.pi * 15.0 * time_array + phase_radians)
time_axis = np.arange(len(exp)) * dt    
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("Stepwise Phase Progression\n0 → π/4 → π/2 → 3π/4 → π")
plt.xlabel("Time (ms)")
# Adjust layout and save second figure
plt.tight_layout()
plt.savefig('stimuli_generation/visual_rate_modulation.png', dpi=300, bbox_inches='tight')
plt.show()

print("Simulation completed. Results saved to 'stimuli_generation/visual_stimuli_simulation.png' and 'stimuli_generation/visual_rate_modulation.png'")