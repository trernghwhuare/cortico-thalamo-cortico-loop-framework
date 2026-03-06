#!/usr/bin/env python3
"""
Sinusoidal pain Stimuli Simulation with Specific Parameters
-----------------------------------------------

This script simulates sinusoidal pain stimuli with the following parameters:
- 30 consecutive step pulses 
- Positive pulses: 4.0V to 5.0V (linear increase)
- Negative pulses: -3.0V to -4.5V (linear decrease)
- Pulse-train duration: 1000 seconds
- Amplitude: Variable based on pulse type
- Frequency: 0.5-2.0 Hz for modulation studies

And examines variations in transfer curves with:
- 365 nm wavelength (P365nm)
- 10 V bias voltage (E10V)
- 30 consecutive pulses with stepwise changes

The implementation uses NEST simulator with sinusoidal_gamma_generator
to generate and record the stimuli.

Based on the research:
Hao Chen, Zhihao Shen, Wen-Tao Guo, Yan-Ping Jiang, Wenhua Li, Dan Zhang, 
Zhenhua Tang, Qi-Jun Sun, Xin-Gui Tang, "Artificial synaptic simulating 
pain-perceptual nociceptor and brain-inspired computing based on 
Au/Bi3.2La0.8Ti3O12/ITO memristor," Journal of Materiomics, Volume 10, 
Issue 6, 2024, Pages 1308-1316, ISSN 2352-8478,
https://doi.org/10.1016/j.jmat.2024.03.011.
(https://www.sciencedirect.com/science/article/pii/S2352847824000716)
"""

import matplotlib.pyplot as plt
import matplotlib
import nest
import numpy as np



def step(t, n, initial, post1, post2, post3, post4, post5, seed=1, dt=0.01):
    """Helper function to simulate generators with parameter changes."""
    nest.ResetKernel()
    nest.local_num_threads = 6
    nest.resolution = dt
    nest.rng_seed = seed

    g = nest.Create("sinusoidal_gamma_generator", n, params=initial)
    sr = nest.Create("spike_recorder")
    nest.Connect(g, sr)
    
    # For 30-step simulation: divide time into 30 parts but apply the same parameter
    # for every 5 consecutive steps to create 6 distinct levels
    # Ensure time_per_step is a multiple of dt
    time_per_step = int(t / 30 / dt) * dt
    
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

    # Sixth level (5 steps)
    g.set(post5)
    for _ in range(5):
        nest.Simulate(time_per_step)
    return sr.events


def step_30(t, n, initial_v, final_v, seed=1, dt=1.0):
    """Helper function to simulate 30 consecutive step pulses."""
    nest.ResetKernel()
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
        
        voltage = initial_v + (final_v - initial_v) * (i / 29)  # Linearly change voltage from initial to final
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
    plt.hist(spikes["times"], bins=np.arange(0.0, max(spikes["times"]) + 1.5, 1.0).tolist(), histtype="step")


# Reset the NEST kernel for a clean start
nest.ResetKernel()
# nest.resolution = params['simulation']['resolution']  # Set resolution
nest.resolution = 0.01
print("Simulating pain stimulus with 365 nm wavelength and 10 V bias voltage...")

###############################################################################
# Set default parameters for sinusoidal gamma generator
plt.figure(figsize=(10, 8))

# Simulate pain stimulus with both P365nm wavelength and E10V bias voltage properties
num_nodes = 4
g = nest.Create(
    "sinusoidal_gamma_generator",
    n=num_nodes,
    params={"rate": 80.0, "amplitude": 20.0, "frequency": 1.0, 
            "phase": 0.0, "individual_spike_trains": False,
        }
)

m = nest.Create("multimeter", num_nodes, {"interval": 0.1, "record_from": ["rate"]})
s = nest.Create("spike_recorder", num_nodes)

nest.Connect(m, g, "one_to_one") # Connect devices
nest.Connect(g, s, "one_to_one")
print(m.get())
nest.Simulate(1000)  # Simulate for 1000 ms (1 s)

# Generate colors using magma colormap
colors = matplotlib.colormaps.get_cmap('viridis')(np.linspace(0, 1, num_nodes)).tolist()
for j in range(num_nodes):
    ev = m[j].events
    t = ev["times"]
    r = ev["rate"]

    spike_times = s[j].events["times"]
    plt.subplot(221)
    # Create PST histogram with the same time base as the firing rate
    bin_width = 50.0
    t_start = 0.0
    # t_stop = params['simulation']['duration'] 
    t_stop = 1000.0
    bins = np.arange(t_start, t_stop + bin_width, bin_width)
    h, e = np.histogram(spike_times, bins=bins)
    plt.plot(t, r, color=colors[j])
    # Convert histogram counts to firing rate (spikes per second)
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

# Simulate pain stimulus with same parameters but individual spike trains
g = nest.Create(
    "sinusoidal_gamma_generator",
    params={"rate": 80.0, "amplitude": 20.0, "frequency": 1.0, "phase": 0.0, "individual_spike_trains": True},
)
p = nest.Create("parrot_neuron", 20)
# p = nest.Create("parrot_neuron", params['parrot_neurons'])
s = nest.Create("spike_recorder")

# nest.Connect(g, p)
# nest.Connect(p, s)

nest.Connect(g, p, "all_to_all")
nest.Connect(p, s, "all_to_all")

# nest.Simulate(params['simulation']['duration'])
nest.Simulate(1000)
ev = s.events

# Plot spike raster for P365nm + E10V stimulus with individual spike trains (subplot 3)
plt.subplot(222)
# Plot spike raster with individual colors for each node
if hasattr(ev, 'keys') and 'times' in ev and 'senders' in ev and len(ev['times']) > 0:
    spike_trains = [ev["times"][ev["senders"] == sender] for sender in np.unique(ev["senders"])]
    lineoffsets = (np.unique(ev["senders"]) - min(ev["senders"])).tolist()
else:
    # Handle empty or unexpected events structure
    spike_trains = []
    lineoffsets = []

if spike_trains:
    colors = matplotlib.colormaps.get_cmap("coolwarm")(np.linspace(0, 1, len(spike_trains))).tolist()
    plt.eventplot(
        spike_trains, 
        lineoffsets=lineoffsets,
        linelengths=0.8, 
        linewidths=1.5, 
        colors=colors,
        alpha=0.7
    )
plt.ylim([-0.5, 19.5])
plt.yticks([])
plt.title("pain Stimulus (365 nm, 10 V)\nIndividual spike trains for target")


# Reset kernel for next simulation
nest.ResetKernel()
nest.local_num_threads = 6

# Simulate pain stimulus with same parameters but one spike train for all targets
g = nest.Create(
    "sinusoidal_gamma_generator",
    params={"rate": 80.0, "amplitude": 18.0, "frequency": 1.0, "phase": 0.0, "individual_spike_trains": False}
    # params=params['sinusoidal_gamma_generator']['one_spike_train_config'],
)
p = nest.Create("parrot_neuron", 20)
# p = nest.Create("parrot_neuron", params['parrot_neurons'])
s = nest.Create("spike_recorder")

nest.Connect(g, p, "all_to_all")
nest.Connect(p, s, "all_to_all")
# nest.Connect(g, p)
# nest.Connect(p, s)

# nest.Simulate(params['simulation']['duration'])
nest.Simulate(1000)
ev = s.events

# Plot spike raster for one spike train for all targets (subplot 4)
plt.subplot(224)
if hasattr(ev, 'keys') and 'times' in ev and 'senders' in ev and len(ev['times']) > 0:
    spike_trains = [ev["times"][ev["senders"] == sender] for sender in np.unique(ev["senders"])]
    lineoffsets = (np.unique(ev["senders"]) - min(ev["senders"])).tolist()
else:
    # Handle empty or unexpected events structure
    spike_trains = []
    lineoffsets = []

if spike_trains:
    colors = matplotlib.colormaps.get_cmap("coolwarm")(np.linspace(0, 1, len(spike_trains))).tolist()
    plt.eventplot(
        spike_trains, 
        lineoffsets=lineoffsets,
        linelengths=0.8, 
        linewidths=1.5, 
        colors=colors,
        alpha=0.7
    )
plt.ylim([-0.5, 19.5])
plt.yticks([])
plt.title("pain Stimulus (365 nm, 10 V)\nOne spike train for all targets")

# Adjust layout and save first figure
plt.tight_layout()
plt.savefig('stimuli_generation/pain_stimuli_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
# Now create the rate modulation plot (2x3 layout)
plt.figure(figsize=(15, 10))

# t = params['rate_modulation']['duration']
t = 1000
n = 1000
# dt = params['rate_modulation']['dt']
dt = 0.01
steps = int(t / dt)
offset = t / 1000.0 * 2 * np.pi

# Grid for rate modulation plots
grid = (2, 3)

# DC rate steps: 5 → 20 → 35 → 50 → 65 → 80 Hz (subplot 1)
plt.subplot(grid[0], grid[1], 1)
spikes = step(
    t,
    n,
    {"rate": 5.0},
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

# For 30 steps with 5 steps per level
level_steps = int(steps / 6)  # 5 steps per level for 30 total steps

for level in range(6):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    
    if level == 0:
        exp[start_idx:end_idx] *= 5.0
    elif level == 1:
        exp[start_idx:end_idx] *= 20.0
    elif level == 2:
        exp[start_idx:end_idx] *= 35.0
    elif level == 3:
        exp[start_idx:end_idx] *= 50.0
    elif level == 4:
        exp[start_idx:end_idx] *= 65.0
    elif level == 5:
        exp[start_idx:end_idx] *= 80.0

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("Stepwise DC Rate Increase\n5 → 20 → 35 → 50 → 65 → 80 Hz (5 steps, 6 levels)")
plt.ylabel("Spikes per second")

# DC rate steps: 100 → 80 → 60 → 40 → 20 → 5 Hz (subplot 2)
plt.subplot(grid[0], grid[1], 2)
spikes = step(
    t,
    n,
    {"order": 6.0, "rate": 100.0, "amplitude": 0.0, "frequency": 0.0, "phase": 0.0},
    {"order": 6.0, "rate": 80.0, "amplitude": 0.0, "frequency": 0.0, "phase": 0.0},
    {"order": 6.0, "rate": 60.0, "amplitude": 0.0, "frequency": 0.0, "phase": 0.0},
    {"order": 6.0, "rate": 40.0, "amplitude": 0.0, "frequency": 0.0, "phase": 0.0},
    {"order": 6.0, "rate": 20.0, "amplitude": 0.0, "frequency": 0.0, "phase": 0.0},
    {"order": 6.0, "rate": 5.0, "amplitude": 0.0, "frequency": 0.0, "phase": 0.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.ones(int(steps))
# For 30 steps with 5 steps per level
level_steps = int(steps / 6)  # 5 steps per level for 30 total steps

for level in range(6):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    
    if level == 0:
        exp[start_idx:end_idx] = 100.0
    elif level == 1:
        exp[start_idx:end_idx] = 80.0
    elif level == 2:
        exp[start_idx:end_idx] = 60.0
    elif level == 3:
        exp[start_idx:end_idx] = 40.0
    elif level == 4:
        exp[start_idx:end_idx] = 20.0
    elif level == 5:
        exp[start_idx:end_idx] = 5.0

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("Stepwise DC Rate Decrease\n100 → 80 → 60 → 40 → 20 → 5 Hz (5 steps, 6 levels)")

# Amplitude steps: 50 → 40 → 30 → 20 → 10 → 5 Hz (subplot 3)
plt.subplot(grid[0], grid[1], 3)
# Ensure amplitude <= rate for all steps
spikes = step(
    t,
    n,
    {"order": 3.0, "rate": 60.0, "amplitude": 50.0, "frequency": 10.0, "phase": 0.0},
    {"order": 3.0, "rate": 60.0, "amplitude": 40.0, "frequency": 10.0, "phase": 0.0},
    {"order": 3.0, "rate": 60.0, "amplitude": 30.0, "frequency": 10.0, "phase": 0.0},
    {"order": 3.0, "rate": 60.0, "amplitude": 20.0, "frequency": 10.0, "phase": 0.0},
    {"order": 3.0, "rate": 60.0, "amplitude": 10.0, "frequency": 10.0, "phase": 0.0},
    {"order": 3.0, "rate": 60.0, "amplitude": 5.0, "frequency": 10.0, "phase": 0.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.zeros(int(steps))
level_steps = int(steps / 6)

# Calculate the frequency from parameters
# frequency = params['rate_modulation']['amplitude_steps']['frequency']
frequency = 10.0
for level in range(6):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    time_offset = (start_idx * dt) / 1000.0  # Convert offset to seconds
    time_array = np.arange(0, level_steps * dt, dt) / 1000.0  # Convert to seconds
    time_array_with_offset = time_array + time_offset
    
    if level == 0:
        exp[start_idx:end_idx] = 60.0 + 50.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 1:
        exp[start_idx:end_idx] = 60.0 + 40.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 2:
        exp[start_idx:end_idx] = 60.0 + 30.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 3:
        exp[start_idx:end_idx] = 60.0 + 20.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 4:
        exp[start_idx:end_idx] = 60.0 + 10.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 5:
        exp[start_idx:end_idx] = 60.0 + 5.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)


time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("Stepwise Rate Modulation\n50 → 40 → 30 → 20 → 10 → 5 (5 steps per level)")


# DC rate and amplitude steps: (20,10) → (35,17) → (50,25) → (65,32) → (80,40) (subplot 4)
plt.subplot(grid[0], grid[1], 4)
# Ensure amplitude <= rate for all steps
spikes = step(
    t,
    n,
    {"rate": 20.0, "amplitude": 10.0, "frequency": 20.0, "phase": 0.0, "order": 1},
    {"rate": 35.0, "amplitude": 17.0, "frequency": 20.0, "phase": 0.0, "order": 1},
    {"rate": 50.0, "amplitude": 25.0, "frequency": 20.0, "phase": 0.0, "order": 1},
    {"rate": 65.0, "amplitude": 32.0, "frequency": 20.0, "phase": 0.0, "order": 1},
    {"rate": 80.0, "amplitude": 40.0, "frequency": 20.0, "phase": 0.0, "order": 1},
    {"rate": 95.0, "amplitude": 47.0, "frequency": 20.0, "phase": 0.0, "order": 1},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.zeros(int(steps))
level_steps = int(steps / 6)

# Calculate the frequency from parameters
frequency = 20.0

for level in range(6):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    # Create time array with proper offset for continuous sine wave
    time_offset = (start_idx * dt) / 1000.0  # Convert offset to seconds
    time_array = np.arange(0, level_steps * dt, dt) / 1000.0  # Convert to seconds
    time_array_with_offset = time_array + time_offset
    
    if level == 0:
        exp[start_idx:end_idx] = 20.0 + 10.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 1:
        exp[start_idx:end_idx] = 35.0 + 17.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 2:
        exp[start_idx:end_idx] = 50.0 + 25.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 3:
        exp[start_idx:end_idx] = 65.0 + 32.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 4:
        exp[start_idx:end_idx] = 80.0 + 40.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)
    elif level == 5:
        exp[start_idx:end_idx] = 95.0 + 47.0 * np.sin(2 * np.pi * frequency * time_array_with_offset)

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("Stepwise DC Rate and Rate Modulation\n(20,10) → (35,17) → (50,25) → (65,32) → (80,40) -> (95,47)")
plt.ylabel("Spikes per second")
plt.xlabel("Time (ms)")


# Rate modulation (increase): 5 → 20 → 35 → 50 → 65 → 80 Hz (subplot 5)
plt.subplot(grid[0], grid[1], 5)
# Ensure amplitude <= rate for all steps
spikes = step(
    t,
    n,
    {"rate": 80.0, "amplitude": 5.0, "frequency": 2.5, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 20.0, "frequency": 10.0, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 35.0, "frequency": 12.5, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 50.0, "frequency": 25.0, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 65.0, "frequency": 32.5, "phase": 0.0, "order": 1.0},
    {"rate": 80.0, "amplitude": 80.0, "frequency": 40.0, "phase": 0.0, "order": 1.0},
    seed=123,
    dt=dt,
)
plot_hist(spikes)
exp = np.zeros(int(steps))
level_steps = int(steps / 6)

for level in range(6):
    start_idx = level * level_steps
    end_idx = (level + 1) * level_steps
    # Create time array that matches the actual simulation structure with correct offset
    time_offset = (start_idx * dt) / 1000.0  # Convert offset to seconds
    time_array = np.arange(0, level_steps * dt, dt) / 1000.0  # Convert to seconds
    time_array_with_offset = time_array + time_offset
    
    if level == 0:
        exp[start_idx:end_idx] = 80.0 + 5.0 * np.sin(2 * np.pi * 2.5 * time_array_with_offset)
    elif level == 1:
        exp[start_idx:end_idx] = 80.0 + 20.0 * np.sin(2 * np.pi * 10.0 * time_array_with_offset)
    elif level == 2:
        exp[start_idx:end_idx] = 80.0 + 35.0 * np.sin(2 * np.pi * 17.5 * time_array_with_offset)
    elif level == 3:
        exp[start_idx:end_idx] = 80.0 + 50.0 * np.sin(2 * np.pi * 25.0 * time_array_with_offset)
    elif level == 4:
        exp[start_idx:end_idx] = 80.0 + 65.0 * np.sin(2 * np.pi * 32.5 * time_array_with_offset)
    elif level == 5:
        exp[start_idx:end_idx] = 80.0 + 80.0 * np.sin(2 * np.pi * 40.0 * time_array_with_offset)

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("Stepwise Rate Modulation (Increase)\n5 → 20 → 35 → 50 → 65 → 80 Hz")
plt.xlabel("Time (ms)")

# Modulation Phase Changes (subplot 6)
plt.subplot(grid[0], grid[1], 6)
# Ensure amplitude <= rate
spikes = step(
    t,
    n,
    {"order": 6.0, "rate": 60.0, "amplitude": 60.0, "frequency": 10.0, "phase": 0},
    {"order": 6.0, "rate": 60.0, "amplitude": 60.0, "frequency": 10.0, "phase": 90},
    {"order": 6.0, "rate": 60.0, "amplitude": 60.0, "frequency": 10.0, "phase": 180},
    {"order": 6.0, "rate": 60.0, "amplitude": 60.0, "frequency": 10.0, "phase": 270},
    {"order": 6.0, "rate": 60.0, "amplitude": 60.0, "frequency": 10.0, "phase": 360},
    {"order": 6.0, "rate": 60.0, "amplitude": 60.0, "frequency": 10.0, "phase": 450},
    seed=123,
    dt=dt,
)
plot_hist(spikes)

# Plot expected pattern with proper time axis
exp = np.zeros(int(steps))
level_steps = steps / 6

for level in range(6):
    start_idx = level * int(level_steps)
    end_idx = (level + 1) * int(level_steps)
    # Create time array for this level segment
    time_array = np.arange(start_idx, end_idx) * dt / 1000.0  # Convert to seconds
    
    # Phase values in degrees from the step function calls
    phase_degrees = [0, 90, 180, 270, 360, 450]
    phase_radians = np.radians(phase_degrees[level])
    
    # Calculate the expected firing rate with proper phase shift
    exp[start_idx:end_idx] = 60 + 60 * np.sin(2 * np.pi * 10.0 * time_array + phase_radians)

time_axis = np.arange(len(exp)) * dt
plt.plot(time_axis, exp, "r", linewidth=2)
plt.title("Stepwise Modulation Phase Changes\n0 → π/4 → π/2 → 3π/4 → π → 5π/4")
plt.xlabel("Time (ms)")

# Adjust layout and save second figure
plt.tight_layout()
plt.savefig('stimuli_generation/pain_rate_modulation.png', dpi=300, bbox_inches='tight')
plt.show()

print("Simulation completed. Results saved to 'stimuli_generation/pain_stimuli_simulation.png' and 'stimuli_generation/pain_rate_modulation.png'")