import os
import numpy as np

# Define response functions for each neuron type and condition
def create_simulations(num_rats=5, num_trials=10):
    simulations = []
    
    # Base firing rate for all neurons
    baseline_rate = 1.5  # Hz (reduced from 2.0 to 1.5)
    
    # Mirror in InC_EE: Both demo and self increase, with self showing a much stronger response
    # REDUCED BY 85.9375% (additional 25% reduction from previous)
    def mirror_ee_air(t):
        return np.ones_like(t) * baseline_rate  # Completely uniform

    def mirror_ee_demo(t):
        response = np.ones_like(t) * baseline_rate
        # Moderate increase after laser (reduced by 85.9375%)
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate + 0.6 * np.exp(-(t[mask] - 0.5)**2 / 0.5)
        return response

    def mirror_ee_self(t):
        response = np.ones_like(t) * baseline_rate
        # Strong increase after laser (reduced by 50% from previous value)
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate + 1.125 * np.exp(-(t[mask] - 0.5)**2 / 0.5)  # Changed from 2.25 to 1.125
        return response
    
    mirror_ii_air_baseline = 2

    # Mirror in InC_II: Air is now uniform
    def mirror_ii_air(t):
        return np.ones_like(t) * mirror_ii_air_baseline # Completely uniform
    
    def mirror_ii_demo(t):
        response = np.ones_like(t) * baseline_rate
        # Decrease after laser (reduced by 81.25%)
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate - 0.5625 * np.exp(-(t[mask] - 1.0)**2 / 0.5)
        # Ensure firing rate doesn't go below 0.1
        return np.maximum(response, 0.1)
    
    def mirror_ii_self(t):
        response = np.ones_like(t) * baseline_rate
        # Decrease after laser (reduced by 81.25%)
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate - 0.421875 * np.exp(-(t[mask] - 1.0)**2 / 0.5)
        # Ensure firing rate doesn't go below 0.1
        return np.maximum(response, 0.1)
    
    # Anti-mirror in InC_EI: Demo increases, self decreases - FURTHER REDUCED
    def antimirror_ei_air(t):
        return np.ones_like(t) * baseline_rate  # Completely uniform

    def antimirror_ei_demo(t):
        response = np.ones_like(t) * baseline_rate
        # Reduced increase after laser
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate + 0.4 * np.exp(-(t[mask] - 1.0)**2 / 0.5)  # Reduced from 0.703125
        return response

    def antimirror_ei_self(t):
        response = np.ones_like(t) * baseline_rate
        # Reduced decrease after laser
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate - 0.4 * np.exp(-(t[mask] - 1.0)**2 / 0.5)  # Reduced from 0.703125
        # Ensure firing rate doesn't go below 0.1
        return np.maximum(response, 0.1)
    
    # Anti-mirror in InC_IE: Air is now uniform
    def antimirror_ie_air(t):
        return np.ones_like(t) * baseline_rate  # Completely uniform
    
    def antimirror_ie_demo(t):
        response = np.ones_like(t) * baseline_rate
        # Decrease after laser (reduced by 81.25%)
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate - 0.28125 * np.exp(-(t[mask] - 0.5)**2 / 0.5)
        # Ensure firing rate doesn't go below 0.1
        return np.maximum(response, 0.1)
    
    def antimirror_ie_self(t):
        response = np.ones_like(t) * baseline_rate
        # Increase after laser (reduced by 81.25%)
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate + 0.421875 * np.exp(-(t[mask] - 0.5)**2 / 0.5)
        return response
    
    # NEW: Mirror in AC: Air condition - now uniform
    def mirror_ac_air(t):
        return np.ones_like(t) * baseline_rate  # Completely uniform
    
    mirror_ac_air_baseline_rate = 2.5

    # NEW: Mirror in ACC: Demo excitatory - INCREASED firing rate
    def mirror_acc_demo_e(t):
        response = np.ones_like(t) * mirror_ac_air_baseline_rate
        # Stronger increase after laser - increased from 2.0 to 3.5
        mask = (t >= 0) & (t <= 3)
        response[mask] = mirror_ac_air_baseline_rate + 4 * np.exp(-(t[mask] - 0.5)**2 / 0.5)
        return response

    # NEW: Mirror in ACC: Self excitatory - INCREASED firing rate
    def mirror_acc_self_e(t):
        response = np.ones_like(t) * baseline_rate
        # Stronger increase after laser - keeping the high value of 7.0
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate + 7.0 * np.exp(-(t[mask] - 0.5)**2 / 0.5)
        return response
    
    InC_air_antimirror_baseline_rate = 0.75

    # NEW: Anti-mirror in InC: Air - now uniform
    def antimirror_inc_air(t):
        return np.ones_like(t) * InC_air_antimirror_baseline_rate  # Completely uniform
    
    # NEW: Anti-mirror in InC: Demo excitatory
    def antimirror_inc_demo_e(t):
        response = np.ones_like(t) * baseline_rate
        # Moderate increase after laser
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate + 1.0 * np.exp(-(t[mask] - 0.5)**2 / 0.5)
        return response
    
    # NEW: Anti-mirror in InC: Self inhibitory - INCREASED inhibition
    def antimirror_inc_self_i(t):
        response = np.ones_like(t) * baseline_rate
        # Stronger decrease after laser
        mask = (t >= 0) & (t <= 3)
        response[mask] = baseline_rate - 1.5 * np.exp(-(t[mask] - 0.5)**2 / 0.5)
        # Ensure firing rate doesn't go below 0.1
        return np.maximum(response, 0.1)
    
    # Add all simulations
    simulations.extend([
        # Mirror neurons in InC_EE
        {
            "region": "InC", "neuron_type": "EE", "response_type": "Mirror",
            "condition": "air", "baseline_firing_rate": baseline_rate,
            "response_function": mirror_ee_air
        },
        {
            "region": "InC", "neuron_type": "EE", "response_type": "Mirror",
            "condition": "demo", "baseline_firing_rate": baseline_rate,
            "response_function": mirror_ee_demo
        },
        {
            "region": "InC", "neuron_type": "EE", "response_type": "Mirror",
            "condition": "self", "baseline_firing_rate": baseline_rate,
            "response_function": mirror_ee_self
        },
        
        # Mirror neurons in InC_II
        {
            "region": "InC", "neuron_type": "II", "response_type": "Mirror",
            "condition": "air", "baseline_firing_rate": baseline_rate,
            "response_function": mirror_ii_air
        },
        {
            "region": "InC", "neuron_type": "II", "response_type": "Mirror",
            "condition": "demo", "baseline_firing_rate": baseline_rate,
            "response_function": mirror_ii_demo
        },
        {
            "region": "InC", "neuron_type": "II", "response_type": "Mirror",
            "condition": "self", "baseline_firing_rate": baseline_rate,
            "response_function": mirror_ii_self
        },
        
        # Anti-mirror neurons in InC_EI
        {
            "region": "InC", "neuron_type": "EI", "response_type": "Anti-mirror",
            "condition": "air", "baseline_firing_rate": InC_air_antimirror_baseline_rate,
            "response_function": antimirror_ei_air
        },
        {
            "region": "InC", "neuron_type": "EI", "response_type": "Anti-mirror",
            "condition": "demo", "baseline_firing_rate": baseline_rate,
            "response_function": antimirror_ei_demo
        },
        {
            "region": "InC", "neuron_type": "EI", "response_type": "Anti-mirror",
            "condition": "self", "baseline_firing_rate": baseline_rate,
            "response_function": antimirror_ei_self
        },
        
        # Anti-mirror neurons in InC_IE
        {
            "region": "InC", "neuron_type": "IE", "response_type": "Anti-mirror",
            "condition": "air", "baseline_firing_rate": InC_air_antimirror_baseline_rate,
            "response_function": antimirror_ie_air
        },
        {
            "region": "InC", "neuron_type": "IE", "response_type": "Anti-mirror",
            "condition": "demo", "baseline_firing_rate": baseline_rate,
            "response_function": antimirror_ie_demo
        },
        {
            "region": "InC", "neuron_type": "IE", "response_type": "Anti-mirror",
            "condition": "self", "baseline_firing_rate": baseline_rate,
            "response_function": antimirror_ie_self
        }
    ])

    # Add new simulations
    simulations.extend([
        # Mirror neurons in AC
        {
            "region": "AC", "neuron_type": "E", "response_type": "Mirror",
            "condition": "air", "baseline_firing_rate": mirror_ii_air_baseline,
            "response_function": mirror_ac_air
        },
        # Mirror neurons in ACC - with increased baseline firing rate
        {
            "region": "ACC", "neuron_type": "E", "response_type": "Mirror",
            "condition": "demo", "baseline_firing_rate": mirror_ac_air_baseline_rate,  # Increased from 1.5 to 2.0
            "response_function": mirror_acc_demo_e
        },
        {
            "region": "ACC", "neuron_type": "E", "response_type": "Mirror",
            "condition": "self", "baseline_firing_rate": 2.0,  # Increased from 1.5 to 2.0
            "response_function": mirror_acc_self_e
        },
        # Anti-mirror neurons in InC (additional types)
        {
            "region": "InC", "neuron_type": "E", "response_type": "Anti-mirror",
            "condition": "air", "baseline_firing_rate": baseline_rate,
            "response_function": antimirror_inc_air
        },
        {
            "region": "InC", "neuron_type": "E", "response_type": "Anti-mirror",
            "condition": "demo", "baseline_firing_rate": baseline_rate,
            "response_function": antimirror_inc_demo_e
        },
        {
            "region": "InC", "neuron_type": "I", "response_type": "Anti-mirror",
            "condition": "self", "baseline_firing_rate": baseline_rate,
            "response_function": antimirror_inc_self_i
        }
    ])

    return simulations, num_rats, num_trials

# Generate simulations
simulations, num_rats, num_trials = create_simulations(num_rats=19, num_trials=10)

# Create data directory
data_dir = "simulated_data"
os.makedirs(data_dir, exist_ok=True)

# Generate data for each simulation
for sim in simulations:
    # Create directory for this simulation
    sim_dir = os.path.join(
        data_dir,
        f"raster_data_{sim['region']}",
        f"{sim['neuron_type']}",
        f"{sim['response_type']}",
        f"{sim['condition']}"
    )
    os.makedirs(sim_dir, exist_ok=True)
    
    # Parameters
    simulation_duration = 8.0  # seconds (total: -2 to 6)
    pre_laser_duration = 2.0  # seconds (-2 to 0)
    bin_size = 0.1  # 100 ms bins
    baseline_firing_rate = sim["baseline_firing_rate"]
    response_function = sim["response_function"]
    num_neurons = 10
    
    # Calculate number of bins
    num_bins = int(simulation_duration / bin_size)
    time_bins = np.linspace(-pre_laser_duration, simulation_duration - pre_laser_duration, num_bins + 1)
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # For each rat
    for rat_id in range(1, num_rats + 1):
        # Set a unique seed for each rat to ensure variability between rats
        rat_seed = 42 + rat_id
        
        # For each trial
        for trial_id in range(1, num_trials + 1):
            # Set a unique seed for each trial to ensure variability between trials
            np.random.seed(rat_seed + trial_id * 100)
            
            # Simulate Poisson spike trains
            spike_trains = []
            
            for neuron in range(num_neurons):
                neuron_spikes = []
                
                # Add some variability to each neuron's response
                neuron_variability = np.random.normal(0, 0.2)
                
                # Calculate firing rate at each time point using the response function
                firing_rates = response_function(time_centers) * (1 + neuron_variability)
                firing_rates = np.maximum(firing_rates, 0.1)  # Ensure positive rates
                
                # Generate spikes for each time bin
                for bin_idx, rate in enumerate(firing_rates):
                    lambda_param = rate * bin_size
                    spike_count = np.random.poisson(lambda_param)
                    
                    if spike_count > 0:
                        # Distribute spikes uniformly within the bin
                        bin_start = time_bins[bin_idx]
                        bin_end = time_bins[bin_idx + 1]
                        spike_times = np.random.uniform(bin_start, bin_end, spike_count)
                        neuron_spikes.extend(spike_times)
                
                spike_trains.append(np.array(sorted(neuron_spikes)))
            
            # Save to CSV
            for neuron_id, spikes in enumerate(spike_trains):
                filename = os.path.join(sim_dir, f"rat_{rat_id}_trial_{trial_id}_neuron_{neuron_id}.csv")
                
                with open(filename, 'w', newline='') as f:
                    # Write metadata as comments
                    f.write(f"# Region: {sim['region']}\n")
                    f.write(f"# Condition: {sim['condition']}\n")
                    f.write(f"# Neuron Type: {sim['neuron_type']}\n")
                    f.write(f"# Response Type: {sim['response_type']}\n")
                    f.write(f"# Rat ID: {rat_id}\n")
                    f.write(f"# Trial ID: {trial_id}\n")
                    f.write(f"# Neuron ID: {neuron_id}\n")
                    f.write("\n")
                    
                    # Write the spike times
                    f.write("time,count\n")
                    for spike_time in spikes:
                        f.write(f"{spike_time:.3f},1\n")

print(f"Data saved successfully for {num_rats} rats with {num_trials} trials each!")
print(f"Total number of simulations: {len(simulations)}")
print(f"Total number of files generated: {len(simulations) * num_rats * num_trials * num_neurons}")
