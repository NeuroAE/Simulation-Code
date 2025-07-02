# Create plots similar to the image showing mirror neurons in ACC and anti-mirror neurons in InC
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Define the regions and conditions to plot
regions_to_plot = {
    "Mirror in ACC": {
        "region": "ACC", 
        "neuron_type": "E", 
        "response_type": "Mirror"
    },
    "Anti-mirror in InC": {
        "region": "InC", 
        "neuron_type": "E",  # This will be overridden in the plot_condition function
        "response_type": "Anti-mirror"
    }
}

conditions = ["air", "demo", "self"]

# Create the figure
fig = plt.figure(figsize=(10, 12))
gs = GridSpec(7, 2, height_ratios=[1, 2, 1, 2, 1, 2, 1])

# Add the schematic diagrams at the top (placeholder rectangles)
for col, title in enumerate(regions_to_plot.keys()):
    ax_schema = fig.add_subplot(gs[0, col])
    # Create a more sophisticated schematic
    if "Mirror in ACC" in title:
        ax_schema.text(0.5, 0.5, "Mirror\nNeurons", ha='center', va='center', fontsize=12, fontweight='bold')
    else:
        ax_schema.text(0.5, 0.5, "Anti-Mirror\nNeurons", ha='center', va='center', fontsize=12, fontweight='bold')
    ax_schema.set_title(title, fontsize=14)
    ax_schema.axis('off')

# Function to load and plot data for a specific configuration
def plot_condition(region_info, condition, row_idx, col_idx):
    # Special case for anti-mirror in InC self condition
    if col_idx == 1 and condition == "self":
        region_info = region_info.copy()
        region_info["neuron_type"] = "I"  # Use inhibitory neurons for self condition
    
    # For air condition in Mirror in ACC, use AC region instead of ACC
    if col_idx == 0 and condition == "air":
        region_info = region_info.copy()
        region_info["region"] = "AC"  # Use AC for air condition
    
    simulation_duration = 8.0
    pre_laser_duration = 2.0
    bin_size = 0.1
    num_neurons = 10
    
    # Create time bins for histogram
    num_bins = int(simulation_duration / bin_size)
    time_bins = np.linspace(-pre_laser_duration, simulation_duration - pre_laser_duration, num_bins + 1)
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # Directory where data is stored
    data_dir = os.path.join(
        "simulated_data",
        f"raster_data_{region_info['region']}",
        f"{region_info['neuron_type']}",
        f"{region_info['response_type']}",
        f"{condition}"
    )
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist")
        return
    
    # Create axes for raster plot and histogram
    ax_raster = fig.add_subplot(gs[row_idx, col_idx])
    ax_hist = fig.add_subplot(gs[row_idx+1, col_idx], sharex=ax_raster)
    
    # Load spike data (using just one rat and one trial for simplicity)
    rat_id = 1
    trial_id = 1
    all_spikes = []
    
    for neuron_id in range(num_neurons):
        filename = os.path.join(data_dir, f"rat_{rat_id}_trial_{trial_id}_neuron_{neuron_id}.csv")
        
        if os.path.exists(filename):
            # Skip header lines and read spike times
            spike_times = []
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '' or line.startswith('time'):
                        continue
                    try:
                        time_val = float(line.split(',')[0])
                        spike_times.append(time_val)
                    except:
                        pass
            
            # Plot raster for this neuron
            ax_raster.eventplot([spike_times], lineoffsets=neuron_id+1, 
                              linelengths=0.8, linewidths=0.5, color='black')
            all_spikes.extend(spike_times)
    
    # Plot histogram
    hist_counts, _ = np.histogram(all_spikes, bins=time_bins)
    # Convert counts to firing rate (imp/sec)
    firing_rates = hist_counts / (bin_size * num_neurons)
    ax_hist.bar(time_centers, firing_rates, width=bin_size, color='black', align='center')
    
    # Set axis limits
    ax_raster.set_ylim(0.5, num_neurons + 0.5)
    ax_raster.set_xlim(-2, 4)  # Match the image time range
    
    # Add vertical line at t=0 (laser activation)
    ax_raster.axvline(x=0, color='red', linestyle='-', linewidth=1)
    ax_hist.axvline(x=0, color='red', linestyle='-', linewidth=1)
    
    # Add condition label with color coding
    if condition == "demo":
        color = "red"
        label = "demo +"
    elif condition == "self":
        color = "blue" if col_idx == 1 else "red"  # Blue for anti-mirror self
        label = "self -" if col_idx == 1 else "self +"
    else:
        color = "black"
        label = "air"
    
    ax_raster.text(0.05, 0.9, label, transform=ax_raster.transAxes, 
                 color=color, fontsize=12, fontweight='bold')
    
    # Remove some axis elements for cleaner look
    ax_raster.set_yticks([])
    if row_idx < 5:  # Only show x-axis for bottom plots
        ax_hist.set_xticklabels([])
    
    # Set y-axis limits for histogram based on region
    if col_idx == 0:  # Mirror in ACC
        ax_hist.set_ylim(0, 16)
        ax_hist.set_yticks([0, 8, 16])
    else:  # Anti-mirror in InC
        ax_hist.set_ylim(0, 7)
        ax_hist.set_yticks([0, 7])
    
    # Add y-axis label only to leftmost plots
    if col_idx == 0:
        ax_hist.set_ylabel('Frequency\n(imp/sec)')
    
    # Add x-axis label only to bottom plots
    if row_idx == 4:
        ax_hist.set_xlabel('Time (s)')

# Plot each condition for each region
for col_idx, (region_name, region_info) in enumerate(regions_to_plot.items()):
    for row_idx, condition in enumerate(conditions):
        plot_condition(region_info, condition, row_idx*2+1, col_idx)

# Create output directory if it doesn't exist
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig(os.path.join(output_dir, "mirror_antimirror_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()
