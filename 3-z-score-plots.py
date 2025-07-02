import pandas as pd
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from io import StringIO
from scipy.ndimage import label

def load_simulated_raster_data():
    """Load simulated raster data from the specified directories, organizing by rat and trial"""
    base_path = "simulated_data/raster_data_InC"
    
    # Structure to hold all the data
    # We'll organize by neuron type, response type, condition, rat, trial
    raster_data = {
        "EE": {"mirror": {"air": {}, "demo": {}, "self": {}}},
        "EI": {"anti_mirror": {"air": {}, "demo": {}, "self": {}}},
        "IE": {"anti_mirror": {"air": {}, "demo": {}, "self": {}}},
        "II": {"mirror": {"air": {}, "demo": {}, "self": {}}}
    }
    
    # Define the paths to load
    paths_to_load = [
        ("EE", "Mirror"),
        ("EI", "Anti-mirror"),
        ("IE", "Anti-mirror"),
        ("II", "Mirror")
    ]
    
    # Load data from each directory
    for neuron_type, neuron_class in paths_to_load:
        for condition in ["air", "demo", "self"]:
            dir_path = os.path.join(base_path, neuron_type, neuron_class, condition)
            if os.path.exists(dir_path):
                csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    # Parse rat and trial from filename
                    # Expected format: rat_X_trial_Y_neuron_Z.csv
                    parts = csv_file.split('_')
                    if len(parts) >= 6 and parts[0] == 'rat' and parts[2] == 'trial':
                        try:
                            rat_id = int(parts[1])
                            trial_id = int(parts[3])
                            
                            file_path = os.path.join(dir_path, csv_file)
                            
                            # Read the file as text first to handle the comment lines
                            with open(file_path, 'r') as f:
                                lines = f.readlines()
                            
                            # Find where the CSV data actually starts (after the comments)
                            data_start = 0
                            for i, line in enumerate(lines):
                                if line.startswith('time,count') or line.startswith('Time,Count'):
                                    data_start = i
                                    break
                            
                            # Extract just the CSV part
                            csv_data = lines[data_start:]
                            
                            # Parse the CSV data
                            if csv_data:
                                # Create a StringIO object to parse with pandas
                                csv_string = ''.join(csv_data)
                                spike_data = pd.read_csv(StringIO(csv_string))
                                
                                # Extract spike times
                                if 'time' in spike_data.columns:
                                    spike_times = spike_data['time'].values
                                elif 'Time' in spike_data.columns:
                                    spike_times = spike_data['Time'].values
                                else:
                                    # Use the first column
                                    spike_times = spike_data.iloc[:, 0].values
                                
                                # Create a DataFrame with just the spike times
                                spike_df = pd.DataFrame({'time': spike_times})
                                
                                # Determine the key for storing the data
                                key = "mirror" if neuron_class == "Mirror" else "anti_mirror"
                                
                                # Initialize rat dictionary if it doesn't exist
                                if rat_id not in raster_data[neuron_type][key][condition]:
                                    raster_data[neuron_type][key][condition][rat_id] = {}
                                
                                # Initialize trial list if it doesn't exist
                                if trial_id not in raster_data[neuron_type][key][condition][rat_id]:
                                    raster_data[neuron_type][key][condition][rat_id][trial_id] = []
                                
                                # Store the spike data
                                raster_data[neuron_type][key][condition][rat_id][trial_id].append(spike_df)
                            
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
    
    return raster_data

def calculate_z_scores(spike_data_dict, time_bins):
    """
    Calculate Z-scores from spike data organized by rat and trial
    
    Parameters:
    -----------
    spike_data_dict : dict
        Dictionary with structure {rat_id: {trial_id: [spike_dataframes]}}
    time_bins : array
        Time bins for binning spike data
    
    Returns:
    --------
    z_scores : array
        Z-scores for each time bin
    time_centers : array
        Centers of time bins for plotting
    """
    if not spike_data_dict:
        return np.zeros(len(time_bins) - 1), (time_bins[:-1] + time_bins[1:]) / 2
    
    # Calculate bin centers for plotting
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # Initialize array to hold binned spike counts for all rats and trials
    all_rat_trial_counts = []
    
    # Process each rat
    for rat_id, trial_dict in spike_data_dict.items():
        # Process each trial for this rat
        for trial_id, spike_data_list in trial_dict.items():
            # Process each neuron in this trial
            for spike_data in spike_data_list:
                # Extract spike times from the DataFrame
                spike_times = spike_data['time'].values
                
                # Bin the spike times
                binned_counts, _ = np.histogram(spike_times, bins=time_bins)
                all_rat_trial_counts.append(binned_counts)
    
    if not all_rat_trial_counts:
        return np.zeros(len(time_bins) - 1), time_centers
    
    # Convert to numpy array
    all_rat_trial_counts = np.array(all_rat_trial_counts)
    
    # Calculate mean across all neurons from all rats and trials
    mean_counts = np.mean(all_rat_trial_counts, axis=0)
    
    # Calculate baseline (pre-stimulus) mean and std
    baseline_idx = time_centers < 0
    baseline_mean = np.mean(mean_counts[baseline_idx])
    baseline_std = np.std(mean_counts[baseline_idx])
    
    # Avoid division by zero
    if baseline_std == 0:
        baseline_std = 1.0
    
    # Calculate Z-scores
    z_scores = (mean_counts - baseline_mean) / baseline_std
    
    return z_scores, time_centers

def calculate_sem(spike_data_dict, time_bins):
    """
    Calculate standard error of the mean for error bars
    
    Parameters:
    -----------
    spike_data_dict : dict
        Dictionary with structure {rat_id: {trial_id: [spike_dataframes]}}
    time_bins : array
        Time bins for binning spike data
    
    Returns:
    --------
    sem : array
        Standard error of the mean for each time bin
    """
    if not spike_data_dict:
        return np.zeros(len(time_bins) - 1)
    
    # Calculate bin centers for plotting
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # Initialize array to hold z-scored data for each rat-trial-neuron
    all_z_scored_counts = []
    
    # Process each rat
    for rat_id, trial_dict in spike_data_dict.items():
        # Process each trial for this rat
        for trial_id, spike_data_list in trial_dict.items():
            # Process each neuron in this trial
            for spike_data in spike_data_list:
                # Extract spike times from the DataFrame
                spike_times = spike_data['time'].values
                
                # Bin the spike times
                binned_counts, _ = np.histogram(spike_times, bins=time_bins)
                
                # Calculate baseline (pre-stimulus) mean and std for this neuron
                baseline_idx = time_centers < 0
                baseline_mean = np.mean(binned_counts[baseline_idx])
                baseline_std = np.std(binned_counts[baseline_idx])
                
                # Avoid division by zero
                if baseline_std == 0:
                    baseline_std = 1.0
                
                # Calculate z-scores for this neuron
                z_scored = (binned_counts - baseline_mean) / baseline_std
                all_z_scored_counts.append(z_scored)
    
    if not all_z_scored_counts:
        return np.zeros(len(time_bins) - 1)
    
    # Convert to numpy array
    all_z_scored_counts = np.array(all_z_scored_counts)
    
    # Calculate SEM across all neurons from all rats and trials
    # SEM = standard deviation / sqrt(n)
    sem = np.std(all_z_scored_counts, axis=0) / np.sqrt(len(all_z_scored_counts))
    
    return sem

def count_neurons(spike_data_dict):
    """Count the total number of neurons across all rats and trials"""
    count = 0
    for rat_id, trial_dict in spike_data_dict.items():
        for trial_id, spike_data_list in trial_dict.items():
            count += len(spike_data_list)
    return count

def create_inc_grid_plot_with_rat_trial_info():
    """Create a 2x2 grid plot for InC neurons with rat and trial count information"""
    # Define time bins for analysis
    time_bins = np.arange(-2, 6.1, 0.1)  # From -2 to 6 seconds in 0.1s bins
    
    # Load simulated data
    raster_data_InC = load_simulated_raster_data()
    
    # Create output directory if it doesn't exist
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a 2x2 figure with specific size
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Data configurations
    plot_configs = [
        {"data": raster_data_InC["EE"]["mirror"], "title": "Mirror Neurons in InC (EE)", "position": (0, 0)},
        {"data": raster_data_InC["II"]["mirror"], "title": "Mirror Neurons in InC (II)", "position": (0, 1)},
        {"data": raster_data_InC["EI"]["anti_mirror"], "title": "Anti-Mirror Neurons in InC (EI)", "position": (1, 0)},
        {"data": raster_data_InC["IE"]["anti_mirror"], "title": "Anti-Mirror Neurons in InC (IE)", "position": (1, 1)}
    ]
    
    # Generate each subplot
    for config in plot_configs:
        row, col = config["position"]
        ax = axs[row, col]
        
        data = config["data"]
        title = config["title"]
        
        # Check if we have data for each condition
        has_air = bool(data['air'])
        has_demo = bool(data['demo'])
        has_self = bool(data['self'])
        
        # Calculate Z-scores for each condition
        air_z_scores, time_centers = calculate_z_scores(data['air'], time_bins)
        demo_z_scores, _ = calculate_z_scores(data['demo'], time_bins)
        self_z_scores, _ = calculate_z_scores(data['self'], time_bins)

        # Calculate SEM for error bars
        air_sem = calculate_sem(data['air'], time_bins) if has_air else np.zeros_like(time_centers)
        demo_sem = calculate_sem(data['demo'], time_bins) if has_demo else np.zeros_like(time_centers)
        self_sem = calculate_sem(data['self'], time_bins) if has_self else np.zeros_like(time_centers)
        
        # Apply smoothing to make the curves look nicer
        air_z_scores = gaussian_filter1d(air_z_scores, sigma=1.5)
        demo_z_scores = gaussian_filter1d(demo_z_scores, sigma=1.5)
        self_z_scores = gaussian_filter1d(self_z_scores, sigma=1.5)
        
        # Also smooth the SEM values for visual consistency
        air_sem = gaussian_filter1d(air_sem, sigma=1.5)
        demo_sem = gaussian_filter1d(demo_sem, sigma=1.5)
        self_sem = gaussian_filter1d(self_sem, sigma=1.5)
        
        # Ensure minimum SEM values for visibility
        min_sem = 0.5
        air_sem = np.maximum(air_sem, min_sem)
        demo_sem = np.maximum(demo_sem, min_sem)
        self_sem = np.maximum(self_sem, min_sem)
        
        # Draw the shadows (SEM) with higher alpha and zorder to ensure visibility
        if has_air:
            ax.fill_between(time_centers, air_z_scores-air_sem, air_z_scores+air_sem, 
                           color='purple', alpha=0.3, zorder=1)
        
        if has_demo:
            ax.fill_between(time_centers, demo_z_scores-demo_sem, demo_z_scores+demo_sem, 
                           color='orange', alpha=0.3, zorder=2)
        if has_self:
            ax.fill_between(time_centers, self_z_scores-self_sem, self_z_scores+self_sem, 
                           color='green', alpha=0.3, zorder=3)
        
        # Then draw the main lines on top with increased linewidth for better visibility
        if has_air:
            ax.plot(time_centers, air_z_scores, color='purple', label='air', linewidth=2, zorder=4)
        
        if has_demo:
            ax.plot(time_centers, demo_z_scores, color='orange', label='demo', linewidth=2, zorder=5)
        
        if has_self:
            ax.plot(time_centers, self_z_scores, color='green', label='self', linewidth=2, zorder=6)
        
        # Add vertical line at laser activation
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=1)
        
        # Add significance markers based on actual data
        threshold = 2.0  # Z-score threshold for significance

        if has_air and has_demo:
            # Find regions where demo differs significantly from air
            demo_diff = demo_z_scores - air_z_scores
            
            # Find continuous regions of significance
            labeled_regions, num_regions = label(np.abs(demo_diff) > threshold)
            
            for region in range(1, num_regions+1):
                region_indices = np.where(labeled_regions == region)[0]
                if len(region_indices) > 5:  # Only show if region is substantial
                    sig_start = time_centers[region_indices[0]]
                    sig_end = time_centers[region_indices[-1]]
                    
                    # Calculate appropriate y-position based on the data in this region
                    data_max = max(
                        np.max(air_z_scores[region_indices]) if has_air else -np.inf,
                        np.max(demo_z_scores[region_indices]) if has_demo else -np.inf,
                        np.max(self_z_scores[region_indices]) if has_self else -np.inf
                    )
                    # Position the bar slightly above the maximum data point
                    y_pos = data_max + 2
                    
                    ax.plot([sig_start, sig_end], [y_pos, y_pos], color='orange', linewidth=2)
                    ax.text((sig_start + sig_end)/2, y_pos+1, '*', fontsize=14, ha='center')
            
        if has_air and has_self:
            # Find regions where self differs significantly from air
            self_diff = self_z_scores - air_z_scores
            
            labeled_regions, num_regions = label(np.abs(self_diff) > threshold)
            
            for region in range(1, num_regions+1):
                region_indices = np.where(labeled_regions == region)[0]
                if len(region_indices) > 5:
                    sig_start = time_centers[region_indices[0]]
                    sig_end = time_centers[region_indices[-1]]
                    
                    # Calculate appropriate y-position based on the data in this region
                    # Position slightly below the demo significance bar if it exists
                    data_max = max(
                        np.max(air_z_scores[region_indices]) if has_air else -np.inf,
                        np.max(demo_z_scores[region_indices]) if has_demo else -np.inf,
                        np.max(self_z_scores[region_indices]) if has_self else -np.inf
                    )
                    # Position the bar slightly above the maximum data point, but below the demo bar
                    y_pos = data_max + 4
                    
                    ax.plot([sig_start, sig_end], [y_pos, y_pos], color='green', linewidth=2)
                    ax.text((sig_start + sig_end)/2, y_pos+1, '*', fontsize=14, ha='center')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-score')
        
        # Set consistent y-limits for all plots
        ax.set_ylim(-8, 20)
        ax.set_xlim(-2, 6)
        
        # Count rats and trials for each condition
        rat_trial_counts = {}
        total_neurons = 0
        
        for condition in ['air', 'demo', 'self']:
            if data[condition]:
                rat_count = len(data[condition])
                trial_counts = []
                neuron_count = 0
                
                for rat_id, trials in data[condition].items():
                    trial_count = len(trials)
                    trial_counts.append(trial_count)
                    
                    # Count neurons in this rat
                    for trial_id, neurons in trials.items():
                        neuron_count += len(neurons)
                
                avg_trials = sum(trial_counts) / len(trial_counts) if trial_counts else 0
                rat_trial_counts[condition] = {
                    'rats': rat_count,
                    'avg_trials': avg_trials,
                    'neurons': neuron_count
                }
                total_neurons += neuron_count
        
        # Create title with rat and trial information
        title_parts = [title]
        
        ax.set_title("\n".join(title_parts), fontsize=9)
    
    # Add a single legend for the entire figure
    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=3, frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the legend
    plt.savefig(os.path.join(output_dir, "InC_neurons_grid_plot_with_rat_info.png"), dpi=300, bbox_inches='tight')
    plt.show()

# Run the function to create the grid plot with rat and trial information
create_inc_grid_plot_with_rat_trial_info()