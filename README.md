# Simulation Code for NeuroAE
In silico experiment with neuronal activity of affective empathy in rats

### Instructions for running code

#### 1. Create simulated data
```
python 1-create-simulations.py
```
This code generates uniform spike trains directly using time-varying firing rates derived from response functions for the "demo" and "self" conditions. For the "air" condition, it generates poisson spike trains. 

#### 2. Load pre-simulated spike data and create raster plots/histograms
```
python 2-raster-plots-and-histograms.py
```

#### 3. Load synthetic spike trains from CSV files organised by neuron types, trial and condition
```
python 3-z-score-plots.py
```
This script bins spikes over time and computes the z-scores (standardised firing rate relative to the baseline) and SEM (standard error of the mean). 

It then plots a 2x2 grid comparing population-level z-scored activity of different neuron classes; while highlighting statistically significant time regions using a threshold. Each subplot is annotated with metadata on the number of rats, trials, and neurons. 

---

The generated spike trains using time-varying firing rates are saved to the "simulated_data" folder in CSV files. 

The raster plots/histograms and z-score plots are saved to the "output_plots" folder in PNG files. 

