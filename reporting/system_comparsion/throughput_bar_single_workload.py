from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

# Define the visual map with properties for each data loader
visual_map = {
    r'$\bf{SUPER}$': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'CoorDL': {'color': '#FEA400', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},
    'Shade': {'color': '#4C8BB8', 'hatch': '--', 'edgecolor': 'black', 'alpha': 1.0},
    'LiData': {'color': '#FF7F0E', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1.0},
}

# Define the workloads
figure_data: Dict[str, Dict[str, float]] = {}
figure_data['ResNet-18/CIFAR-10'] = {'CoorDL': 5118.62461267584, 'Shade': 1989.18672796396, r'$\bf{SUPER}$': 4622.45918749042}
figure_data['ResNet-50/ImageNet'] = {'CoorDL': 435.187500337274, 'Shade': 380.291178986009, r'$\bf{SUPER}$': 489.852536604153}
figure_data['Albef/COCO'] = {'CoorDL': 90, 'Shade': 70.6977693463706, r'$\bf{SUPER}$': 193.21531208424}
figure_data['Pythia-14m/OpenWebText'] = {'LiData': 294016.381, r'$\bf{SUPER}$': 571924.4428}

# Loop over each workload and create individual plots
bar_width = 0.5
for workload in figure_data.keys():
    # Create a new figure for each workload
    plt.figure(figsize=(4, 2.5))
    
    # Extract the relevant data loaders for the current workload
    loaders = list(figure_data[workload].keys())  # Get only the data loaders present for this workload
    data_to_plot = [figure_data[workload][dl] for dl in loaders]  # Get the corresponding values
    
    # Create the bar plot for this workload
    bars = plt.bar(loaders, data_to_plot, 
                   color=[visual_map[dl]['color'] for dl in loaders], 
                   edgecolor=[visual_map[dl]['edgecolor'] for dl in loaders], 
                   hatch=[visual_map[dl]['hatch'] for dl in loaders], 
                   width=bar_width, alpha=1.0)
    
    # Set titles and labels
    plt.title(f'{workload}', fontsize=12)
    plt.ylabel('Throughput (samples/s)', fontsize=12)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()