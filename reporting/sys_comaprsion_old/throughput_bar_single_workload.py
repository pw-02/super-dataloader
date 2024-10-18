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
figure_data['ViT-32/Cifar10'] = {'CoorDL': 1239.59875458971, 'Shade': 790, r'$\bf{SUPER}$': 1698.78408196743}
figure_data['ResNet-50/ImageNet'] = {'CoorDL': 825, 'Shade': 382, r'$\bf{SUPER}$': 1066}
figure_data['Albef/COCO'] = {'CoorDL': 284.0157, 'Shade': 284.015788228895, r'$\bf{SUPER}$': 321.434132910756}
figure_data['Pythia-14m/OpenWebText'] = {'LiData': 294016.381, r'$\bf{SUPER}$': 571924.4428}

# Loop over each workload and create individual plots
bar_width = 0.75
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
    plt.ylim(0, 2000)    
    plt.ylabel('Throughput (samples/s)', fontsize=12)

    # # Create a legend based on the bars
    # handles = [plt.Rectangle((0,0),1,1, color=visual_map[dl]['color'], hatch=visual_map[dl]['hatch'], edgecolor=visual_map[dl]['edgecolor']) for dl in loaders]
    # plt.legend(handles, loaders)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
