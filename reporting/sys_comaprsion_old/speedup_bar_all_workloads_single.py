from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Define the visual map and figure data
visual_map = {
    r'$\bf{SUPER}$': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'CoorDL': {'color': '#FEA400', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},
    'Shade': {'color': '#4C8BB8', 'hatch': '////', 'edgecolor': 'black', 'alpha': 1.0},
    'LiData': {'color': '#FF7F0E', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1.0},
}

# Define the workloads and corresponding data

# Define the workloads
figure_data: Dict[str, Dict[str, float]] = {}
# figure_data['ResNet-18/Cifar10'] = {'CoorDL': 1.2,'Shade': 1, r'$\bf{SUPER}$': 1.4}
# figure_data['ResNet-18/Cifar10'] = {'Shade': 1, 'CoorDL': 1.5, r'$\bf{SUPER}$': 2.78}
figure_data['ResNet-50/ImageNet'] = {'Shade': 1, 'CoorDL': 1.5, r'$\bf{SUPER}$': 2.78}
figure_data['ResNet-50/ImageNet'] = {'Shade': 1, 'CoorDL': 1.5, r'$\bf{SUPER}$': 2.78}

figure_data['Albef/Coco'] ={'Shade': 1, 'CoorDL': 1.3, r'$\bf{SUPER}$': 4}
figure_data['Pythia14m/OpenWebText'] = {'LiData': 1, r'$\bf{SUPER}$': 3.5}


# figure_data = {
#     'ResNet-18': {'CoorDL': 12, 'Shade': 10, r'$\bf{SUPER}$': 14},
#     'ResNet-50': {'CoorDL': 12, 'Shade': 10, r'$\bf{SUPER}$': 14},
#     'Albef/COCO':{'CoorDL': 12,'Shade': 10, r'$\bf{SUPER}$': 14},
#     'Pythia-14m': {'LiData': 10, r'$\bf{SUPER}$': 14}}

# Create a new figure and set of axes
fig, ax = plt.subplots(figsize=(8, 3)) 

# Define the x-axis positions for each workload
x = np.arange(len(figure_data))

# Set bar width
bar_width = 0.27

# Create a list for legend handles
legend_handles = []
already_added = []

# Iterate over each workload and dynamically plot bars based on the available dataloaders
for i, workload in enumerate(figure_data):
    dataloaders = list(figure_data[workload].keys())
    num_dataloaders = len(dataloaders)
    
    # Adjust the starting position for the bars so they are centered for workloads with fewer dataloaders
    start_position = x[i] - (bar_width * (num_dataloaders - 1) / 2)
    
    # Plot each dataloader's bar at the adjusted positions
    for j, dataloader in enumerate(dataloaders):
        value = figure_data[workload][dataloader]
        style = visual_map[dataloader]
        
        # Plot the bar
        ax.bar(start_position + j * bar_width, value, bar_width,
               color=style['color'], 
               hatch=style['hatch'], 
               edgecolor=style['edgecolor'], 
               alpha=style['alpha'])
        
        # Add to legend handles if it's the first occurrence
        if dataloader not in already_added:
            already_added.append(dataloader)
            legend_handles.append(Patch(color=style['color'], label=dataloader, hatch=style['hatch'], edgecolor=style['edgecolor']))

# Set x-axis labels and title
ax.set_ylabel('Training Speedup wrt Baseline', fontsize=12)
# Set the tick positions and labels for the x-axis
ax.set_xticks(x)
ax.set_xticklabels(figure_data.keys(), fontsize=12)
ax.tick_params(axis='y', labelsize=12)  # Set font size for y-tick labels

# Add the legend with custom handles
ax.legend(handles=legend_handles, ncol=4, loc='upper center')
ax.set_ylim(0, 5)
plt.subplots_adjust(left=0.074, right=0.956, top=0.945, bottom=0.097, wspace=0.2, hspace=0.2)

# Display the plot
plt.tight_layout()
plt.show()
