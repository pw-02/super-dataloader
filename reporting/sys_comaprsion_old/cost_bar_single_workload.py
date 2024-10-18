from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

# Define the visual map with properties for each data loader
visual_map = {
    r'$\bf{SUPER}$': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'CoorDL':  {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'Shade': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'LitData': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
}

# Define the workloads and their associated costs
figure_data: Dict[str, Dict[str, Dict[str, float]]] = {
    'ResNet-50/ImageNet': {
        'CoorDL': {"Compute": 36.1275267572086, "Cache": 8, "Prefetch": 0},
        'Shade': {"Compute": 77.8915604829167, "Cache": 19.1928877791239, "Prefetch": 0},
        r'$\bf{SUPER}$': {"Compute": 26.30234985, "Cache": 4.857332747, "Prefetch": 1.905632259},
    },
    'ViT-32/Cifar10': {
        'CoorDL': {"Compute":1.097, "Cache":0.044, "Prefetch": 0},
        'Shade': {"Compute": 1.719, "Cache": 0.044, "Prefetch": 0},
        r'$\bf{SUPER}$': {"Compute": 0.800, "Cache":  0.0044, "Prefetch": 0.002},
    },
    'Albef/COCO': {
        'CoorDL': {"Compute":16.90456879, "Cache":1.933336844, "Prefetch": 0},
        'Shade': {"Compute": 15.90456879, "Cache": 1.633336844, "Prefetch": 0},
        r'$\bf{SUPER}$': {"Compute": 5.824730907, "Cache": 0.022080679, "Prefetch": 0.489402845},
    },
    'Pythia-14m/OpenWebText': {
        'LitData': {"Compute": 18.78310914, "Cache": 1.928951649, "Prefetch": 0},
        r'$\bf{SUPER}$': {"Compute": 9.740027848, "Cache": 0.036922981, "Prefetch": 2.485014487},
    }
}

# Loop over each workload and create individual stacked bar plots
bar_width = 0.75
for workload, loaders_data in figure_data.items():
    # Create a new figure for each workload
    # plt.figure(figsize=(6, 4))
    plt.figure(figsize=(4, 2.5))
    # Extract the relevant data loaders for the current workload
    loaders = list(loaders_data.keys())
    
    # Separate the compute, Cache, and Prefetch costs for stacking
    compute_costs = [loaders_data[dl]['Compute'] for dl in loaders]
    Cache_costs = [loaders_data[dl]['Cache'] for dl in loaders]
    Prefetch_costs = [loaders_data[dl]['Prefetch'] for dl in loaders]
    
    # Create the stacked bar plot
    bottom_Cache = compute_costs
    bottom_Prefetch = np.array(compute_costs) + np.array(Cache_costs)
    
    plt.bar(loaders, compute_costs, color=[visual_map[dl]['color'] for dl in loaders], 
            edgecolor=[visual_map[dl]['edgecolor'] for dl in loaders], 
            hatch='//', 
            label='Compute', width=bar_width)
    
    plt.bar(loaders, Cache_costs, bottom=compute_costs, color='#FEA400', 
            edgecolor=[visual_map[dl]['edgecolor'] for dl in loaders], 
            hatch='..', 
            label='Cache', width=bar_width)
    
    plt.bar(loaders, Prefetch_costs, bottom=bottom_Prefetch, color='lightgrey', 
            edgecolor=[visual_map[dl]['edgecolor'] for dl in loaders], 
            hatch='', 
            label='Prefetch', width=bar_width)
    
    plt.ylim(0, 2)
    
    # Calculate total costs for the current workload
    total_costs = np.array(compute_costs) + np.array(Cache_costs) + np.array(Prefetch_costs)
    
    # Set titles and labels
    plt.title(f'{workload}', fontsize=12)
    plt.ylabel('Training Cost ($)', fontsize=12)
    
    # Format y-tick labels to include $
    y_ticks = plt.yticks()[0]  # Get the current y-ticks
    plt.yticks(y_ticks, [f"${tick:.0f}" for tick in y_ticks])  # Format y-ticks
    
    # Annotate total costs on top of the last bar
    for i, total in enumerate(total_costs):
        plt.text(i, total + 0.05, f"${total:.2f}", ha='center', va='bottom')
    
    plt.legend(loc='upper right', fontsize=8, ncol=1)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
