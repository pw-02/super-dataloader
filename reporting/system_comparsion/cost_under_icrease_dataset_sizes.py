from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# Custom formatter to display Y-tick labels in 'K'
def thousands_formatter(x, pos):
    return f'{x / 1000:.1f}K'
# Custom formatter to add a dollar sign to Y-ticks in the cost subplot
def dollar_formatter(x, pos):
    return f'${x:.0f}'

def percent_formatter(x, pos):
    return f'{int(x)}%'

# # Define the visual map and figure data
# visual_map = {
#     r'$\bf{SUPER}$': {'color': '#005250', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
#     'CoorDL': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
#     'Shade': {'color': '#4C8BB8', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
#     'LiData': {'color': '#FF7F0E', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
# }

visual_map = {
    r'$\bf{SUPER}$': {'color': '#005250', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
    'CoorDL': {'color': '#4C8BB8', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'s', 'linestyle':'-'},
    'Shade': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'^', 'linestyle':'-'},
    'LiData': {'color': '#FF7F0E', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
}
workload_data: Dict[str, Dict[str, float]] = {}
workload_data['Resnet50/ImageNet'] = {
        "Cost" : { 
        "CoorDL": {'30': 4.14, '60': 10.031, '90': 17.88, '120': 26.716, '150': 37.33},
        "Shade": {'30': 4.14, '60': 10.031, '90': 17.88, '120': 26.716, '150': 37.33},
        r'$\bf{SUPER}$': {'30':3.315, '60': 7.604, '90':11.58, '120': 15.27, '150': 19.04}},
}
x_tick_labels = [30,60,90,120,150]
x_label = 'Dataset Size (GB)'
bar_width = 0.25
for workload in workload_data:
    workload_name = workload
    fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(3.25, 2.3))
    workload_throuhgput = workload_data[workload]["Cost"]
    dataset_sizes = list(workload_throuhgput["CoorDL"].keys())
    x = np.arange(len(dataset_sizes))

    # Plotting the bars for cost
    workload_cost = workload_data[workload]["Cost"]

    ax2.plot(x, 
            workload_cost['CoorDL'].values(),  
            label='CoorDL', 
            color=visual_map['CoorDL']['color'], 
            linestyle=visual_map['CoorDL']['linestyle'], 
            marker=visual_map['CoorDL']['marker'])
    ax2.plot(x,
            workload_cost['Shade'].values(),  
            label='Shade', 
            color=visual_map['Shade']['color'], 
            linestyle=visual_map['Shade']['linestyle'], 
            marker=visual_map['Shade']['marker'])
    ax2.plot(x, 
            workload_cost[r'$\bf{SUPER}$'].values(), 
            label=r'$\bf{SUPER}$', 
            color=visual_map[r'$\bf{SUPER}$']['color'], 
            linestyle=visual_map[r'$\bf{SUPER}$']['linestyle'], 
            marker=visual_map[r'$\bf{SUPER}$']['marker'])

    ax2.set_ylabel('Dataloading Cost Per Epoch ($)', fontsize=11)
    ax2.set_ylim(0, 45)  # Adjust limits for clarity
    # Get current y-limits
    current_ylim = ax2.get_ylim()
    padding = 0.25
    ax2.set_ylim(current_ylim[0], current_ylim[1] + padding)  # Extend the upper limit
    ax2.legend( fontsize=10)
    ax2.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))  # Apply custom formatter

    ax2.set_xticks(x + bar_width)  # Center ticks under the grouped bars
    ax2.set_xticklabels(x_tick_labels, fontsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xlabel(x_label, fontsize=11)
    # ax2.legend()
    ax2.legend(ncol=1, loc='upper left', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
