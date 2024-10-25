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
     'io': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'transform': {'color': '#FEA400', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},
    'gpu': {'color': '#4C8BB8', 'hatch': '/////', 'edgecolor': 'black', 'alpha': 1.0},
}
workload_data: Dict[str, Dict[str, float]] = {}
workload_data['Resnet50/ImageNet'] = {
    "Throughput" : 
    { 
        "CoorDL": {'30': 922.22, '60': 922.22, '90': 922.22, '120': 922.22, '150': 922.22},
        "Shade": {'30': 922.22, '60': 922.22, '90': 922.22, '120': 922.22, '150': 922.22},
        r'$\bf{SUPER}$': {'30': 1406.76, '60': 1406.76, '90': 1406.76, '120': 1406.76, '150': 1406.76}
    },
    "Cost" : 
    { 
        "CoorDL": {'30': 4.14, '60': 10.031, '90': 17.88, '120': 26.716, '150': 37.33},
        "Shade": {'30': 4.14, '60': 10.031, '90': 17.88, '120': 26.716, '150': 37.33},
        r'$\bf{SUPER}$': {'30':3.315, '60': 7.604, '90':11.58, '120': 15.27, '150': 19.04}
    },
    "Time Breakdown": {
        "IO": { "CoorDL":  {'30': 4, '60': 4, '90': 4, '120': 4, '150': 4},
                "Shade": {'30': 4, '60': 4, '90': 4, '120': 4, '150': 4},
                r'$\bf{SUPER}$': {'30': 3, '60': 3, '90': 3, '120': 3, '150': 3}},
        "Transform": { "CoorDL": {'30': 34, '60': 34, '90': 34, '120': 34, '150': 34},
                        "Shade": {'30': 34, '60': 34, '90': 34, '120': 34, '150': 34},
                        r'$\bf{SUPER}$':{'30': 2, '60': 2, '90': 2, '120': 2, '150': 2}},
        "GPU": { "CoorDL": {'30': 62, '60': 62, '90': 62, '120': 62, '150': 62},
                "Shade": {'100': 62, '80': 62, '60': 62, '40': 62, '20': 62},
                r'$\bf{SUPER}$': {'100': 95, '80': 95, '60': 95, '40': 95, '20': 95}},
    },
}


x_tick_labels = [30,60,90,120,150]
x_label = 'Dataset Size (GB)'

for workload in workload_data:
    workload_name = workload
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11.4, 2.5))
    bar_width = 0.25
    workload_throuhgput = workload_data[workload]["Throughput"]
    dataset_sizes = list(workload_throuhgput["CoorDL"].keys())
    x = np.arange(len(dataset_sizes))

    ax1.plot(x, 
            workload_throuhgput['CoorDL'].values(),  
            label='CoorDL', 
            color=visual_map['CoorDL']['color'], 
            linestyle=visual_map['CoorDL']['linestyle'], 
            marker=visual_map['CoorDL']['marker'])
    ax1.plot(x,
            workload_throuhgput['Shade'].values(),  
            label='Shade', 
            color=visual_map['Shade']['color'], 
            linestyle=visual_map['Shade']['linestyle'], 
            marker=visual_map['Shade']['marker'])
    ax1.plot(x, 
            workload_throuhgput[r'$\bf{SUPER}$'].values(), 
            label=r'$\bf{SUPER}$', 
            color=visual_map[r'$\bf{SUPER}$']['color'], 
            linestyle=visual_map[r'$\bf{SUPER}$']['linestyle'], 
            marker=visual_map[r'$\bf{SUPER}$']['marker'])
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax1.set_ylabel('Throughput (samples/s)', fontsize=11)
    ax1.set_ylim(500, 1500)  # Adjusted limits for clarity
    # Get current y-limits
    current_ylim = ax1.get_ylim()
    # Add padding to the upper limit
    padding = 200
    ax1.set_ylim(current_ylim[0], current_ylim[1] + padding)  # Extend the upper limit
    # Optionally, adjust the legend placement if necessary
    ax1.legend(ncol=3, loc='upper center', fontsize=8)
    ax1.set_xticks(x + bar_width)  # Center ticks under the grouped bars
    ax1.set_xticklabels(x_tick_labels, fontsize=11)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_xlabel(x_label, fontsize=11)


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

    ax2.set_ylabel('Cost Per Epoch ($)', fontsize=11)
    ax2.set_ylim(0, 45)  # Adjust limits for clarity
    # Get current y-limits
    current_ylim = ax2.get_ylim()
    # Add padding to the upper limit
    padding = 0.25
    ax2.set_ylim(current_ylim[0], current_ylim[1] + padding)  # Extend the upper limit
    # Optionally, adjust the legend placement if necessary
    ax2.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))  # Apply custom formatter
    ax2.set_xticks(x + bar_width)  # Center ticks under the grouped bars
    ax2.set_xticklabels(x_tick_labels, fontsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.legend(ncol=3, loc='upper center', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plotting the bars for time breakdown
    io_times = workload_data[workload]["Time Breakdown"]["IO"]
    transform_times = workload_data[workload]["Time Breakdown"]["Transform"]
    gpu_times = workload_data[workload]["Time Breakdown"]["GPU"]
    group_labels = ['CoordL', 'Shade', 'Super']  # Data loader labels for each bar group
    for label, offset in zip(io_times, range(3)):
        io_values = list(io_times[label].values())
        ax3.bar(
            x + offset * bar_width,
            io_times[label].values(),
            width=bar_width,
            label='IO',
            color='#005250',
            hatch='----',
            edgecolor='black',
            alpha=0.8
        )
        transform_values = list(transform_times[label].values())
        ax3.bar(
            x + offset * bar_width,
            transform_times[label].values(),
            width=bar_width,
            bottom=list(io_times[label].values()),
            label='Transform',
            color='#FEA400',
            hatch='..',
            edgecolor='black',
            alpha=0.8

        )
        gpu_values = list(gpu_times[label].values())
        ax3.bar(
            x + offset * bar_width,
            gpu_times[label].values(),
            width=bar_width,
            bottom=[i + j for i, j in zip(io_times[label].values(), transform_times[label].values())],
            label='GPU',
            color='#4C8BB8',
            hatch='///////',
            edgecolor='black',
            alpha=0.8
        )
    ax3.set_ylabel('Time Breakdown (%)', fontsize=11)
    ax3.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=11)
    ax3.yaxis.set_major_formatter(FuncFormatter(percent_formatter))  # Apply custom formatter
    ax3.set_ylim(0, 100)  # Adjust limits for clarity
    current_ylim = ax3.get_ylim()
    padding = 20
    ax3.set_ylim(current_ylim[0], current_ylim[1] + padding)
    ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
    
    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels(x_tick_labels, fontsize=11)
     # Remove duplicate legend entries
    handles, labels = ax3.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l and labels.index(l) == i]
    ax3.legend(*zip(*unique), ncol=3, loc='upper center', fontsize=8)

   


plt.tight_layout()
plt.show()




