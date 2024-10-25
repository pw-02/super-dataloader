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
        "IO": { "CoorDL": 4,"Shade": 4, r'$\bf{SUPER}$': 3},
        "Transform": { "CoorDL": 34,"Shade":34,r'$\bf{SUPER}$':2},
        "GPU": { "CoorDL": 62, "Shade": 62, r'$\bf{SUPER}$': 95},
    },
    "Cost Breakdown": {
        "GPU": { "CoorDL":  {'30': 67, '60': 55, '90': 47, '120': 41, '150': 36},
                "Shade": {'30': 67, '60': 55, '90': 47, '120': 41, '150': 36},
                r'$\bf{SUPER}$': {'30': 68, '60': 68, '90': 68, '120': 68, '150': 68}},
        "Cache": { "CoorDL": {'30': 33, '60': 45, '90': 53, '120': 59, '150': 64},
                        "Shade": {'30': 33, '60': 45, '90': 53, '120': 59, '150': 64},
                        r'$\bf{SUPER}$':{'30': 32, '60': 32, '90': 32, '120': 32, '150': 32}},
    },
}


x_tick_labels = [30,60,90,120,150]
x_label = 'Dataset Size (GB)'

for workload in workload_data:
    workload_name = workload
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11.4, 2.3))
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

    ax2.set_ylabel('Epoch Cost ($)', fontsize=11)
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
    workload_time_breakdown = workload_data[workload]["Time Breakdown"]
    io_values = [workload_time_breakdown['IO'][workload] for workload in workload_time_breakdown['IO']]
    transform_values = [workload_time_breakdown['Transform'][workload] for workload in workload_time_breakdown['Transform']]
    gpu_values = [workload_time_breakdown['GPU'][workload] for workload in workload_time_breakdown['GPU']]
    labels = list(workload_time_breakdown['IO'].keys())
    # Create the stacked bar chart
    bar_width = 0.5
    ax3.bar(labels, io_values, width=bar_width, label='IO %', color=visual_map['io']['color'], hatch=visual_map['io']['hatch'])
    ax3.bar(labels, transform_values, width=bar_width, bottom=io_values, label='Transform %', color=visual_map['transform']['color'], hatch=visual_map['transform']['hatch'])
    ax3.bar(labels, gpu_values, bottom=np.array(io_values) + np.array(transform_values), width=bar_width, label='GPU %', color=visual_map['gpu']['color'], hatch=visual_map['gpu']['hatch'])
    ax3.set_ylabel('Time Breakdown (%)', fontsize=11)
    ax3.yaxis.set_major_formatter(FuncFormatter(percent_formatter))  # Apply custom formatter
    ax3.set_ylim(0, 100)  # Adjust limits for clarity
    # Get current y-limits
    current_ylim = ax3.get_ylim()
    # Add padding to the upper limit
    padding = 20
    ax3.set_ylim(current_ylim[0], current_ylim[1] + padding)
    ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])

    ax3.legend(loc='upper center', ncols=3, fontsize=8)


    # # Plotting the bars for cost breakdown
    # gpu_costs = workload_data[workload]["Cost Breakdown"]["GPU"]
    # cache_costs = workload_data[workload]["Cost Breakdown"]["Cache"]

    # # io_times = workload_data[workload]["Time Breakdown"]["IO"]
    # # transform_times = workload_data[workload]["Time Breakdown"]["Transform"]
    # # gpu_times = workload_data[workload]["Time Breakdown"]["GPU"]
    # group_labels = ['CoordL', 'Shade', 'Super']  # Data loader labels for each bar group
    # for label, offset in zip(cache_costs, range(3)):
    #     io_values = list(cache_costs[label].values())
    #     ax4.bar(
    #         x + offset * bar_width,
    #         cache_costs[label].values(),
    #         width=bar_width,
    #         label='IO',
    #         color='#005250',
    #         hatch='----',
    #         edgecolor='black',
    #         alpha=0.8
    #     )
    #     transform_values = list(gpu_costs[label].values())
    #     ax4.bar(
    #         x + offset * bar_width,
    #         gpu_costs[label].values(),
    #         width=bar_width,
    #         bottom=list(cache_costs[label].values()),
    #         label='Transform',
    #         color='#FEA400',
    #         hatch='..',
    #         edgecolor='black',
    #         alpha=0.8

    #     )
        
    # ax4.set_ylabel('Cost Breakdown (%)', fontsize=11)
    # ax4.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=11)
    # ax4.yaxis.set_major_formatter(FuncFormatter(percent_formatter))  # Apply custom formatter
    # ax4.set_ylim(0, 100)  # Adjust limits for clarity
    # current_ylim = ax3.get_ylim()
    # padding = 20
    # ax4.set_ylim(current_ylim[0], current_ylim[1] + padding)
    # ax4.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
    
    # ax4.set_xticks(x + bar_width)
    # ax4.set_xticklabels(x_tick_labels, fontsize=11)
    #  # Remove duplicate legend entries
    # handles, labels = ax4.get_legend_handles_labels()
    # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l and labels.index(l) == i]
    # ax4.legend(*zip(*unique), ncol=3, loc='upper center', fontsize=8)

plt.tight_layout()
plt.show()




