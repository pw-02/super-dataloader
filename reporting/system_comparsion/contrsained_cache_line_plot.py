from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
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
    'CoorDL': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
    'Shade': {'color': '#4C8BB8', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
    'LiData': {'color': '#FF7F0E', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
}
workload_data: Dict[str, Dict[str, float]] = {}
workload_data['ViT-32/Cifar10'] = {
    "Thouhgput" : { "CoorDL": {'10': 416, '25': 538, '50': 844, '75': 1482, '100': 1496},
                    "Shade": {'10': 481, '25': 573, '50': 926, '75': 1313, '100': 1312},
                    r'$\bf{SUPER}$': {'10': 1665, '25': 1665, '50': 1665, '75': 1665, '100': 1665}},
    "Cost" : { "CoorDL": {'10': 1.70, '25': 1.31, '50': 0.83, '75':0.47, '100': 0.47},
                "Shade": {'10': 1.47, '25': 1.23, '50': 0.76, '75': 0.54, '100': 0.54},
                r'$\bf{SUPER}$': {'10':0.408, '25': 0.408, '50':0.408, '75': 0.408, '100': 0.408}},
    "CacheHit" : { "CoorDL": {'10': 10, '25': 25, '50': 50, '75': 75, '100': 100},
                    "Shade": {'10': 24, '25': 49, '50': 81, '75': 100, '100': 100},
                    r'$\bf{SUPER}$': {'10': 100, '25': 100, '50': 100, '75': 100, '100': 100}},
    "Time Breakdown": {
        "IO": { "CoorDL": {'10': 76, '25': 68, '50': 50, '75': 11, '100': 10},
                "Shade": {'10': 68, '25': 62, '50': 37, '75': 6, '100': 6},
                r'$\bf{SUPER}$': {'10': 6, '25': 6, '50': 6, '75': 6, '100': 6}},
        "Transform": { "CoorDL": {'10': 3, '25': 4, '50': 5, '75': 2, '100': 9},
                        "Shade": {'10': 4, '25': 5, '50': 8, '75': 16, '100': 16},
                        r'$\bf{SUPER}$': {'10': 3.7, '25': 3.7, '50': 3.7, '75': 3.7, '100': 3.7}},
        "GPU": { "CoorDL": {'10': 24, '25': 32, '50': 50, '75': 89, '100': 90},
                "Shade": {'10': 28, '25': 34, '50': 55, '75': 78, '100': 78},
                r'$\bf{SUPER}$': {'10': 94, '25': 94, '50': 94, '75': 94, '100': 94}},
    }}
workload_data['ResNet-50/ImageNet'] = {
    "Thouhgput" : { "CoorDL": {'10': 552, '25': 617, '50': 766, '75': 1012, '100': 1314},
                    "Shade": {'10': 857, '25': 964, '50': 1218, '75': 1304, '100': 1304},
                    r'$\bf{SUPER}$': {'10': 1327, '25': 1327, '50': 1327, '75': 1327, '100': 1327}},
    "Cost" : { "CoorDL": {'10': 31.41, '25': 28.24, '50': 22.64, '75':17.15, '100': 13.2033},
                "Shade": {'10': 20.24, '25': 17.99, '50': 14.24, '75': 11.6670, '100': 11.6670},
                r'$\bf{SUPER}$': {'10':10.03, '25': 10.03, '50':10.03, '75': 10.03, '100': 10.03}},
    "CacheHit" : { "CoorDL": {'10': 10, '25': 25, '50': 50, '75': 75, '100': 100},
                    "Shade": {'10': 15, '25': 37, '50': 75, '75': 100, '100': 100},
                    r'$\bf{SUPER}$': {'10': 98, '25': 98, '50': 98, '75': 98, '100': 98}},
    "Time Breakdown": {
        "IO": { "CoorDL": {'10': 76, '25': 68, '50': 50, '75': 9, '100': 1},
                "Shade": {'10': 68, '25': 62, '50': 37, '75': 6, '100': 6},
                r'$\bf{SUPER}$': {'10': 6, '25': 6, '50': 6, '75': 6, '100': 6}},
        "Transform": { "CoorDL": {'10': 3, '25': 5, '50': 5, '75': 2, '100': 9},
                        "Shade": {'10': 4, '25': 5, '50': 8, '75': 16, '100': 16},
                        r'$\bf{SUPER}$': {'10': 6, '25': 6, '50': 6, '75': 6, '100': 6}},
        "GPU": { "CoorDL": {'10': 21, '25': 27, '50': 45, '75': 89, '100': 90},
                "Shade": {'10': 28, '25': 34, '50': 55, '75': 78, '100': 78},
                r'$\bf{SUPER}$': {'10': 88, '25': 88, '50': 88, '75': 88, '100': 88}},
    }}

for workload in workload_data:
    workload_name = workload

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17.75, 2.7))
    bar_width = 0.25
    workload_throuhgput = workload_data[workload]["Thouhgput"]
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

    # Set y-axis label and limits for throughput
    ax1.set_ylabel('Throughput (samples/s)', fontsize=11)
    ax1.set_ylim(0, 2000)  # Adjusted limits for clarity
    # Get current y-limits
    current_ylim = ax1.get_ylim()
    # Add padding to the upper limit
    padding = 100
    ax1.set_ylim(current_ylim[0], current_ylim[1] + padding)  # Extend the upper limit
    # Optionally, adjust the legend placement if necessary
    ax1.legend(loc='best')

    ax1.set_xticks(x + bar_width)  # Center ticks under the grouped bars
    ax1.set_xticklabels([10, 25,50,75,100], fontsize=11)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_xlabel('Dataset Size', fontsize=11)
    # ax1.legend()
    ax1.legend(ncol=3, loc='upper center', fontsize=9)

    # Plotting the bars for cost
    workload_cost = workload_data[workload]["Cost"]
    # ax2.bar(x, workload_cost['CoorDL'].values(), width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
    # ax2.bar(x + bar_width, workload_cost['Shade'].values(), width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
    # ax2.bar(x + 2 * bar_width, workload_cost[r'$\bf{SUPER}$'].values(), width=bar_width, label=r'$\bf{SUPER}$', color=visual_map[r'$\bf{SUPER}$']['color'], hatch=visual_map[r'$\bf{SUPER}$']['hatch'], edgecolor='black')
    # Set y-axis label and limits for cost

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

    ax2.set_ylabel('Training Cost Per Epoch ($)', fontsize=11)
    ax2.set_ylim(0, 40)  # Adjust limits for clarity
    # Get current y-limits
    current_ylim = ax2.get_ylim()
    # Add padding to the upper limit
    padding = 0.25
    ax2.set_ylim(current_ylim[0], current_ylim[1] + padding)  # Extend the upper limit
    # Optionally, adjust the legend placement if necessary
    ax2.legend(loc='best')

    ax2.set_xticks(x + bar_width)  # Center ticks under the grouped bars
    ax2.set_xticklabels([10, 25, 50, 75, 100], fontsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    ax1.set_xlabel('Dataset Size', fontsize=11)
    # ax2.legend()
    ax2.legend(ncol=3, loc='upper center', fontsize=9)

    #create a stacked bar chart for time breakdown
    # Plot 3: Time Breakdown (Stacked Bar)
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
        #  # Add Data Loader Label on Top of GPU Bar
        # for i, gpu_value in enumerate(gpu_values):
        #     total_height = io_values[i] + transform_values[i] + gpu_value
        #     ax3.text(
        #         x[i] + offset * bar_width,  # X-position aligned with the bar
        #         total_height + 2,  # Y-position slightly above the bar
        #          ['CoordL', 'Shade', 'Super'] ,  # Data loader label ('CoordL', 'Shade', 'Super')
        #         ha='center', va='bottom', fontsize=3, fontweight='bold'
        #     )

    ax3.set_ylabel('Time Breakdown (%)', fontsize=11)
    ax3.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=11)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    current_ylim = ax3.get_ylim()
    padding = 10
    ax3.set_ylim(current_ylim[0], current_ylim[1] + padding)
    ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])

    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels([10, 25, 50, 75, 100], fontsize=11)
    # Remove duplicate legend entries
    handles, labels = ax3.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l and labels.index(l) == i]
    ax3.legend(*zip(*unique), ncol=3, loc='upper center', fontsize=9)



    # # Plot 4: Cache Hit % (Stacked Bar)
    # worklaod_cache_hit = workload_data[workload]["CacheHit"]
    # ax4.bar(x, worklaod_cache_hit['CoorDL'].values(), width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
    # ax4.bar(x + bar_width, worklaod_cache_hit['Shade'].values(), width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
    # ax4.bar(x + 2 * bar_width, worklaod_cache_hit[r'$\bf{SUPER}$'].values(), width=bar_width, label=r'$\bf{SUPER}$', color=visual_map[r'$\bf{SUPER}$']['color'], hatch=visual_map[r'$\bf{SUPER}$']['hatch'], edgecolor='black')

    # # Set y-axis label and limits for cost
    # ax4.set_ylabel('Cache Hit %', fontsize=11)
    # # Get current y-limits
    # current_ylim = ax4.get_ylim()
    # # Add padding to the upper limit
    # padding = 15
    # ax4.set_ylim(current_ylim[0], current_ylim[1] + padding)  # Extend the upper limit
    # # Optionally, adjust the legend placement if necessary
    # ax4.legend(loc='best')
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    # ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])

    # ax4.set_xticks(x + bar_width)  # Center ticks under the grouped bars
    # ax4.set_xticklabels([10, 25, 50, 75, 100], fontsize=11)
    # ax4.tick_params(axis='y', labelsize=12)
    # ax4.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=11)
    # # ax2.legend()
    # ax4.legend(ncol=3, loc='upper center', fontsize=9)

    plt.tight_layout()
    plt.show()






# # Define the cache sizes (in percentage) and corresponding throughput values
# cache_sizes = [100, 75, 50, 25]  # Cache sizes as percentage of dataset
# throughput_your_solution = [1066, 1066, 1066, 1066]  # Your solution (constant throughput)
# throughput_baseline_1 = [971, 734, 658, 596]  # Baseline 1 throughput
# throughput_baseline_2 = [971, 830, 830, 741]  # Baseline 2 throughput

# # Define the cost values (for illustration, using the same values as throughput)
# cost_your_solution = [32, 32, 32, 32]  # Your solution (constant cost)
# cost_baseline_1 = [41, 47, 52, 58]  # Baseline 1 cost
# cost_baseline_2 = [41, 41, 41, 46]  # Baseline 2 cost

# # Create a new figure and set of axes for two subplots
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))

# # Plotting the lines for throughput
# ax1.plot(cache_sizes, throughput_your_solution, label=r'$\bf{SUPER}$', marker='o', linestyle='-', color=visual_map[r'$\bf{SUPER}$']['color'])
# ax1.plot(cache_sizes, throughput_baseline_1, label='CoorDL', marker='x', linestyle='--', color=visual_map['CoorDL']['color'])
# ax1.plot(cache_sizes, throughput_baseline_2, label='Shade', marker='s', linestyle='-.', color=visual_map['Shade']['color'])

# # Set y-axis label and limits for throughput
# ax1.set_ylabel('Throughput (samples/s)', fontsize=12)
# ax1.set_ylim(550, 1200)  # Adjusted limits for clarity
# ax1.set_xticks(cache_sizes)
# ax1.set_xticklabels([100, 75, 50, 25], fontsize=12)
# ax1.tick_params(axis='y', labelsize=12)
# ax1.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=12)
# ax1.legend()
# # ax1.set_title('Throughput vs. Cache Size', fontsize=14)

# # Plotting the lines for cost
# ax2.plot(cache_sizes, cost_your_solution, label=r'$\bf{SUPER}$', marker='o', linestyle='-', color=visual_map[r'$\bf{SUPER}$']['color'])
# ax2.plot(cache_sizes, cost_baseline_1, label='CoorDL', marker='x', linestyle='--', color=visual_map['CoorDL']['color'])
# ax2.plot(cache_sizes, cost_baseline_2, label='Shade', marker='s', linestyle='-.', color=visual_map['Shade']['color'])

# # Set y-axis label and limits for cost
# ax2.set_ylabel('Training Cost ($)', fontsize=12)
# ax2.set_ylim(20, 70)  # Adjust limits for clarity
# ax2.set_xticks(cache_sizes)
# ax2.set_xticklabels([100, 75, 50, 25], fontsize=12)
# ax2.tick_params(axis='y', labelsize=12)
# ax2.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=12)
# # ax2.legend(ncol=3, loc='upper center', fontsize=10)
# ax2.legend()

# # ax2.set_title('Cost vs. Cache Size', fontsize=14)

# # Adjust layout
# plt.tight_layout()

# # Display the plot
# plt.show()
