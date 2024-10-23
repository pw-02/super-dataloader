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
# workload_data['ViT-32/Cifar10'] = {
#     "Thouhgput" : { "CoorDL": {'30': 416, '60': 538, '90': 844, '120': 1482, '150': 1496},
#                     "Shade": {'30': 481, '60': 573, '90': 926, '120': 1313, '150': 1312},
#                     r'$\bf{SUPER}$': {'30': 1665, '60': 1665, '90': 1665, '120': 1665, '150': 1665}},
#     "Cost" : { "CoorDL": {'30': 1.70, '60': 1.31, '90': 0.83, '120':0.47, '150': 0.47},
#                 "Shade": {'30': 1.47, '60': 1.23, '90': 0.76, '120': 0.54, '150': 0.54},
#                 r'$\bf{SUPER}$': {'30':0.408, '60': 0.408, '90':0.408, '120': 0.408, '150': 0.408}},
#     "CacheHit" : { "CoorDL": {'30': 10, '60': 25, '90': 50, '120': 75, '150': 100},
#                     "Shade": {'30': 24, '60': 49, '90': 81, '120': 100, '150': 100},
#                     r'$\bf{SUPER}$': {'30': 100, '60': 100, '90': 100, '120': 100, '150': 100}},
#     "Time Breakdown": {
#         "IO": { "CoorDL": {'30': 76, '60': 68, '90': 50, '120': 11, '150': 10},
#                 "Shade": {'30': 68, '60': 62, '90': 37, '120': 6, '150': 6},
#                 r'$\bf{SUPER}$': {'30': 6, '60': 6, '90': 6, '120': 6, '150': 6}},
#         "Transform": { "CoorDL": {'30': 3, '60': 4, '90': 5, '120': 2, '150': 9},
#                         "Shade": {'30': 4, '60': 5, '90': 8, '120': 16, '150': 16},
#                         r'$\bf{SUPER}$': {'30': 3.7, '60': 3.7, '90': 3.7, '120': 3.7, '150': 3.7}},
#         "GPU": { "CoorDL": {'30': 24, '60': 32, '90': 50, '120': 89, '150': 90},
#                 "Shade": {'30': 28, '60': 34, '90': 55, '120': 78, '150': 78},
#                 r'$\bf{SUPER}$': {'30': 94, '60': 94, '90': 94, '120': 94, '150': 94}},
#     }}
workload_data['ResNet-50/ImageNet'] = {
    "Thouhgput" : { "CoorDL": {'30': 1327, '60': 1327, '90': 1327, '120': 1327, '150': 1327},
                     "Shade": {'30': 1325, '60': 1327, '90': 1322, '120': 1328, '150': 1328},
                    r'$\bf{SUPER}$': {'30': 1435, '60': 1435, '90': 1435, '120': 1435, '150': 1435}},
    "Cost" : { 
        "CoorDL": {'30': 4.29, '60': 8.73, '90': 13.13, '120': 17.55, '150': 21.97},
        "Shade": {'30': 4.29, '60': 8.73, '90': 13.13, '120': 17.55, '150': 21.97},
        r'$\bf{SUPER}$': {'30':2.429, '60': 4.858, '90':7.287, '120': 9.716, '150': 12.145}
        },

    # "Time Breakdown": {
    #     "IO": { "CoorDL": {'30': 76, '60': 68, '90': 50, '120': 9, '150': 1},
    #             # "Shade": {'30': 68, '60': 62, '90': 37, '120': 6, '150': 6},
    #             r'$\bf{SUPER}$': {'30': 6, '60': 6, '90': 6, '120': 6, '150': 6}},
    #     "Transform": { "CoorDL": {'30': 3, '60': 5, '90': 5, '120': 2, '150': 9},
    #                     # "Shade": {'30': 4, '60': 5, '90': 8, '120': 16, '150': 16},
    #                     r'$\bf{SUPER}$': {'30': 6, '60': 6, '90': 6, '120': 6, '150': 6}},
    #     "GPU": { "CoorDL": {'30': 21, '60': 27, '90': 45, '120': 89, '150': 90},
    #             # "Shade": {'30': 28, '60': 34, '90': 55, '120': 78, '150': 78},
    #             r'$\bf{SUPER}$': {'30': 88, '60': 88, '90': 88, '120': 88, '150': 88}},
    # }
    }


x_tick_lables = [30,60,90,120,150]
x_label = 'Dataset Size (GB)'
for workload in workload_data:
    workload_name = workload

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7.75, 2.7))
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
    ax1.set_xticklabels(x_tick_lables, fontsize=11)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_xlabel(x_label, fontsize=11)
    # ax1.legend()
    ax1.legend(ncol=3, loc='upper center', fontsize=8)

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
    ax2.set_xticklabels(x_tick_lables, fontsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xlabel(x_label, fontsize=11)
    # ax2.legend()
    ax2.legend(ncol=3, loc='upper center', fontsize=8)

    # #create a stacked bar chart for time breakdown
    # # Plot 3: Time Breakdown (Stacked Bar)
    # io_times = workload_data[workload]["Time Breakdown"]["IO"]
    # transform_times = workload_data[workload]["Time Breakdown"]["Transform"]
    # gpu_times = workload_data[workload]["Time Breakdown"]["GPU"]
    # group_labels = ['CoordL', 'Shade', 'Super']  # Data loader labels for each bar group

    # for label, offset in zip(io_times, range(3)):
    #     io_values = list(io_times[label].values())
    #     ax3.bar(
    #         x + offset * bar_width,
    #         io_times[label].values(),
    #         width=bar_width,
    #         label='IO',
    #         color='#005250',
    #         hatch='----',
    #         edgecolor='black',
    #         alpha=0.8
    #     )
    #     transform_values = list(transform_times[label].values())
    #     ax3.bar(
    #         x + offset * bar_width,
    #         transform_times[label].values(),
    #         width=bar_width,
    #         bottom=list(io_times[label].values()),
    #         label='Transform',
    #         color='#FEA400',
    #         hatch='..',
    #         edgecolor='black',
    #         alpha=0.8

    #     )
    #     gpu_values = list(gpu_times[label].values())
    #     ax3.bar(
    #         x + offset * bar_width,
    #         gpu_times[label].values(),
    #         width=bar_width,
    #         bottom=[i + j for i, j in zip(io_times[label].values(), transform_times[label].values())],
    #         label='GPU',
    #         color='#4C8BB8',
    #         hatch='///////',
    #         edgecolor='black',
    #         alpha=0.8
    #     )
        #  # Add Data Loader Label on Top of GPU Bar
        # for i, gpu_value in enumerate(gpu_values):
        #     total_height = io_values[i] + transform_values[i] + gpu_value
        #     ax3.text(
        #         x[i] + offset * bar_width,  # X-position aligned with the bar
        #         total_height + 2,  # Y-position slightly above the bar
        #          ['CoordL', 'Shade', 'Super'] ,  # Data loader label ('CoordL', 'Shade', 'Super')
        #         ha='center', va='bottom', fontsize=3, fontweight='bold'
        #     )

    # ax3.set_ylabel('Time Breakdown (%)', fontsize=11)
    # ax3.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=11)
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    # current_ylim = ax3.get_ylim()
    # padding = 10
    # ax3.set_ylim(current_ylim[0], current_ylim[1] + padding)
    # ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])

    # ax3.set_xticks(x + bar_width)
    # ax3.set_xticklabels([10, 25, 50, 75, 100], fontsize=11)
    # # Remove duplicate legend entries
    # handles, labels = ax3.get_legend_handles_labels()
    # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l and labels.index(l) == i]
    # ax3.legend(*zip(*unique), ncol=3, loc='upper center', fontsize=9)



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