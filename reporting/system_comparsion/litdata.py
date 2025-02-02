from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

def percent_formatter(x, pos):
    return f'{int(x)}%'

# Custom formatter to display Y-tick labels in 'K'
def thousands_formatter(x, pos):
    return f'{x / 1000:.1f}K'
# Custom formatter to add a dollar sign to Y-ticks in the cost subplot
def dollar_formatter(x, pos):
    return f'${x:.0f}'

# # Define the visual map and figure data
# visual_map = {
#     r'$\bf{DisDP}$': {'color': '#005250', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
#     'CoorDL': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
#     'Shade': {'color': '#4C8BB8', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
#     'LitData': {'color': '#FF7F0E', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
# }
#81ACCD
visual_map = {
    r'$\bf{DisDP}$': {'color': '#005250', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
    'CoorDL': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
    'Shade': {'color': '#73A3C7', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
    'LitData': {'color': '#FF7F0E', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
}
workload_data: Dict[str, Dict[str, float]] = {}

workload_data['Pythia14m/OWT'] = {
    "Thoughgput" : { "LitData": {'80': 17.94533576, '60': 17.94533576, '40': 17.94533576, '20': 17.94533576},
                    r'$\bf{DisDP}$': { '80': 34.907, '60': 34.907, '40': 34.907, '20': 34.907}},
    "Cost" : { "LitData": { '80': 1.84678, '60': 1.84678, '40':1.84678, '20':1.84678},
                r'$\bf{DisDP}$': {'80': 0.531078482, '60':0.531078482, '40': 0.531078482, '20': 0.531078482}},
    # "CacheHit" : { "CoorDL": {'80': 25, '60': 50, '40': 75, '20': 100},
    #                 "Shade": {'80': 49, '60': 81, '40': 100, '20': 100},
    #                 r'$\bf{DisDP}$': {'80': 100, '60': 100, '40': 100, '20': 100}},
    "Time Breakdown": {
        "IO": { "LitData": { '80': 1.6927691, '60':1.6927691, '40': 1.6927691, '20': 1.6927691},
                r'$\bf{DisDP}$': { '80': 1.4359577, '60': 1.4359577, '40': 1.4359577, '20': 1.4359577}},
        "Transform": { "LitData": { '80': 50.6356277, '60': 50.6356277, '40': 50.6356277, '20': 50.6356277},
                        r'$\bf{DisDP}$': {'80': 8.2940553, '60': 8.2940553, '40': 8.2940553, '20': 8.2940553}},
        "GPU": { "LitData": { '80':  47.6716032, '60':  47.6716032, '40':  47.6716032, '20':  47.6716032},
                r'$\bf{DisDP}$': {'80': 90.9346479, '60': 90.9346479, '40': 90.9346479, '20': 90.9346479}},
    }}


x_tick_lables = [80,60,40,20]
x_label = 'Baseline Cache Size (% of Dataset)'

for workload in workload_data:
    workload_name = workload
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16.4, 3.5))

    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(4, 8))
    bar_width = 0.25
    workload_throuhgput = workload_data[workload]["Thoughgput"]
    dataset_sizes = list(workload_throuhgput["LitData"].keys())
    x = np.arange(len(dataset_sizes))

    ax1.plot(x, 
            workload_throuhgput['LitData'].values(),  
            label='LitData', 
            color=visual_map['LitData']['color'], 
            linestyle=visual_map['LitData']['linestyle'], 
            marker=visual_map['LitData']['marker'])
   
    ax1.plot(x, 
            workload_throuhgput[r'$\bf{DisDP}$'].values(), 
            label=r'$\bf{DisDP}$', 
            color=visual_map[r'$\bf{DisDP}$']['color'], 
            linestyle=visual_map[r'$\bf{DisDP}$']['linestyle'], 
            marker=visual_map[r'$\bf{DisDP}$']['marker'])

    # Set y-axis label and limits for throughput
    ax1.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax1.set_ylim(0, 50)  # Adjusted limits for clarity
    # Get current y-limits
    current_ylim = ax1.get_ylim()
    # Add padding to the upper limit
    padding = 2
    ax1.set_ylim(current_ylim[0], current_ylim[1] + padding)  # Extend the upper limit
    # Optionally, adjust the legend placement if necessary

    ax1.set_xticks(x + bar_width)  # Center ticks under the grouped bars
    ax1.set_xticklabels(x_tick_lables, fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_xlabel(x_label, fontsize=12)
    # ax1.legend()
    ax1.legend(ncol=3, loc='upper center', fontsize=12)
    ax1.set_title('Throughput vs. Cache Size (Pythia14m/OWT)', fontsize=12)
    # Plotting the bars for cost
    workload_cost = workload_data[workload]["Cost"]
    # ax2.bar(x, workload_cost['CoorDL'].values(), width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
    # ax2.bar(x + bar_width, workload_cost['Shade'].values(), width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
    # ax2.bar(x + 2 * bar_width, workload_cost[r'$\bf{DisDP}$'].values(), width=bar_width, label=r'$\bf{DisDP}$', color=visual_map[r'$\bf{DisDP}$']['color'], hatch=visual_map[r'$\bf{DisDP}$']['hatch'], edgecolor='black')
    # Set y-axis label and limits for cost

    ax2.plot(x, 
            workload_cost['LitData'].values(),  
            label='LitData', 
            color=visual_map['LitData']['color'], 
            linestyle=visual_map['LitData']['linestyle'], 
            marker=visual_map['LitData']['marker'])
    ax2.plot(x, 
            workload_cost[r'$\bf{DisDP}$'].values(), 
            label=r'$\bf{DisDP}$', 
            color=visual_map[r'$\bf{DisDP}$']['color'], 
            linestyle=visual_map[r'$\bf{DisDP}$']['linestyle'], 
            marker=visual_map[r'$\bf{DisDP}$']['marker'])

    ax2.set_ylabel('Training Cost Per Epoch ($)', fontsize=12)
    ax2.set_ylim(0, 3)  # Adjust limits for clarity
    # Get current y-limits
    current_ylim = ax2.get_ylim()
    # Add padding to the upper limit
    padding = 0.25
    ax2.set_ylim(current_ylim[0], current_ylim[1] + padding)  # Extend the upper limit
    # Optionally, adjust the legend placement if necessary
    ax2.legend(loc='best')
    ax2.set_title('Cost vs. Cache Size (Pythia14m/OWT)', fontsize=12)

    ax2.set_xticks(x + bar_width)  # Center ticks under the grouped bars
    ax2.set_xticklabels(x_tick_lables, fontsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_xlabel(x_label, fontsize=12)
    # ax2.legend()
    ax2.legend(ncol=3, loc='upper center', fontsize=12)

    #create a stacked bar chart for time breakdown
    # Plot 3: Time Breakdown (Stacked Bar)
    io_times = workload_data[workload]["Time Breakdown"]["IO"]
    transform_times = workload_data[workload]["Time Breakdown"]["Transform"]
    gpu_times = workload_data[workload]["Time Breakdown"]["GPU"]
    group_labels = ['CoordL', 'Shade', 'DisDP']  # Data loader labels for each bar group

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
        #          ['CoordL', 'Shade', 'DisDP'] ,  # Data loader label ('CoordL', 'Shade', 'DisDP')
        #         ha='center', va='bottom', fontsize=3, fontweight='bold'
        #     )
    ax3.set_title('Time Breakdown vs. Cache Size (Pythia14m/OWT)', fontsize=12)

    ax3.set_ylabel('Time Breakdown (%)', fontsize=12)
    ax3.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    current_ylim = ax3.get_ylim()
    padding = 10
    ax3.set_ylim(current_ylim[0], current_ylim[1] + padding)
    ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])

    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels(x_tick_lables, fontsize=12)
    # Remove duplicate legend entries
    handles, labels = ax3.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l and labels.index(l) == i]
    ax3.legend(*zip(*unique), ncol=3, loc='upper center', fontsize=12)



    # # Plot 4: Cache Hit % (Stacked Bar)
    # worklaod_cache_hit = workload_data[workload]["CacheHit"]
    # ax4.bar(x, worklaod_cache_hit['CoorDL'].values(), width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
    # ax4.bar(x + bar_width, worklaod_cache_hit['Shade'].values(), width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
    # ax4.bar(x + 2 * bar_width, worklaod_cache_hit[r'$\bf{DisDP}$'].values(), width=bar_width, label=r'$\bf{DisDP}$', color=visual_map[r'$\bf{DisDP}$']['color'], hatch=visual_map[r'$\bf{DisDP}$']['hatch'], edgecolor='black')

    # # Set y-axis label and limits for cost
    # ax4.set_ylabel('Cache Hit %', fontsize=12)
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
    # ax4.set_xticklabels([10, 25, 50, 75, 100], fontsize=12)
    # ax4.tick_params(axis='y', labelsize=12)
    # ax4.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=12)
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
# ax1.plot(cache_sizes, throughput_your_solution, label=r'$\bf{DisDP}$', marker='o', linestyle='-', color=visual_map[r'$\bf{DisDP}$']['color'])
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
# ax2.plot(cache_sizes, cost_your_solution, label=r'$\bf{DisDP}$', marker='o', linestyle='-', color=visual_map[r'$\bf{DisDP}$']['color'])
# ax2.plot(cache_sizes, cost_baseline_1, label='CoorDL', marker='x', linestyle='--', color=visual_map['CoorDL']['color'])
# ax2.plot(cache_sizes, cost_baseline_2, label='Shade', marker='s', linestyle='-.', color=visual_map['Shade']['color'])

# # Set y-axis label and limits for cost
# ax2.set_ylabel('Training Cost ($)', fontsize=12)
# ax2.set_ylim(20, 70)  # Adjust limits for clarity
# ax2.set_xticks(cache_sizes)
# ax2.set_xticklabels([100, 75, 50, 25], fontsize=12)
# ax2.tick_params(axis='y', labelsize=12)
# ax2.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=12)
# # ax2.legend(ncol=3, loc='upper center', fontsize=12)
# ax2.legend()

# # ax2.set_title('Cost vs. Cache Size', fontsize=14)

# # Adjust layout
# plt.tight_layout()

# # Display the plot
# plt.show()
