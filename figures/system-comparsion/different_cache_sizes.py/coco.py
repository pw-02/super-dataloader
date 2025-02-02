from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

workload_name = 'Albef/COCO'
title = 'DisDP'
baseline = 'CoorDL'

def percent_formatter(x, pos):
    return f'{int(x)}%'

# Custom formatter to display Y-tick labels in 'K'
def thousands_formatter(x, pos):
    return f'{x / 1000:.1f}K'
# Custom formatter to add a dollar sign to Y-ticks in the cost subplot
def dollar_formatter(x, pos):
    return f'${x:.0f}'

visual_map = {
    r'$\bf{DisDP}$': {'color': '#005250', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
    baseline: {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
}


workload_data: Dict[str, Dict[str, float]] = {}

workload_data[workload_name] =  {
    "Thoughgput" : { "CoorDL": { '80': 132.69776935, '60': 111.69776935, '40': 96.69776935, '20': 70.69776935},
                    r'$\bf{DisDP}$': { '80': 193.2153121, '60': 193.2153121, '40': 193.2153121, '20': 193.2153121}},
    "Cost" : { "CoorDL": { '80': 5.5, '60': 6.956807873, '40':8.95, '20':10.77},
                r'$\bf{DisDP}$': { '80': 4.778720882, '60':4.778720882, '40': 4.778720882, '20': 4.778720882}},
    "CacheHit" : { "CoorDL": { '80': 25, '60': 50, '40': 75, '20': 100},
                    r'$\bf{DisDP}$': { '80': 100, '60': 100, '40': 100, '20': 100}},
    "Time Breakdown": {
        "IO": { "CoorDL":  { '80': 10.0716939, '60': 12.0716939, '40': 14.0716939, '20': 15.0716939},
                r'$\bf{DisDP}$': { '80': 10, '60': 10, '40': 10, '20': 10}},
        "Transform": { "CoorDL": { '80': 56, '60': 56, '40': 56, '20': 56},
                        r'$\bf{DisDP}$': {'80': 3, '60': 3, '40': 3, '20': 3}},
        "GPU": { "CoorDL": { '80': 33.9283061, '60': 32, '40': 29.9283061, '20': 28.9283061},
                r'$\bf{DisDP}$': { '80': 88, '60': 88, '40': 88, '20': 88}},
    }}

x_tick_lables = ['80%', '60%', '40%', '20%']
x_label = 'Baseline Cache Size as a % of Dataset'
barwidth = 0.35  # Width of the bars

for workload in workload_data:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(19.5, 3.4))  # Adjusted to fit within single column
    workload_throuhgput = workload_data[workload]["Thoughgput"]
    # dataset_sizes = list(workload_throuhgput["CoorDL"].keys())
    # x = np.arange(len(dataset_sizes))
    x = np.arange(len(x_tick_lables))  # X-axis positions

    # Bar plot for CoorDL
    ax1.bar(x - barwidth/2, workload_throuhgput[baseline].values(),barwidth,  label=baseline, hatch='//', color=visual_map[baseline]['color'], edgecolor='black')
    # Bar plot for DisDP
    ax1.bar(x + barwidth/2, workload_throuhgput[r'$\bf{DisDP}$'].values(),barwidth, hatch='\\\\', label=r'$\bf{DisDP}$', color=visual_map[r'$\bf{DisDP}$']['color'],edgecolor='black')
    ax1.set_ylabel('Samples/s')
    ax1.set_xlabel('Baseline Cache Size as % of the Dataset')
    ax1.set_xticks(x)
    ax1.set_ylim(0, 300) 
    ax1.set_title(f'Throughput of {title} vs. Baseline ({workload_name})', fontsize=12)
    ax1.set_xticklabels(x_tick_lables)
    ax1.legend(ncol=2, loc='upper center', fontsize=10)
    ax1.grid(axis='y', linestyle='--', linewidth=0.5)

    #label speedups
    for i, sh in enumerate(workload_throuhgput[r'$\bf{DisDP}$'].values()):
        speedup = sh/list(workload_throuhgput[baseline].values())[i]
        if i == 0:
            ax1.text(x[i] + barwidth/2, sh + 15, f'x{speedup:.1f}\nspeedup', ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')
        else:
            ax1.text(x[i] + barwidth/2, sh + 15, f'x{speedup:.1f}' , ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')
   
    # Plotting the bars for cost
    workload_cost = workload_data[workload]["Cost"]
    ax2.bar(x - barwidth/2, workload_cost[baseline].values(),barwidth,  label=baseline, hatch='//', color=visual_map[baseline]['color'], edgecolor='black')
    ax2.bar(x + barwidth/2, workload_cost[r'$\bf{DisDP}$'].values(),barwidth, hatch='\\\\', label=r'$\bf{DisDP}$', color=visual_map[r'$\bf{DisDP}$']['color'],edgecolor='black')
    ax2.set_ylabel('Training Cost Per Epoch ($)')
    ax2.set_xlabel(x_label)
    ax2.set_xticks(x)
    # ax2.set_ylim(0, 1850) 
    ax2.set_title(f'Cost of {title} vs. Baseline ({workload_name})', fontsize=12)
    ax2.set_xticklabels(x_tick_lables)
    ax2.legend(ncol=1, loc='upper left', fontsize=10)
    ax2.grid(axis='y', linestyle='--', linewidth=0.5)
    ax2.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))  # Apply custom formatter

    #label savings
    for i, sh in enumerate(workload_cost[r'$\bf{DisDP}$'].values()):
        savings = list(workload_cost[baseline].values())[i]/sh
        if i == 0:
            ax2.text(x[i] + barwidth/2 +0.1, sh + 1.75, f'x{savings:.1f} \n savings', ha='center', va='top', fontsize=11, color='black', fontweight='bold')
        else:
            ax2.text(x[i] + barwidth/2 +0.05, sh + 1.2, f'x{savings:.1f}', ha='center', va='top', fontsize=11, color='black', fontweight='bold')

    #create a stacked bar chart for time breakdown
    # Plot 3: Time Breakdown (Stacked Bar)

    io_times = workload_data[workload]["Time Breakdown"]["IO"]
    transform_times = workload_data[workload]["Time Breakdown"]["Transform"]
    gpu_times = workload_data[workload]["Time Breakdown"]["GPU"]
    group_labels = [baseline, 'DisDP']  # Data loader labels for each bar group

    for label, offset in zip(io_times, range(3)):
        io_values = list(io_times[label].values())
        transform_values = list(transform_times[label].values())
        gpu_values = list(gpu_times[label].values())
         # Bar for IO
        io_bar = ax3.bar(
            x + offset * barwidth,
            io_values,
            width=barwidth,
            label='IO',
            color='#005250',
            hatch='----',
            edgecolor='black',
            # alpha=0.8
        )
        # Bar for Transform
        transform_bar = ax3.bar(
            x + offset * barwidth,
            transform_values,
            width=barwidth,
            bottom=io_values,
            label='Transform',
            color='#FEA400',
            hatch='..',
            edgecolor='black',
            # alpha=0.8
        )
        # Bar for GPU
        gpu_bar = ax3.bar(
            x + offset * barwidth,
            gpu_values,
            width=barwidth,
            bottom=[i + j for i, j in zip(io_values, transform_values)],
            label='GPU',
            color='#4C8BB8',
            hatch='///////',
            edgecolor='black',
            # alpha=0.8
        )

        # Adding labels on top of each stack
    for i, (io, transform, gpu) in enumerate(zip(io_values, transform_values, gpu_values)):
        # Calculate the top of each stack
        total_height = io + transform + gpu

        
        label = "CoordL|DisDP "
      

        # Add label above the bar (adjust y-position as needed)
        ax3.text(
            x[i] + offset * barwidth - 0.18,  # X position (left of the bar)
            total_height, 
            f'{label}',  # Format the label
            ha='center', 
            va='bottom', 
            fontsize=10, 
            fontweight='bold'
        )

    ax3.set_title(f'Time Breakdown of {title} vs. Baseline ({workload_name})', fontsize=12)
    ax3.set_ylabel('Time Breakdown (%)', fontsize=12)
    ax3.set_xlabel(x_label)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    current_ylim = ax3.get_ylim()
    padding = 5
    ax3.set_ylim(current_ylim[0], current_ylim[1] + padding)
    ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
    xpadding = 0.1
    current_xlim = ax3.get_xlim()
    ax3.set_xlim(current_xlim[0], current_xlim[1] + xpadding)
    ax3.set_xticks(x + barwidth)
    ax3.set_xticklabels(x_tick_lables, fontsize=12)
    # Remove duplicate legend entries
    handles, labels = ax3.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l and labels.index(l) == i]
    ax3.legend(*zip(*unique), ncol=1, loc='center right', fontsize=10)



    # # Plot 4: Cache Hit % (Stacked Bar)
    # worklaod_cache_hit = workload_data[workload]["CacheHit"]
    # ax4.bar(x, worklaod_cache_hit[baseline].values(), width=bar_width, label=baseline, color=visual_map[baseline]['color'], hatch=visual_map[baseline]['hatch'], edgecolor='black')
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
# ax1.plot(cache_sizes, throughput_baseline_1, label=baseline, marker='x', linestyle='--', color=visual_map[baseline]['color'])
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
# ax2.plot(cache_sizes, cost_baseline_1, label=baseline, marker='x', linestyle='--', color=visual_map[baseline]['color'])
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
