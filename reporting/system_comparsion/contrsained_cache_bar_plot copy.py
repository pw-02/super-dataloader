from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

# Define the visual map and figure data
visual_map = {
    r'$\bf{SUPER}$': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'CoorDL': {'color': '#FEA400', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},
    'Shade': {'color': '#4C8BB8', 'hatch': '////', 'edgecolor': 'black', 'alpha': 1.0},
    'LiData': {'color': '#FF7F0E', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1.0},
}

# Define the workloads

figure_data: Dict[str, Dict[str, float]] = {}

# Define the cache sizes (in percentage) and corresponding throughput values
cache_sizes = [10, 25, 50, 75, 100]  # Cache sizes as percentage of dataset
throughput_your_solution = [1665, 1665, 1665, 1665,1665]  # Your solution (constant throughput)
# throughput_baseline_1 = [971, 734, 658, 596]  # Baseline 1 throughput
# throughput_baseline_2 = [971, 830, 830, 741]  # Baseline 2 throughput

throughput_baseline_1 = [416, 538, 844, 1482,1496]  # Baseline 1 throughput
throughput_baseline_2 = [481, 573, 926, 1312, 1312]  # Baseline 2 throughput

# Define the cost values (for illustration, using example values)
cost_your_solution = [0.40, 0.40, 0.40, 0.40,0.40]  # Your solution (constant cost)
# cost_baseline_1 = [41, 47, 52, 58]  # Baseline 1 cost
# cost_baseline_2 = [41, 41, 41, 46]  # Baseline 2 cost
cost_baseline_1 = [1.70,1.314, 0.83, 0.47, 0.473]  # Baseline 1 cost
cost_baseline_2 = [1.46,1.23, 0.763, 0.538, 0.539]  # Baseline 2 cost

# Create a new figure and set of axes for two subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8.7, 2.7))

# Bar width and positions
bar_width = 0.25
x = np.arange(len(cache_sizes))

# Plotting the bars for throughput
ax1.bar(x, throughput_baseline_1, width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
ax1.bar(x + bar_width, throughput_baseline_2, width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
ax1.bar(x + 2 * bar_width, throughput_your_solution, width=bar_width, label=r'$\bf{SUPER}$', color=visual_map[r'$\bf{SUPER}$']['color'], hatch=visual_map[r'$\bf{SUPER}$']['hatch'], edgecolor='black')

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
ax1.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=11)
# ax1.legend()
ax1.legend(ncol=3, loc='upper center', fontsize=10)

# Plotting the bars for cost
ax2.bar(x, cost_baseline_1, width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
ax2.bar(x + bar_width, cost_baseline_2, width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
ax2.bar(x + 2 * bar_width, cost_your_solution, width=bar_width, label=r'$\bf{SUPER}$', color=visual_map[r'$\bf{SUPER}$']['color'], hatch=visual_map[r'$\bf{SUPER}$']['hatch'], edgecolor='black')

# Set y-axis label and limits for cost
ax2.set_ylabel('Training Cost Per Epoch ($)', fontsize=11)
ax2.set_ylim(0, 2)  # Adjust limits for clarity
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
ax2.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=11)
# ax2.legend()
ax2.legend(ncol=3, loc='upper center', fontsize=10)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()
