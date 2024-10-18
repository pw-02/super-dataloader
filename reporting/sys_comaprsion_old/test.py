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

# Define the cache sizes (in percentage) and corresponding throughput values
cache_sizes = [25, 50, 75, 100]  # Cache sizes as percentage of dataset
throughput_your_solution = [1066, 1066, 1066, 1066]  # Your solution (constant throughput)
throughput_baseline_1 = [596, 658, 734, 971]  # Baseline 1 throughput
throughput_baseline_2 = [741, 830, 830, 971]  # Baseline 2 throughput

# Define the cost values (for illustration, using example values)
cost_your_solution = [32, 32, 32, 32]  # Your solution (constant cost)
cost_baseline_1 = [58, 52, 47, 41]  # Baseline 1 cost
cost_baseline_2 = [46, 41, 41, 41]  # Baseline 2 cost

# Create a new figure and set of axes for four subplots (1 row, 4 columns)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(17.4, 2.5))  # Double the width for 4 plots

# Bar width and positions
bar_width = 0.25
x = np.arange(len(cache_sizes))

# Subplot 1: Throughput (Baseline 1, Baseline 2, Your Solution)
axs[0].bar(x, throughput_baseline_1, width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
axs[0].bar(x + bar_width, throughput_baseline_2, width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
axs[0].bar(x + 2 * bar_width, throughput_your_solution, width=bar_width, label=r'$\bf{SUPER}$', color=visual_map[r'$\bf{SUPER}$']['color'], hatch=visual_map[r'$\bf{SUPER}$']['hatch'], edgecolor='black')

# Set y-axis label and limits for throughput in subplot 1
axs[0].set_ylabel('Throughput (samples/s)', fontsize=12)
axs[0].set_ylim(550, 1200)  # Adjusted limits for clarity
axs[0].set_xticks(x + bar_width)  # Center ticks under the grouped bars
axs[0].set_xticklabels([25, 50, 75, 100], fontsize=12)
axs[0].tick_params(axis='y', labelsize=12)
axs[0].set_xlabel('Cache Size (% of Dataset)', fontsize=12)
axs[0].legend(ncol=3, loc='upper center', fontsize=10)

# Subplot 2: Cost (Baseline 1, Baseline 2, Your Solution)
axs[1].bar(x, cost_baseline_1, width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
axs[1].bar(x + bar_width, cost_baseline_2, width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
axs[1].bar(x + 2 * bar_width, cost_your_solution, width=bar_width, label=r'$\bf{SUPER}$', color=visual_map[r'$\bf{SUPER}$']['color'], hatch=visual_map[r'$\bf{SUPER}$']['hatch'], edgecolor='black')

# Set y-axis label and limits for cost in subplot 2
axs[1].set_ylabel('Training Cost ($)', fontsize=12)
axs[1].set_ylim(20, 65)  # Adjust limits for clarity
axs[1].set_xticks(x + bar_width)  # Center ticks under the grouped bars
axs[1].set_xticklabels([25, 50, 75, 100], fontsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[1].set_xlabel('Cache Size (% of Dataset)', fontsize=12)
axs[1].legend(ncol=3, loc='upper center', fontsize=10)

# Subplot 3: Duplicate of Throughput
axs[2].bar(x, throughput_baseline_1, width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
axs[2].bar(x + bar_width, throughput_baseline_2, width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
axs[2].bar(x + 2 * bar_width, throughput_your_solution, width=bar_width, label=r'$\bf{SUPER}$', color=visual_map[r'$\bf{SUPER}$']['color'], hatch=visual_map[r'$\bf{SUPER}$']['hatch'], edgecolor='black')

# Set y-axis label and limits for throughput in subplot 3
axs[2].set_ylabel('Throughput (samples/s)', fontsize=12)
axs[2].set_ylim(550, 1200)  # Adjusted limits for clarity
axs[2].set_xticks(x + bar_width)  # Center ticks under the grouped bars
axs[2].set_xticklabels([25, 50, 75, 100], fontsize=12)
axs[2].tick_params(axis='y', labelsize=12)
axs[2].set_xlabel('Cache Size (% of Dataset)', fontsize=12)
axs[2].legend(ncol=3, loc='upper center', fontsize=10)

# # Subplot 4: Duplicate of Cost
# axs[3].bar(x, cost_baseline_1, width=bar_width, label='CoorDL', color=visual_map['CoorDL']['color'], hatch=visual_map['CoorDL']['hatch'], edgecolor='black')
# axs[3].bar(x + bar_width, cost_baseline_2, width=bar_width, label='Shade', color=visual_map['Shade']['color'], hatch=visual_map['Shade']['hatch'], edgecolor='black')
# axs[3].bar(x + 2 * bar_width, cost_your_solution, width=bar_width, label=r'$\bf{SUPER}$', color=visual_map[r'$\bf{SUPER}$']['color'], hatch=visual_map[r'$\bf{SUPER}$']['hatch'], edgecolor='black')

# # Set y-axis label and limits for cost in subplot 4
# axs[3].set_ylabel('Training Cost ($)', fontsize=12)
# axs[3].set_ylim(20, 65)  # Adjust limits for clarity
# axs[3].set_xticks(x + bar_width)  # Center ticks under the grouped bars
# axs[3].set_xticklabels([25, 50, 75, 100], fontsize=12)
# axs[3].tick_params(axis='y', labelsize=12)
# axs[3].set_xlabel('Cache Size (% of Dataset)', fontsize=12)
# axs[3].legend(ncol=3, loc='upper center', fontsize=10)

# Adjust layout for better spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
