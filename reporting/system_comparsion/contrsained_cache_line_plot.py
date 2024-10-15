from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Define the visual map and figure data
visual_map = {
    r'$\bf{SUPER}$': {'color': '#005250', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
    'CoorDL': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
    'Shade': {'color': '#4C8BB8', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
    'LiData': {'color': '#FF7F0E', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
}

# Define the cache sizes (in percentage) and corresponding throughput values
cache_sizes = [100, 75, 50, 25]  # Cache sizes as percentage of dataset
throughput_your_solution = [1066, 1066, 1066, 1066]  # Your solution (constant throughput)
throughput_baseline_1 = [971, 734, 658, 596]  # Baseline 1 throughput
throughput_baseline_2 = [971, 830, 830, 741]  # Baseline 2 throughput

# Define the cost values (for illustration, using the same values as throughput)
cost_your_solution = [32, 32, 32, 32]  # Your solution (constant cost)
cost_baseline_1 = [41, 47, 52, 58]  # Baseline 1 cost
cost_baseline_2 = [41, 41, 41, 46]  # Baseline 2 cost

# Create a new figure and set of axes for two subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.5))

# Plotting the lines for throughput
ax1.plot(cache_sizes, throughput_your_solution, label=r'$\bf{SUPER}$', marker='o', linestyle='-', color=visual_map[r'$\bf{SUPER}$']['color'])
ax1.plot(cache_sizes, throughput_baseline_1, label='CoorDL', marker='x', linestyle='--', color=visual_map['CoorDL']['color'])
ax1.plot(cache_sizes, throughput_baseline_2, label='Shade', marker='s', linestyle='-.', color=visual_map['Shade']['color'])

# Set y-axis label and limits for throughput
ax1.set_ylabel('Throughput (samples/s)', fontsize=12)
ax1.set_ylim(550, 1200)  # Adjusted limits for clarity
ax1.set_xticks(cache_sizes)
ax1.set_xticklabels([100, 75, 50, 25], fontsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=12)
ax1.legend()
# ax1.set_title('Throughput vs. Cache Size', fontsize=14)

# Plotting the lines for cost
ax2.plot(cache_sizes, cost_your_solution, label=r'$\bf{SUPER}$', marker='o', linestyle='-', color=visual_map[r'$\bf{SUPER}$']['color'])
ax2.plot(cache_sizes, cost_baseline_1, label='CoorDL', marker='x', linestyle='--', color=visual_map['CoorDL']['color'])
ax2.plot(cache_sizes, cost_baseline_2, label='Shade', marker='s', linestyle='-.', color=visual_map['Shade']['color'])

# Set y-axis label and limits for cost
ax2.set_ylabel('Training Cost ($)', fontsize=12)
ax2.set_ylim(20, 70)  # Adjust limits for clarity
ax2.set_xticks(cache_sizes)
ax2.set_xticklabels([100, 75, 50, 25], fontsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.set_xlabel('Baseline Cache Size (% of Dataset)', fontsize=12)
# ax2.legend(ncol=3, loc='upper center', fontsize=10)
ax2.legend()

# ax2.set_title('Cost vs. Cache Size', fontsize=14)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()
