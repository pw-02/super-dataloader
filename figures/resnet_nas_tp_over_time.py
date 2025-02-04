import matplotlib.pyplot as plt
import numpy as np

# Define the target number of mini-batches
num_minibatches = 7000

# Simulate processing speeds (mini-batches per second) for each system
np.random.seed(42)  # Reproducibility
coorDL_speed = np.random.normal(10, 1, size=num_minibatches)  # CoorDL processes ~10 minibatches/sec
disDL_no_prefetch_speed = np.random.normal(9, 1, size=num_minibatches)  # DisDL w/o prefetching
disDL_with_prefetch_speed = np.random.normal(11, 1.5, size=num_minibatches)  # DisDL with prefetching

# Compute cumulative time to reach each mini-batch
coorDL_time = np.cumsum(1 / coorDL_speed)  # Time at which each batch is completed
disDL_no_prefetch_time = np.cumsum(1 / disDL_no_prefetch_speed)
disDL_with_prefetch_time = np.cumsum(1 / disDL_with_prefetch_speed)

# Visual map for styling
visual_map = {
    'CoorDL': {'color': '#FEA400', 'linestyle': '--', 'marker': '', 'linewidth': 2.5},
    r'$\bf{DisDP}$': {'color': '#005250', 'linestyle': ':', 'marker': '', 'linewidth': 2.5},
}

# Create the plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15.5, 3.4))  # Adjusted to fit within single column

# Plot number of mini-batches processed over time
ax1.plot(coorDL_time, np.arange(1, num_minibatches + 1), label='CoorDL',
        color=visual_map['CoorDL']['color'], linestyle=visual_map['CoorDL']['linestyle'], linewidth=visual_map['CoorDL']['linewidth'])
ax1.plot(disDL_no_prefetch_time, np.arange(1, num_minibatches + 1), label='DisDL (no prefetch)',
        color=visual_map[r'$\bf{DisDP}$']['color'], linestyle=visual_map[r'$\bf{DisDP}$']['linestyle'], linewidth=visual_map[r'$\bf{DisDP}$']['linewidth'])
ax1.plot(disDL_with_prefetch_time, np.arange(1, num_minibatches + 1), label='DisDL (with prefetch)',
        color='#1f77b4', linestyle='-', linewidth=2.5)

# Labels and title
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Mini-Batches Processed', fontsize=12)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)

# Legend
ax1.legend(loc='upper left', fontsize=9, ncol=1, frameon=False)
# Show the plot

# Plot number of mini-batches processed over time
ax2.plot(coorDL_time, np.arange(1, num_minibatches + 1), label='CoorDL',
        color=visual_map['CoorDL']['color'], linestyle=visual_map['CoorDL']['linestyle'], linewidth=visual_map['CoorDL']['linewidth'])
ax2.plot(disDL_no_prefetch_time, np.arange(1, num_minibatches + 1), label='DisDL (no prefetch)',
        color=visual_map[r'$\bf{DisDP}$']['color'], linestyle=visual_map[r'$\bf{DisDP}$']['linestyle'], linewidth=visual_map[r'$\bf{DisDP}$']['linewidth'])
ax2.plot(disDL_with_prefetch_time, np.arange(1, num_minibatches + 1), label='DisDL (with prefetch)',
        color='#1f77b4', linestyle='-', linewidth=2.5)

# Labels and title
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Mini-Batches Processed', fontsize=12)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)

# Legend
ax2.legend(loc='upper left', fontsize=9, ncol=1, frameon=False)
# Show the plot

# Plot number of mini-batches processed over time
ax3.plot(coorDL_time, np.arange(1, num_minibatches + 1), label='CoorDL',
        color=visual_map['CoorDL']['color'], linestyle=visual_map['CoorDL']['linestyle'], linewidth=visual_map['CoorDL']['linewidth'])
ax3.plot(disDL_no_prefetch_time, np.arange(1, num_minibatches + 1), label='DisDL (no prefetch)',
        color=visual_map[r'$\bf{DisDP}$']['color'], linestyle=visual_map[r'$\bf{DisDP}$']['linestyle'], linewidth=visual_map[r'$\bf{DisDP}$']['linewidth'])
ax3.plot(disDL_with_prefetch_time, np.arange(1, num_minibatches + 1), label='DisDL (with prefetch)',
        color='#1f77b4', linestyle='-', linewidth=2.5)

# Labels and title
ax3.set_xlabel('Time (seconds)', fontsize=12)
ax3.set_ylabel('Mini-Batches Processed', fontsize=12)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)

# Legend
ax3.legend(loc='upper left', fontsize=9, ncol=1, frameon=False)
# Show the plot



plt.tight_layout()
plt.show()
