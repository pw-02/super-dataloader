import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Labels and data
coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
dataset_label = 'ImageNet'
line_width = 1.5

visual_map_plot_1 = {
    coordl_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
}

# Data
batch_sizes = np.array([64,124,256,512])  # Number of concurrent jobs
coordl = np.array([136.5865588, 267.86081, 401.7902429, 535.7215442])
super_vals = np.array([292.8484753, 426.7897239, 595.4783372, 595.4783372])
ideal = super_vals  # Ideal follows "Super" values

# Create subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 4.5), sharex=False)

# Plot the throughput graph
ax1.plot(batch_sizes, coordl, 
         label=coordl_label,
         color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])
ax1.plot(batch_sizes, super_vals, 
         label=disdl_label,
         color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])
# ax1.plot(concurrent_jobs, ideal,
#          label=disdl_label,
#          color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and Title for ax1
ax1.set_xticks(batch_sizes)
ax1.set_ylabel("Throughput (Samples/sec)")
ax2.set_xlabel("Number of Concurrent Jobs")
ax1.legend()

# Plot the latency graph
batch_retrieval_latency = np.array([10, 12, 14, 14])  # in milliseconds
preprocessing_latency = np.array([5, 6, 7, 8])  # in milliseconds

ax2.plot(batch_sizes, batch_retrieval_latency, 
         label='Batch Retrieval Latency',
         color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])
ax2.plot(batch_sizes, preprocessing_latency, 
         label='Preprocessing Latency',
         color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and Title for ax2
ax2.set_xticks(batch_sizes)
ax2.set_ylabel("Average Latency/Batch (ms)")
ax2.set_xlabel("Number of Concurrent Jobs")
ax2.legend()

# Improve x-tick formatting
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensures ticks are integers
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensures ticks are integers

# Make sure the layout is tight and no overlaps
plt.tight_layout()

# Show the plot
plt.show()
