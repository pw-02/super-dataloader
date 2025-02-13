import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec

coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
dataset_label = 'ImageNet'
line_width = 1.5
visual_map_plot_1 = {
    coordl_label: {'color': 'black', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
}



fig = plt.figure(figsize=(9.25, 3.8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])  # First two plots are twice as wide

ax1 = fig.add_subplot(gs[0, 0])  # First plot
ax2 = fig.add_subplot(gs[0, 1])  # Second plot
# ax3 = fig.add_subplot(gs[0, 2])  # Third plot
# ax4 = fig.add_subplot(gs[0, 3])  # Fourth plot

# Titles for clarity
# Data
path  = 'C:\\Users\\pw\\Desktop\\super_results\\\image_classification\\data_for_paper_batches_over_time.csv'

# Read the CSV file
df = pd.read_csv(path, sep=",")
# Extract columns
num_samples = df["Batch"] * 256
disdl_data = df["Elapsed Time (Super)"]
elapsed_coordl = df["Elapsed Time (CoorDL)"]
elapsed_baseline = df["Elapsed Time (Pytorch)"]
ax1.plot(elapsed_coordl, num_samples, label=coordl_label,
        color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])
ax1.plot(elapsed_baseline, num_samples, label=baseline_label,
        color=visual_map_plot_1[baseline_label]['color'], linestyle=visual_map_plot_1[baseline_label]['linestyle'], linewidth=visual_map_plot_1[baseline_label]['linewidth'])
ax1.plot(disdl_data, num_samples, label=disdl_label,
        color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and title
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Aggregated Number of Samples Processed', fontsize=11)
ax1.set_title(f"{dataset_label}: Aggregated Throughput of 4 Jobs", fontsize=12)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)

# Legend
ax1.legend(loc='upper left', fontsize=9, ncol=1, frameon=True)
# Show the plot
# Apply thousands formatter to the y-axis
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

#--------------------------------------------------------------------------------

path  = "C:\\Users\\pw\\Desktop\\super_results\\image_classification\\data_for_paper_cost_over_time.csv"
# Read the CSV file
df = pd.read_csv(path, sep="\t")
# Extract columns
num_samples = df["Batch"]
cost_disdp = df["Cost (Super)"]
cost_coordl = df["Cost (CoorDL)"]
cost_baseline = df["Cost (Pytorch)"]

# Plot number of mini-batches processed over time
ax2.plot(cost_coordl, num_samples, label=coordl_label,
        color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])
ax2.plot(cost_baseline, num_samples, label=baseline_label,
        color=visual_map_plot_1[baseline_label]['color'], linestyle=visual_map_plot_1[baseline_label]['linestyle'], linewidth=visual_map_plot_1[baseline_label]['linewidth'])
ax2.plot(cost_disdp, num_samples, label=disdl_label,
        color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and title
ax2.set_title(f"{dataset_label}: Aggregated Cost of 4 Jobs", fontsize=12)

ax2.set_xlabel('Cost ($)', fontsize=12)
ax2.set_ylabel('Aggregated Number of Samples Processed', fontsize=11)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Legend
ax2.legend(loc='upper left', fontsize=9, ncol=1, frameon=True)
# Show the plot

plt.tight_layout()
plt.show()
