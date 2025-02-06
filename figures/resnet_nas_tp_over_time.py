import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker

# Data
path  = 'C:\\Users\\pw\\Desktop\\super_results\\image_transformer\\data_for_paper_batches_over_time.csv'
coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Baseline'

# Read the CSV file
df = pd.read_csv(path, sep="\t")
# Extract columns
num_samples = df["Batch"] * 256
disdl_data = df["Elapsed Time (Super)"]
elapsed_coordl = df["Elapsed Time (CoorDL)"]
elapsed_super_redis = df["Elapsed Time (super with redis)"]

line_styles = ['-', '--', '-.', ':']
line_width = 1.5
visual_map_plot_1 = {
    coordl_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
}

# Create the plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17.5, 3.4))  # Adjusted to fit within single column

# Plot number of mini-batches processed over time
ax1.plot(elapsed_coordl, num_samples, label=coordl_label,
        color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])
ax1.plot(elapsed_super_redis, num_samples, label=baseline_label,
        color=visual_map_plot_1[baseline_label]['color'], linestyle=visual_map_plot_1[baseline_label]['linestyle'], linewidth=visual_map_plot_1[baseline_label]['linewidth'])
ax1.plot(disdl_data, num_samples, label=disdl_label,
        color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and title
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Aggregated Number of Samples Processed', fontsize=9)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)

# Legend
ax1.legend(loc='upper left', fontsize=9, ncol=1, frameon=True)
# Show the plot
# Apply thousands formatter to the y-axis
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))


plt.tight_layout()
plt.show()


# # Visual map for styling
# visual_map = {
#     'CoorDL': {'color': '#FEA400', 'linestyle': '-.', 'marker': '', 'linewidth': 2},
#     r'$\bf{DisDP}$': {'color': '#005250', 'linestyle': ':', 'marker': '', 'linewidth': 2},
# }
# Visual map for styling