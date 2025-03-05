import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec

# # Set global font properties
plt.rc('font', family='serif')  # Set font family, weight, and size
# plt.rc('axes', titlesize=16)  # Set the font size for axes titles
# plt.rc('axes', labelsize=14)  # Set the font size for axes labels
font_size = 13

tensorsocket_label = 'TensorSocket'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
dataset_label = 'ImageNet'
line_width = 1.75
visual_map_plot_1 = {
    tensorsocket_label: {'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
}

# visual_map_plot_1 = {
#     tensorsocket_label: {'color': '#007777', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
#     disdl_label: {'color': '#B0C4DE', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
#     baseline_label: {'color': '#FFA500', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
# }




fig = plt.figure(figsize=(16.5, 3.5))
gs = gridspec.GridSpec(1, 4, width_ratios=[2, 2, 1.5, 1.5])  # First two plots are twice as wide

ax1 = fig.add_subplot(gs[0, 0])  # First plot
ax2 = fig.add_subplot(gs[0, 1])  # Second plot
ax3 = fig.add_subplot(gs[0, 2])  # Third plot
ax4 = fig.add_subplot(gs[0, 3])  # Fourth plot

# Titles for clarity
# Data
path  = r"C:\Users\pw\Desktop\disdl(new)\nas\image_classification\imagenet\paper_data_batches_over_time.csv"

# Read the CSV file
df = pd.read_csv(path, sep=",")
# Extract columns
num_samples = df["Batch"] * 256
disdl_data = df["Elapsed Time(DisDL)"]
elapsed_tensorsocket = df["Elapsed Time(TensorSocket)"]
# elapsed_baseline = df["Elapsed Time (Pytorch)"]
ax1.plot(elapsed_tensorsocket, num_samples, label=tensorsocket_label,
        color=visual_map_plot_1[tensorsocket_label]['color'], linestyle=visual_map_plot_1[tensorsocket_label]['linestyle'], linewidth=visual_map_plot_1[tensorsocket_label]['linewidth'])
# ax1.plot(elapsed_baseline, num_samples, label=baseline_label,
#         color=visual_map_plot_1[baseline_label]['color'], linestyle=visual_map_plot_1[baseline_label]['linestyle'], linewidth=visual_map_plot_1[baseline_label]['linewidth'])
ax1.plot(disdl_data, num_samples, label=disdl_label,
        color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and title
ax1.set_xlabel('Time (seconds)', fontsize=font_size)
ax1.set_ylabel('Aggregated samples processed', fontsize=font_size)
# for label in ax1.get_xticklabels():
#     label.set_fontsize(font_size)
ax1.set_title(f"{dataset_label}: Aggregated Throughput of 4 Jobs", fontsize=font_size)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)

# Legend
ax1.legend(loc='upper left', fontsize=9, ncol=1, frameon=True)
# Show the plot
# Apply thousands formatter to the y-axis
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

#--------------------------------------------------------------------------------

path  = r"C:\Users\pw\Desktop\disdl(new)\nas\image_classification\imagenet\paper_data_costs_over_time .csv"
# Read the CSV file
df = pd.read_csv(path, sep=",")
# Extract columns
num_samples = df["Batch"]
cost_disdp = df["Cost(DisDL)"]
cost_tensorsocket = df["Cost(TensorSocket)"]

# Plot number of mini-batches processed over time
ax2.plot(cost_tensorsocket, num_samples, label=tensorsocket_label,
        color=visual_map_plot_1[tensorsocket_label]['color'], linestyle=visual_map_plot_1[tensorsocket_label]['linestyle'], linewidth=visual_map_plot_1[tensorsocket_label]['linewidth'])
# ax2.plot(cost_baseline, num_samples, label=baseline_label,
        # color=visual_map_plot_1[baseline_label]['color'], linestyle=visual_map_plot_1[baseline_label]['linestyle'], linewidth=visual_map_plot_1[baseline_label]['linewidth'])
ax2.plot(cost_disdp, num_samples, label=disdl_label,
        color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and title
ax2.set_title(f"{dataset_label}: Aggregated Cost of 4 Jobs", fontsize=font_size)

ax2.set_xlabel('Cost ($)', fontsize=font_size)
ax2.set_ylabel('Aggregated samples processed', fontsize=font_size)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)
# ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Legend
ax2.legend(loc='upper left', fontsize=9, ncol=1, frameon=True)
# Show the plot

# #-------------------------------------------------------------------------------------

# visual_map_stacked_bar = {
#    'gpu': {'color': '#005250', 'hatch': '\\\\\',', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
#    'transform': {'color': '#FEA400', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
#    'io': {'color': '#4C8BB8', 'hatch': '...', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
# }

# visual_map_stacked_bar = {
#     'gpu': {'color': '#005250', 'hatch': '/////', 'edgecolor': 'black', 'alpha': 0.9},
#     'transform': {'color': '#FEA400', 'hatch': 'xxxx', 'edgecolor': 'black', 'alpha': 0.9},
#     'io': {'color': '#FF7F0E', 'hatch': '....', 'edgecolor': 'black', 'alpha': 0.9},
# }

# visual_map_stacked_bar = {
#     'gpu': {'color': '#4C8BB8', 'hatch': '/////', 'edgecolor': 'black', 'alpha': 1.0},
#     'transform': {'color': '#FEA400', 'hatch': '....', 'edgecolor': 'black', 'alpha': 1.0},
#     'io': {'color': '#005250', 'hatch': '--', 'edgecolor': 'black', 'alpha': 1.0},
# }

visual_map_stacked_bar = {
    'gpu': {'color': '#007777', 'hatch': '...', 'edgecolor': 'black', 'alpha': 1.0},
    'transform': {'color': '#FFA500', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
    'io': {'color': '#B0C4DE', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
}


time_breakdown = {
    'IO': {tensorsocket_label: 23,baseline_label: 22, disdl_label: 7},
    'Transform': {tensorsocket_label: 29,baseline_label: 41, disdl_label: 20},
    'GPU': {tensorsocket_label: 47,baseline_label: 37, disdl_label: 73}
}

# Extracting the data in the right order
loaders = [tensorsocket_label, baseline_label, disdl_label]
bar_width = 0.75  # Set bar width

io_times = [time_breakdown['IO'][l] for l in loaders]
transform_times = [time_breakdown['Transform'][l] for l in loaders]
gpu_times = [time_breakdown['GPU'][l] for l in loaders]

# Define bar positions
x = np.arange(len(loaders))
ax3.bar(x, 
        io_times, 
        width=bar_width,
        label='IO', 
        color=visual_map_stacked_bar['io']['color'],
        edgecolor=visual_map_stacked_bar['io']['edgecolor'], 
        hatch=visual_map_stacked_bar['io']['hatch'])
ax3.bar(x, 
        transform_times,
        width=bar_width,
        bottom=io_times, 
        label='Transform', 
        color=visual_map_stacked_bar['transform']['color'],
        edgecolor=visual_map_stacked_bar['transform']['edgecolor'], 
        hatch=visual_map_stacked_bar['transform']['hatch'])
ax3.bar(x, 
        gpu_times,
        width=bar_width,
        bottom=np.array(io_times) + np.array(transform_times), 
        label='GPU',
        color=visual_map_stacked_bar['gpu']['color'],
        edgecolor=visual_map_stacked_bar['gpu']['edgecolor'], 
        hatch=visual_map_stacked_bar['gpu']['hatch'])

# Labeling
ax3.set_xticks(x)
ax3.set_xticklabels(loaders)
ax3.set_ylabel("Percentage (%)")
ax3.set_title(f"{dataset_label}: % Breakdown of Time", fontsize=font_size)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.set_ylim(0, 100)  # Manually set a higher limit
padding = 15
current_ylim = ax3.get_ylim()
ax3.set_ylim(current_ylim[0], current_ylim[1] + padding)
ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
ax3.legend(loc="upper center", ncol=3, fontsize=9)  # Moves legend above plot
# #-------------------------------------------------------------------------------------

visual_map_bar = {
   tensorsocket_label: {'color': '#4C8BB8', 'hatch': '///', 'edgecolor': 'black', 'alpha': 1.0},
    disdl_label: {'color': '#FEA400', 'hatch': '....', 'edgecolor': 'black', 'alpha': 1.0},
    baseline_label: {'color': '#005250', 'hatch': '--', 'edgecolor': 'black', 'alpha': 1.0},
}

visual_map_bar = {
   disdl_label: {'color': '#005250', 'hatch': '\\\\\',', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
   tensorsocket_label: {'color': '#FEA400', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'}
}

visual_map_bar = {
    disdl_label: {'color': '#007777', 'hatch': '...', 'edgecolor': 'black', 'alpha': 1.0},
    'transform': {'color': 'white', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
    tensorsocket_label: {'color': '#FFA500', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
}

cache_hit_percentage = {
    'TensorSocket': 74,
    disdl_label: 96,
}

# Extracting the data in the right order
# systems = [tensorsocket_label, baseline_label, disdl_label]
systems = [tensorsocket_label, disdl_label]
# values = np.array(list(cache_hit_percentage.values()))

# Create bars with the appropriate visual properties
for i, system in enumerate(systems):
    visual_props = visual_map_bar[system]
    ax4.bar(
        systems[i], cache_hit_percentage[system],  # Using the actual values, not the mean
        color=visual_props['color'], 
        hatch=visual_props['hatch'], 
        edgecolor=visual_props['edgecolor'], 
        alpha=visual_props['alpha'],
        width=0.5
    )

# Labeling
ax4.set_xticklabels(systems)
ax4.set_ylabel("Cache Hit %")
ax4.set_title(f"{dataset_label}: Cache Hit Rate", fontsize=font_size)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
# ax4.set_ylim(0, 100)  # Manually set a higher limit
# padding = 15
# current_ylim = ax3.get_ylim()
# ax4.set_ylim(current_ylim[0], current_ylim[1])
# ax4.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
# ax4.legend(loc="upper center", ncol=3, fontsize=9)  # Moves legend above plot

plt.tight_layout()
plt.show()
