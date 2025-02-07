import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
line_width = 1.5
visual_map_plot_1 = {
    coordl_label: {'color': 'black', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
}


#--------------------------------------------------------------------------------
# Data
path  = 'C:\\Users\\pw\\Desktop\\super_results\\image_transformer\\data_for_paper_batches_over_time.csv'

# Read the CSV file
df = pd.read_csv(path, sep=",")
# Extract columns
num_samples = df["Batch"] * 256
disdl_data = df["Elapsed Time (Super)"]
elapsed_coordl = df["Elapsed Time (CoorDL)"]
elapsed_baseline = df["Elapsed Time (Pytorch)"]

# Create the plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17.5, 4.2))  # Adjusted to fit within single column

# Plot number of mini-batches processed over time
ax1.plot(elapsed_coordl, num_samples, label=coordl_label,
        color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])
ax1.plot(elapsed_baseline, num_samples, label=baseline_label,
        color=visual_map_plot_1[baseline_label]['color'], linestyle=visual_map_plot_1[baseline_label]['linestyle'], linewidth=visual_map_plot_1[baseline_label]['linewidth'])
ax1.plot(disdl_data, num_samples, label=disdl_label,
        color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

# Labels and title
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Aggregated Number of Samples Processed', fontsize=11)
ax1.set_title("Cifar10: Aggregated Throughput of 4 Jobs", fontsize=12)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)

# Legend
ax1.legend(loc='upper left', fontsize=9, ncol=1, frameon=True)
# Show the plot
# Apply thousands formatter to the y-axis
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

#--------------------------------------------------------------------------------

path  = 'C:\\Users\\pw\\Desktop\\super_results\\image_transformer\\data_for_paper_cost_over_time.csv'
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
ax2.set_title("Cifar10: Aggregated Cost of 4 Jobs", fontsize=12)

ax2.set_xlabel('Cost ($)', fontsize=12)
ax2.set_ylabel('Aggregated Number of Samples Processed', fontsize=11)

# Enable grid
# ax.grid(True, linestyle='--', alpha=0.6)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Legend
ax2.legend(loc='upper left', fontsize=9, ncol=1, frameon=True)
# Show the plot

#-------------------------------------------------------------------------------------

# visual_map_stacked_bar = {
#     'gpu': {'color': '#005250', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
#     'transform': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
#     'io': {'color': '#FF7F0E', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
# }
visual_map_stacked_bar = {
    'gpu': {'color': '#005250', 'hatch': '////', 'edgecolor': 'black', 'alpha': 0.9},
    'transform': {'color': '#FEA400', 'hatch': 'xxxx', 'edgecolor': 'black', 'alpha': 0.9},
    'io': {'color': '#FF7F0E', 'hatch': '....', 'edgecolor': 'black', 'alpha': 0.9},
}

visual_map_stacked_bar = {
    'gpu': {'color': 'white', 'hatch': '///', 'edgecolor': 'black', 'alpha': 1.0},
    'transform': {'color': 'white', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
    'io': {'color': 'white', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},
}



time_breakdown = {
    'IO': {coordl_label: 23,baseline_label: 17, disdl_label: 7},
    'Transform': {coordl_label: 25,baseline_label: 54, disdl_label: 9},
    'GPU': {coordl_label: 52,baseline_label: 29, disdl_label: 84}
}

# Extracting the data in the right order
loaders = [coordl_label, baseline_label, disdl_label]
bar_width = 0.5  # Set bar width

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
ax3.set_title("Cifar10: Aggregated % Breakdown of Time", fontsize=12)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
# ax3.set_ylim(0, 100)  # Manually set a higher limit
padding = 12
current_ylim = ax3.get_ylim()
ax3.set_ylim(current_ylim[0], current_ylim[1] + padding)
ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
ax3.legend(loc="upper center", ncol=3)  # Moves legend above plot

# Add space above for legend
# plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.show()


# # Visual map for styling
# visual_map = {
#     'CoorDL': {'color': '#FEA400', 'linestyle': '-.', 'marker': '', 'linewidth': 2},
#     r'$\bf{DisDP}$': {'color': '#005250', 'linestyle': ':', 'marker': '', 'linewidth': 2},
# }
# Visual map for styling