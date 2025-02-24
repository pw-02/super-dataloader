import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec


# Labels for the different models
coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
dataset_label = 'ImageNet'
line_width = 1.5

# Create subplots
fig = plt.figure(figsize=(11.5, 2.8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])  # First two plots are twice as wide

ax1 = fig.add_subplot(gs[0, 0])  # First plot
ax2 = fig.add_subplot(gs[0, 1])  # Second plot
ax3 = fig.add_subplot(gs[0, 2])  # Third plot

visual_map_stacked_bar = {
    'gpu': {'color': '#007777', 'hatch': '...', 'edgecolor': 'black', 'alpha': 1.0},
    'transform': {'color': '#FFA500', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
    'io': {'color': '#B0C4DE', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
}


time_breakdown = {
    'IO': {coordl_label: 23,baseline_label: 22, disdl_label: 7},
    'Transform': {coordl_label: 29,baseline_label: 41, disdl_label: 20},
    'GPU': {coordl_label: 47,baseline_label: 37, disdl_label: 73}
}

# Extracting the data in the right order
loaders = [coordl_label, baseline_label, disdl_label]
bar_width = 0.75  # Set bar width

io_times = [time_breakdown['IO'][l] for l in loaders]
transform_times = [time_breakdown['Transform'][l] for l in loaders]
gpu_times = [time_breakdown['GPU'][l] for l in loaders]


# Bar positions
x = np.arange(len(loaders))
ax1.bar(x, 
        io_times, 
        width=bar_width,
        label='IO', 
        color=visual_map_stacked_bar['io']['color'],
        edgecolor=visual_map_stacked_bar['io']['edgecolor'], 
        hatch=visual_map_stacked_bar['io']['hatch'])
ax1.bar(x, 
        transform_times,
        width=bar_width,
        bottom=io_times, 
        label='Transform', 
        color=visual_map_stacked_bar['transform']['color'],
        edgecolor=visual_map_stacked_bar['transform']['edgecolor'], 
        hatch=visual_map_stacked_bar['transform']['hatch'])
ax1.bar(x, 
        gpu_times,
        width=bar_width,
        bottom=np.array(io_times) + np.array(transform_times), 
        label='GPU',
        color=visual_map_stacked_bar['gpu']['color'],
        edgecolor=visual_map_stacked_bar['gpu']['edgecolor'], 
        hatch=visual_map_stacked_bar['gpu']['hatch'])

# Labeling
ax1.set_xticks(x)
ax1.set_xticklabels(loaders)
ax1.set_ylabel("Percentage (%)")
# ax3.set_title(f"{dataset_label}: % Breakdown of Time", fontsize=font_size)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax1.set_ylim(0, 100)  # Manually set a higher limit
padding = 15
current_ylim = ax1.get_ylim()
ax1.set_ylim(current_ylim[0], current_ylim[1] + padding)
ax1.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
ax1.legend(loc="upper center", ncol=3, fontsize=9)  # Moves legend above plot
ax1.set_title("(i) ImagNet", fontsize=12)
ax2.set_title("(ii) CIFAR10", fontsize=12)
ax3.set_title("(iii)CIFAR100", fontsize=12)
#----------------------------------
ax2.bar(x, 
        io_times, 
        width=bar_width,
        label='IO', 
        color=visual_map_stacked_bar['io']['color'],
        edgecolor=visual_map_stacked_bar['io']['edgecolor'], 
        hatch=visual_map_stacked_bar['io']['hatch'])
ax2.bar(x, 
        transform_times,
        width=bar_width,
        bottom=io_times, 
        label='Transform', 
        color=visual_map_stacked_bar['transform']['color'],
        edgecolor=visual_map_stacked_bar['transform']['edgecolor'], 
        hatch=visual_map_stacked_bar['transform']['hatch'])
ax2.bar(x, 
        gpu_times,
        width=bar_width,
        bottom=np.array(io_times) + np.array(transform_times), 
        label='GPU',
        color=visual_map_stacked_bar['gpu']['color'],
        edgecolor=visual_map_stacked_bar['gpu']['edgecolor'], 
        hatch=visual_map_stacked_bar['gpu']['hatch'])

# Labeling
ax2.set_xticks(x)
ax2.set_xticklabels(loaders)
ax2.set_ylabel("Percentage (%)")
# ax3.set_title(f"{dataset_label}: % Breakdown of Time", fontsize=font_size)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.set_ylim(0, 100)  # Manually set a higher limit
padding = 15
current_ylim = ax2.get_ylim()
ax2.set_ylim(current_ylim[0], current_ylim[1] + padding)
ax2.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
ax2.legend(loc="upper center", ncol=3, fontsize=9)  # Moves legend above plot

#----------------------------------
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
# ax3.set_title(f"{dataset_label}: % Breakdown of Time", fontsize=font_size)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.set_ylim(0, 100)  # Manually set a higher limit
padding = 15
current_ylim = ax3.get_ylim()
ax3.set_ylim(current_ylim[0], current_ylim[1] + padding)
ax3.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
ax3.legend(loc="upper center", ncol=3, fontsize=9)  # Moves legend above plot











plt.tight_layout()  # Adjusts the layout to prevent overlap
plt.show()
