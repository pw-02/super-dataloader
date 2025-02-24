import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec

# Set global font properties
plt.rc('font', family='serif')  # Set font family, weight, and size
plt.rc('axes', titlesize=16)  # Set the font size for axes titles
plt.rc('axes', labelsize=14)  # Set the font size for axes labels
font_size = 12

coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
dataset_label = 'ImageNet'
line_width = 1.5
visual_map_plot_1 = {
    coordl_label: {'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
}

# visual_map_plot_1 = {
#     coordl_label: {'color': '#007777', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
#     disdl_label: {'color': '#B0C4DE', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
#     baseline_label: {'color': '#FFA500', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
# }




fig = plt.figure(figsize=(4, 2.8))
gs = gridspec.GridSpec(1, 1, width_ratios=[ 1])  # First two plots are twice as wide

ax4 = fig.add_subplot(gs[0, 0])  # First plot
# ax2 = fig.add_subplot(gs[0, 1])  # Second plot
# ax3 = fig.add_subplot(gs[0, 2])  # Third plot
# ax4 = fig.add_subplot(gs[0, 3])  # Fourth plot


# #-------------------------------------------------------------------------------------

visual_map_bar = {
   coordl_label: {'color': '#4C8BB8', 'hatch': '///', 'edgecolor': 'black', 'alpha': 1.0},
    disdl_label: {'color': '#FEA400', 'hatch': '....', 'edgecolor': 'black', 'alpha': 1.0},
    baseline_label: {'color': '#005250', 'hatch': '--', 'edgecolor': 'black', 'alpha': 1.0},
}

visual_map_bar = {
   disdl_label: {'color': '#005250', 'hatch': '\\\\\',', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
   coordl_label: {'color': '#FEA400', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'}
}

visual_map_bar = {
    disdl_label: {'color': '#007777', 'hatch': '...', 'edgecolor': 'black', 'alpha': 1.0},
    'transform': {'color': 'white', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
    coordl_label: {'color': '#FFA500', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
}

cache_hit_percentage = {
    'CoorDL': 74,
    disdl_label: 96,
}

# Extracting the data in the right order
# systems = [coordl_label, baseline_label, disdl_label]
systems = [coordl_label, disdl_label]
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
# ax4.set_title(f"{dataset_label}: Cache Hit Rate", fontsize=font_size)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
# ax4.set_ylim(0, 100)  # Manually set a higher limit
# padding = 15
# current_ylim = ax3.get_ylim()
# ax4.set_ylim(current_ylim[0], current_ylim[1])
# ax4.set_yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
# ax4.legend(loc="upper center", ncol=3, fontsize=9)  # Moves legend above plot

plt.tight_layout()
plt.show()
