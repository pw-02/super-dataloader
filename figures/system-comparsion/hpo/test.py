import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

# Labels for the different models
coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
dataset_label = 'ImageNet'
line_width = 1.5

visual_map_plot_1 = {
    disdl_label: {'color': '#007777', 'hatch': '...', 'edgecolor': 'black', 'alpha': 1.0},
    coordl_label: {'color': '#FFA500', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
    baseline_label: {'color': '#B0C4DE', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
}

# Expanded categories with more models
categories = [
    'ResNet50', 'ViT-B', 'ALBEF', 
    'EfficientNet-B3', 'ConvNeXt-T', 'MLP-Mixer-B16'
]

# Dummy performance values (samples per second)
values1 = [10, 15, 20, 12, 14, 17]  # CoorDL
values2 = [12, 18, 22, 14, 16, 20]  # Pytorch
values3 = [14, 12, 25, 16, 18, 22]  # DisDP

fig = plt.figure(figsize=(9, 2.5))
gs = gridspec.GridSpec(1, 1, width_ratios=[1])
ax1 = fig.add_subplot(gs[0, 0])

# Set up the x locations for the bars
x = np.arange(len(categories))
width = 0.2  # Adjusted width to accommodate three bars

# Create the bar chart
ax1.bar(x - width, values1, width, label=coordl_label, 
        color=visual_map_plot_1[coordl_label]['color'],
        hatch=visual_map_plot_1[coordl_label]['hatch'], 
        edgecolor=visual_map_plot_1[coordl_label]['edgecolor'],
        alpha=visual_map_plot_1[coordl_label]['alpha'])
ax1.bar(x, values2, width, label=baseline_label, 
        color=visual_map_plot_1[baseline_label]['color'],
        hatch=visual_map_plot_1[baseline_label]['hatch'], 
        edgecolor=visual_map_plot_1[baseline_label]['edgecolor'],
        alpha=visual_map_plot_1[baseline_label]['alpha'])
ax1.bar(x + width, values3, width, label=disdl_label, 
        color=visual_map_plot_1[disdl_label]['color'],
        hatch=visual_map_plot_1[disdl_label]['hatch'], 
        edgecolor=visual_map_plot_1[disdl_label]['edgecolor'],
        alpha=visual_map_plot_1[disdl_label]['alpha'])

# Labels and title
ax1.set_ylabel('Samples/Second', fontsize=11)
ax1.set_xticks(x, categories)
ax1.legend(loc='upper left', fontsize=9, ncol=1, frameon=True)

# Apply thousands formatter to the y-axis
# ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

plt.tight_layout()
plt.show()
