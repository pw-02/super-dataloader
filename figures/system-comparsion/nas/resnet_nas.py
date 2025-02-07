import matplotlib.pyplot as plt
import numpy as np

coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
disdl_no_prefetch_label = 'DisDP(w/o prefetch)'

# Sample data: Model names and throughputs for each system
models = ['ResNet18', 'SqueezeNet', 'ResNet50', 'MobileNetV2']
coorDL_throughput = [150, 200, 180, 250]  # CoorDL throughput in samples/second
disDL_no_prefetch_throughput = [120, 170, 160, 220]  # DisDL w/o prefetching throughput
disDL_with_prefetch_throughput = [160, 210, 190, 260]  # DisDL with prefetching throughput

# Set positions for the bars
x = np.arange(len(models))  # The label locations
width = 0.2  # Width of the bars

# Define the visual map for the systems
visual_map = {
    disdl_label: {'color': '#005250', 'hatch': '/////', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
    disdl_no_prefetch_label: {'color': '#E3E3E3', 'hatch': '....', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
    coordl_label: {'color': '#FEA400', 'hatch': '\\\\\\', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
}


# Create a figure and axis
fig, ax = plt.subplots(figsize=(5, 3))

# Plot the bars for each system using the visual_map
ax.bar(x - width, coorDL_throughput, width, label=coordl_label, color=visual_map[coordl_label]['color'], 
       edgecolor=visual_map[coordl_label]['edgecolor'], alpha=visual_map[coordl_label]['alpha'], hatch=visual_map[coordl_label]['hatch'])
ax.bar(x, disDL_no_prefetch_throughput, width, label=disdl_no_prefetch_label, color=visual_map[disdl_no_prefetch_label]['color'], 
       edgecolor=visual_map[disdl_no_prefetch_label]['edgecolor'], alpha=visual_map[disdl_no_prefetch_label]['alpha'], hatch=visual_map[disdl_no_prefetch_label]['hatch'])
ax.bar(x + width, disDL_with_prefetch_throughput, width, label=disdl_label, color=visual_map[disdl_label]['color'], 
       edgecolor=visual_map[disdl_label]['edgecolor'], alpha=visual_map[disdl_label]['alpha'],hatch=visual_map[disdl_label]['hatch'])

# Set labels and title
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Samples/second', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models)
#legend
ax.legend(loc='upper left', fontsize=9, ncol=1, frameon=False)
# ax.grid(axis='y', linestyle='--', linewidth=0.5)

# # Optionally, add text labels on top of the bars
# for i in range(len(models)):
#     ax.text(i - width, coorDL_throughput[i] + 5, f'{coorDL_throughput[i]}', ha='center', fontsize=10)
#     ax.text(i, disDL_no_prefetch_throughput[i] + 5, f'{disDL_no_prefetch_throughput[i]}', ha='center', fontsize=10)
#     ax.text(i + width, disDL_with_prefetch_throughput[i] + 5, f'{disDL_with_prefetch_throughput[i]}', ha='center', fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()
