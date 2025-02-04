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


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(19.5, 3.4))  # Adjusted to fit within single column

# Plot the bars for each system using the visual_map
ax1.bar(x - width, coorDL_throughput, width, label=coordl_label, color=visual_map[coordl_label]['color'], 
       edgecolor=visual_map[coordl_label]['edgecolor'], alpha=visual_map[coordl_label]['alpha'], hatch=visual_map[coordl_label]['hatch'])
ax1.bar(x, disDL_no_prefetch_throughput, width, label=disdl_no_prefetch_label, color=visual_map[disdl_no_prefetch_label]['color'], 
       edgecolor=visual_map[disdl_no_prefetch_label]['edgecolor'], alpha=visual_map[disdl_no_prefetch_label]['alpha'], hatch=visual_map[disdl_no_prefetch_label]['hatch'])
ax1.bar(x + width, disDL_with_prefetch_throughput, width, label=disdl_label, color=visual_map[disdl_label]['color'], 
       edgecolor=visual_map[disdl_label]['edgecolor'], alpha=visual_map[disdl_label]['alpha'],hatch=visual_map[disdl_label]['hatch'])

# Set labels and title
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('Samples/second', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
#legend
ax1.legend(loc='upper left', fontsize=9, ncol=1, frameon=False)


# Plot the bars for each system using the visual_map
ax2.bar(x - width, coorDL_throughput, width, label=coordl_label, color=visual_map[coordl_label]['color'], 
       edgecolor=visual_map[coordl_label]['edgecolor'], alpha=visual_map[coordl_label]['alpha'], hatch=visual_map[coordl_label]['hatch'])
ax2.bar(x, disDL_no_prefetch_throughput, width, label=disdl_no_prefetch_label, color=visual_map[disdl_no_prefetch_label]['color'], 
       edgecolor=visual_map[disdl_no_prefetch_label]['edgecolor'], alpha=visual_map[disdl_no_prefetch_label]['alpha'], hatch=visual_map[disdl_no_prefetch_label]['hatch'])
ax2.bar(x + width, disDL_with_prefetch_throughput, width, label=disdl_label, color=visual_map[disdl_label]['color'], 
       edgecolor=visual_map[disdl_label]['edgecolor'], alpha=visual_map[disdl_label]['alpha'],hatch=visual_map[disdl_label]['hatch'])

# Set labels and title
ax2.set_xlabel('Model', fontsize=12)
ax2.set_ylabel('Samples/second', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
#legend
ax2.legend(loc='upper left', fontsize=9, ncol=1, frameon=False)


# Plot the bars for each system using the visual_map
ax3.bar(x - width, coorDL_throughput, width, label=coordl_label, color=visual_map[coordl_label]['color'], 
       edgecolor=visual_map[coordl_label]['edgecolor'], alpha=visual_map[coordl_label]['alpha'], hatch=visual_map[coordl_label]['hatch'])
ax3.bar(x, disDL_no_prefetch_throughput, width, label=disdl_no_prefetch_label, color=visual_map[disdl_no_prefetch_label]['color'], 
       edgecolor=visual_map[disdl_no_prefetch_label]['edgecolor'], alpha=visual_map[disdl_no_prefetch_label]['alpha'], hatch=visual_map[disdl_no_prefetch_label]['hatch'])
ax3.bar(x + width, disDL_with_prefetch_throughput, width, label=disdl_label, color=visual_map[disdl_label]['color'], 
       edgecolor=visual_map[disdl_label]['edgecolor'], alpha=visual_map[disdl_label]['alpha'],hatch=visual_map[disdl_label]['hatch'])

# Set labels and title
ax3.set_xlabel('Model', fontsize=12)
ax3.set_ylabel('Samples/second', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(models)
#legend
ax3.legend(loc='upper left', fontsize=9, ncol=1, frameon=False)



# Show the plot
plt.tight_layout()
plt.show()
