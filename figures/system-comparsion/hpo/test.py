import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

# Labels for the different models
coordl_label = 'TensorSocket'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Peak'
dataset_label = 'ImageNet'
line_width = 1.5

visual_map_plot_1 = {
    disdl_label: {'color': '#007777', 'hatch': '...', 'edgecolor': 'black', 'alpha': 1.0},
    coordl_label: {'color': '#FFA500', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
    baseline_label: {'color': '#B0C4DE', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
}

# Expanded categories with more models

categories = [
    'AlexNet', 'MobileNetL', 'MobileNetS', 'RegNetX_4GF', 'ResNet18', 'ResNet50', 'ShufflenetV2'
]
job_speed_per_path = {
    'AlexNet': 0.036923768,
    'MobileNetL': 0.108870757,
    'MobileNetS': 0.047705063,
    'RegNetX_4GF': 0.151863368,
    'ResNet18': 0.103246557,
    'ResNet50': 0.340791997,
    'ShufflenetV2': 0.057448398,
}
num_jobs = 4
samples_per_batch = 128
optimal_throughputs =[]
for jobspeed in job_speed_per_path.values():
    baatchespersec = 1/jobspeed
    total_batches_per_sec = baatchespersec * num_jobs
    total_samples_per_sec = total_batches_per_sec * samples_per_batch
    optimal_throughputs.append(total_samples_per_sec)

# Dummy performance values (samples per second)
values1 = [2036.64067008781, 1998.392322, 2033.271491,1946.195471,1835.009251,1336.221136,2030.692505]  # CoorDL
values3 = [2313.28429375731, 2275.66799277362, 2345.0902926565, 2240.15109, 2292.679817, 1445.584623, 2313.390423]  # DisDP

fig = plt.figure(figsize=(9, 2.5))
gs = gridspec.GridSpec(1, 1, width_ratios=[1])
ax1 = fig.add_subplot(gs[0, 0])

# Set up the x locations for the bars
x = np.arange(len(categories))
width = 0.35  # Adjusted width to accommodate three bars

# Create the bar chart
ax1.bar(x - width, values1, width, label=coordl_label, 
        color=visual_map_plot_1[coordl_label]['color'],
        hatch=visual_map_plot_1[coordl_label]['hatch'], 
        edgecolor=visual_map_plot_1[coordl_label]['edgecolor'],
        alpha=visual_map_plot_1[coordl_label]['alpha'])
ax1.bar(x, values3, width, label=disdl_label, 
        color=visual_map_plot_1[disdl_label]['color'],
        hatch=visual_map_plot_1[disdl_label]['hatch'], 
        edgecolor=visual_map_plot_1[disdl_label]['edgecolor'],
        alpha=visual_map_plot_1[disdl_label]['alpha'])

ax1.set_ylim(0, 2800)

# Labels and title
ax1.set_ylabel('Samples/Second', fontsize=11)
ax1.set_xticks(x - width / 2)
ax1.set_xticklabels(categories)
ax1.legend(loc='upper right', fontsize=9, ncol=1, frameon=True)

ax1.set_xlabel('Model', fontsize=11)
plt.tight_layout()
plt.show()
