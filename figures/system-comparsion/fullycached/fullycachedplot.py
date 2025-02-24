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
    coordl_label: {'color': 'black', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
}

file_paths = [
    r"C:\Users\pw\Desktop\super_results\scalability\increasing_data_sizes_unlimited_budget.csv", 
   r"C:\Users\pw\Desktop\super_results\scalability\increasing_data_sizes_100gb_cache.csv",
   r"C:\Users\pw\Desktop\super_results\scalability\increasing_data_sizes_600gb_cache.csv",

]


for file_path in file_paths:

    fig = plt.figure(figsize=(4.8, 3.2))
    gs = gridspec.GridSpec(1,1, width_ratios=[1])  # First two plots are twice as wide
    ax1 = fig.add_subplot(gs[0, 0])  # First plot

    df = pd.read_csv(file_path, delimiter=',')
    # Extract relevant columns5
    dataset_size = df['dataset_size(num_batches)']
    coordl_throughput = df['coordl_throughput(samples/sec)']
    disdl_throughput = df['disdl_throughput(samples/sec)']
    # disdl_throughput_no_prefetch = df['disdl_throughput(samples/sec)(no prefetching)']

    ax1.plot(dataset_size, coordl_throughput, 
                label=coordl_label,
                color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])

    ax1.plot(dataset_size, disdl_throughput, 
            label=disdl_label,
                color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

    # ax1.set_xticks(dataset_size)
    ax1.set_ylabel("Throughput (Samples/sec)")
    ax1.legend()
    # ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
    #set y axis limit range
    # ax1.set_ylim(500, 4000)

    # for label in ax1.get_xticklabels():
    #     label.set_fontsize(font_size)
    for label in ax1.get_yticklabels():
        label.set_fontsize(font_size)


    plt.tight_layout()

        # Save and show
    plt.show()
