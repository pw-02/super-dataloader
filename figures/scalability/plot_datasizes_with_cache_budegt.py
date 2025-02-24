import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec





coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
dataset_label = 'ImageNet'
line_width = 1.5
visual_map_plot_1 = {
    coordl_label: {'color': 'black', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
}


file_paths = [
    r"C:\Users\pw\Desktop\super_results\scalability\increasing_data_sizes_unlimited_budget.csv", 
   r"C:\Users\pw\Desktop\super_results\scalability\increasing_data_sizes_100gb_cache.csv",
   r"C:\Users\pw\Desktop\super_results\scalability\increasing_data_sizes_600gb_cache.csv",

]

for file_path in file_paths:
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6.5, 8), sharex=False)  # Added sharex=True
    # Load data from CSV
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
    # ax1.plot(dataset_size, disdl_throughput_no_prefetch,
    # label=disdl_label,
    #         color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

    # Labels and Title
    # ax1.set_xlabel("Dataset Size (Num Batches)")
    ax1.set_xticks(dataset_size)

    ax1.set_ylabel("Throughput (Samples/sec)")
    ax1.legend()
    # ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
    #set y axis limit range
    ax1.set_ylim(500, 4000)
    #--------------------------------------------------------------------------------

    # coordl_cost = df['coordl(cost_per_epoch)']
    # disdl_cost = df['disdl(cost_per_epoch)']
    # # disdl_throughput_no_prefetch = df['disdl_throughput(samples/sec)(no prefetching)']

    # ax2.plot(dataset_size, coordl_cost, 
    #         label=coordl_label,
    #         color=visual_map_plot_1[coordl_label]['color'], linestyle=visual_map_plot_1[coordl_label]['linestyle'], linewidth=visual_map_plot_1[coordl_label]['linewidth'])

    # ax2.plot(dataset_size, disdl_cost, 
    #         label=disdl_label,
    #          color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])
    # # a1.plot(dataset_size, disdl_throughput_no_prefetch,
    # # label=disdl_label,
    # #         color=visual_map_plot_1[disdl_label]['color'], linestyle=visual_map_plot_1[disdl_label]['linestyle'], linewidth=visual_map_plot_1[disdl_label]['linewidth'])

    # # Labels and Title
    # ax2.set_xlabel("Dataset Size (Num Batches)")
    # ax2.set_xticks(dataset_size)

    # ax2.set_ylabel("Cost per Epoch $")
    # ax2.legend()
    # # ax1.grid(True, linestyle='--', alpha=0.6)
    # ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

    # plt.tight_layout()

    # # # Save and show
    # plt.show()

    # Extract relevant columns
    coordl_compute_cost = df['coordl_compute_cost']
    disdl_compute_cost = df['disdl_compute_cost']
    # baseline_compute_cost = df['disdl_compute_cost(no prefetching)']

    disdl_prefetch_cost = df['disdl_prefetching_cost']
    coordl_prefetch_cost = df['coordl_prefetching_cost']
    # baseline_prefetch_cost = df['disdl_prefetching_cost(no prefetching)']

    coordl_cache_cost = df['coordl_cache_cost']
    disdl_cache_cost = df['disdl_cahe_cost']
    # baseline_cache_cost = df['disdl_cache_cost(no prefetching)']

    # # Extract cost data for compute, prefetch, and cache
    compute_cost = [coordl_compute_cost, disdl_compute_cost]
    prefetch_cost = [coordl_prefetch_cost, disdl_prefetch_cost]
    cache_cost = [coordl_cache_cost, disdl_cache_cost]

    # Define visual styling map for stacked bars
    alpha_value = 1.0
    visual_map_stacked_bar = {
    'compute': {'color': 'white', 'hatch': '/', 'edgecolor': 'black', 'alpha': 1.0},
    'prefetch': {'color': 'white', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
    'cache': {'color': 'white', 'hatch': '.....', 'edgecolor': 'black', 'alpha': 1.0}}

    # # Prepare the batch sizes and labels
    # batch_sizes = ['100', '80', '60', '40', '20']
    barwidth = 0.35  # Width of each bar

    # Initialize plot
    # Set x-axis positions for each batch size
    x = range(len(dataset_size))
# visual_map = {
#     r'$\bf{DisDP}$': {'color': '#005250', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
#     'CoorDL': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
# }
    # Loop through the cost categories and plot them
    group_labels = ['CoorDL', r'$\bf{DisDP}$']
    for offset, label in enumerate(group_labels):
        if label == 'CoorDL':
            compute_values = coordl_compute_cost
            prefetch_values = coordl_prefetch_cost
            cache_values = coordl_cache_cost
            color = '#FEA400'
            edgecolor = 'black',
        else:
            compute_values = disdl_compute_cost
            prefetch_values = disdl_prefetch_cost
            cache_values = disdl_cache_cost
            color = '#005250'
            edgecolor = 'black',
        
        
        # Bar for Compute
        ax2.bar(
            [pos + offset * barwidth for pos in x], 
            compute_values,
            width=barwidth, 
            label=f'{label} Compute', 
            color=color, 
            hatch=visual_map_stacked_bar['compute']['hatch'], 
            edgecolor=edgecolor,
            alpha=alpha_value
        )
        
        # # Bar for Prefetch
        # ax2.bar(
        #     [pos + offset * barwidth for pos in x], 
        #     prefetch_values, 
        #     width=barwidth, 
        #     bottom=compute_values,
        #     label=f'{label} Prefetch', 
        #     color=color,
        #     hatch=visual_map_stacked_bar['prefetch']['hatch'],
        #     edgecolor=edgecolor,
        #     alpha=alpha_value
        # )
        
        # Bar for Cache
        ax2.bar(
            [pos + offset * barwidth for pos in x], 
            cache_values, 
            width=barwidth, 
            bottom=[i + j for i, j in zip(compute_values, prefetch_values)],
            label=f'{label} Cache', 
            color=color,
            hatch=visual_map_stacked_bar['cache']['hatch'],
            edgecolor=edgecolor,
            alpha=0.6
        )

    # Labels and title
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Cost per epoch ($)')
    ax2.set_title('Cost Breakdown: Compute, Prefetch, and Cache')

    # Set the x-ticks to be in the center of each group of bars
    # ax2.set_xticks(dataset_size)
    ax2.set_xticks([pos + barwidth for pos in x])
    ax2.set_xticklabels(dataset_size)

    # ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Add a legend
    ax2.legend()
    plt.tight_layout()

    # Save and show
    plt.show()
