import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Function to format the y-axis as percentages
def percent_formatter(x, pos):
    return f'{int(x)}%'

# Define the visual map
visual_map = {
    'io': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'transform': {'color': '#FEA400', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},
    'gpu': {'color': '#4C8BB8', 'hatch': '/////', 'edgecolor': 'black', 'alpha': 1.0},
}

# Data for the specified workloads
result_data = {
    'ResNet-50/ImageNet': {
        'CoorDL': {'IO %': 45, 'Transform %': 15, 'GPU %': 55},
        'SHADE': {'IO %': 74, 'Transform %': 8, 'GPU %': 26},
        r'$\bf{SUPER}$': {'IO %': 29, 'Transform %': 10, 'GPU %': 71},
    },
    'Albef/COCO': {
        'CoorDL': {'IO %': 56, 'Transform %': 12, 'GPU %': 32},
        'SHADE': {'IO %': 35, 'Transform %': 7, 'GPU %': 65},
        r'$\bf{SUPER}$': {'IO %': 29, 'Transform %': 9, 'GPU %': 73},
    },
    'ViT/Cifar10': {
        'CoorDL': {'IO %': 56, 'Transform %': 12, 'GPU %': 32},
        'SHADE': {'IO %': 35, 'Transform %': 7, 'GPU %': 65},
        r'$\bf{SUPER}$': {'IO %': 29, 'Transform %': 9, 'GPU %': 73},
    },
    'Pythia-14m/OpenWebText': {
        'LitData': {'IO %': 2, 'Transform %': 51, 'GPU %': 48},
        r'$\bf{SUPER}$': {'IO %': 2, 'Transform %': 7, 'GPU %': 91},
    }
}
bar_width = 0.75
# Iterate over the models and create a separate bar chart for each
for model_name, workloads in result_data.items():
    labels = list(workloads.keys())
    io_values = [workloads[label]['IO %'] for label in labels]
    transform_values = [workloads[label]['Transform %'] for label in labels]
    gpu_values = [workloads[label]['GPU %'] for label in labels]

    # Create a new figure for each model
    plt.figure(figsize=(4.5, 3))
    
    # Create the stacked bar chart
    plt.bar(labels, io_values, width=bar_width, label='IO %', color=visual_map['io']['color'], hatch=visual_map['io']['hatch'])
    plt.bar(labels, transform_values, width=bar_width, bottom=io_values, label='Transform %', color=visual_map['transform']['color'], hatch=visual_map['transform']['hatch'])
    plt.bar(labels, gpu_values, bottom=np.array(io_values) + np.array(transform_values), width=bar_width, label='GPU %', color=visual_map['gpu']['color'], hatch=visual_map['gpu']['hatch'])

    # Customize the chart
    plt.ylabel('Time Breakdown (%)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    current_ylim = plt.ylim()
    padding = 20
    plt.ylim(current_ylim[0], current_ylim[1] + padding)    
    plt.title(f'{model_name}', fontsize=12)
    # Manually set y-ticks and y-tick labels
    plt.yticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])

    # Create a legend for the figure
    plt.legend(loc='upper center', ncols=3, fontsize=10)

    # Show the plot for the current model
    plt.tight_layout()
    plt.show()
