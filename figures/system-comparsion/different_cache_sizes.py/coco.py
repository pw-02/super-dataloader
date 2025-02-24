import matplotlib.pyplot as plt

# Assuming the cost breakdown data is already in this format
cost_breakdown = {
    "Compute": {
        "CoorDL": {'80': 10.07, '60': 12.07, '40': 14.07, '20': 15.07},
        r'$\bf{DisDP}$': {'80': 10, '60': 10, '40': 10, '20': 10}
    },
    "Prefetch": {
        "CoorDL": {'80': 56, '60': 56, '40': 56, '20': 56},
        r'$\bf{DisDP}$': {'80': 3, '60': 3, '40': 3, '20': 3}
    },
    "Cache": {
        "CoorDL": {'80': 33.93, '60': 32, '40': 29.93, '20': 28.93},
        r'$\bf{DisDP}$': {'80': 88, '60': 88, '40': 88, '20': 88}
    }
}

# Prepare the batch sizes and labels
batch_sizes = ['80', '60', '40', '20']
group_labels = ['CoorDL', r'$\bf{DisDP}$']
barwidth = 0.35  # Width of each bar

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 7))

# Set x-axis positions for each batch size
x = range(len(batch_sizes))

# Loop through the cost categories and plot them
for offset, label in enumerate(group_labels):
    # Extract data for each category
    compute_values = [cost_breakdown["Compute"][label][batch_size] for batch_size in batch_sizes]
    prefetch_values = [cost_breakdown["Prefetch"][label][batch_size] for batch_size in batch_sizes]
    cache_values = [cost_breakdown["Cache"][label][batch_size] for batch_size in batch_sizes]

    # Bar for Compute
    ax.bar(
        [pos + offset * barwidth for pos in x], 
        compute_values,
        width=barwidth, 
        label=f'{label} Compute', 
        color='#005250', 
        hatch='----', 
        edgecolor='black'
    )
    
    # Bar for Prefetch
    ax.bar(
        [pos + offset * barwidth for pos in x], 
        prefetch_values, 
        width=barwidth, 
        bottom=compute_values,
        label=f'{label} Prefetch', 
        color='#FEA400', 
        hatch='..', 
        edgecolor='black'
    )
    
    # Bar for Cache
    ax.bar(
        [pos + offset * barwidth for pos in x], 
        cache_values, 
        width=barwidth, 
        bottom=[i + j for i, j in zip(compute_values, prefetch_values)],
        label=f'{label} Cache', 
        color='#4C8BB8', 
        hatch='///////', 
        edgecolor='black'
    )

# Labels and title
ax.set_xlabel('Batch Size')
ax.set_ylabel('Cost')
ax.set_title('Cost Breakdown: Compute, Prefetch, and Cache')

# Set the x-ticks to be in the center of each group of bars
ax.set_xticks([pos + barwidth / 2 for pos in x])
ax.set_xticklabels(batch_sizes)

# Add a legend
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
