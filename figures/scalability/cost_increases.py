import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# Data
dataset_size = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
super_cost = np.array([21.37531264, 42.75062528, 64.12593791, 85.50125055, 106.8765632, 128.2518758, 
                       149.6271885, 171.0025011, 192.3778137, 213.7531264])
coordl_cost = np.array([28.20046079, 78.26437698, 150.1925337, 243.9850048, 359.6417615, 497.1627627, 
                        656.548111, 837.7976711, 1040.911567, 1265.889698])

# Create the plot
plt.figure(figsize=(6, 4))
ax = plt.gca()  # Get current axis

# Plot the costs
plt.plot(dataset_size, super_cost, label='Super', color='#007E7E', marker='o', linestyle='-', linewidth=2, markersize=6)
plt.plot(dataset_size, coordl_cost, label='CoorDL', color='red', marker='s', linestyle='-', linewidth=2, markersize=6)

# Format x-axis to show in "K" units
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

# Format y-axis to show dollar symbol
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))

# Add labels and title
plt.xlabel('Dataset Size (Batches per Epoch)', fontsize=12)
plt.ylabel('Cost per Epoch ($)', fontsize=12)

# Add grid, legend, and customize appearance
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')

# Improve spacing
plt.tight_layout()

# Show the plot
plt.show()
