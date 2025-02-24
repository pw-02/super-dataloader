import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.gridspec as gridspec

# Set global font properties
plt.rc('font', family='serif')  # Set font family, weight, and size
plt.rc('axes', titlesize=16)  # Set the font size for axes titles
# plt.rc('axes', labelsize=14)  # Set the font size for axes labels

# Given time to process one batch in seconds for both jobs
time_per_batch_fast = 0.107508104  # seconds for the fast job
time_per_batch_slow = 0.339788711  # seconds for the slow job


#compute throuhgput batches/second
throughput_fast = 1 / time_per_batch_fast
throughput_slow = 1 / time_per_batch_slow

print(f"Throughput fast: {throughput_fast} batches/second")
print(f"Throughput slow: {throughput_slow} batches/second")


# Calculate batches processed per hour for both jobs
batches_per_hour_fast = 3600 / time_per_batch_fast
batches_per_hour_slow = 3600 / time_per_batch_slow

# Create a list of hours to plot
hours = np.arange(1, 11)  # from 1 to 10 hours
batches_processed_fast = batches_per_hour_fast * hours  # calculate batches for fast job
batches_processed_slow = batches_per_hour_slow * hours  # calculate batches for slow job

# Calculate the difference in batches processed per hour
batches_difference = batches_processed_fast - batches_processed_slow

# Calculate processing rate (batches per hour)
processing_rate_fast = batches_per_hour_fast
processing_rate_slow = batches_per_hour_slow

# Calculate memory usage based on the difference in cached batches
batch_size_mb = 35.51037311553955  # each batch requires 50 MB
memory_usage_difference = batches_difference * batch_size_mb / 1024  # in GB, considering positive values only
memory_usage_difference[memory_usage_difference < 0] = 0  # Avoid negative memory usage

font_size = 13
# Plotting
fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 1, width_ratios=[1])  # First two plots are twice as wide
ax1 = fig.add_subplot(gs[0, 0])  # First plot

visual_map_stacked_bar = {
    'bar': {'color': 'white', 'hatch': '/', 'edgecolor': 'black', 'alpha': 1.0},
    'line': {'color': 'white', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},
}

# First y-axis (left) - Number of Cached Batches (Bar Plot)
bars = ax1.bar(hours, batches_difference, 
               color=visual_map_stacked_bar['bar']['color'], 
               hatch=visual_map_stacked_bar['bar']['hatch'], 
               edgecolor=visual_map_stacked_bar['bar']['edgecolor'], 
               alpha=visual_map_stacked_bar['bar']['alpha'], 
               label="Number of Mini-batches")  # Blue

ax1.set_ylabel("Number of Mini-batches", fontsize=font_size)
ax1.tick_params(axis="y", labelsize=font_size)  # Set y-tick font size

ax1.set_xlabel("Training Time (Hours)", fontsize=font_size)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

# Second y-axis (right) - Total Cached Size GB (Line Plot)
ax2 = ax1.twinx()
line, = ax2.plot(hours, memory_usage_difference, 
                 color="black", 
                 linewidth=3, 
                 label="Storage Requirement (GB)")  # Orange
ax2.set_ylabel("Storage Requirement (GB)", fontsize=font_size)
ax2.tick_params(axis="y", labelsize=font_size)  # Set y-tick font size for the second axis

# Combine legends from both axes
handles = [bars, line]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc="upper left")

plt.tight_layout()  # Adjusts plot to fit into the figure area
plt.show()
