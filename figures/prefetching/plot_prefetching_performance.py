import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import numpy as np
coordl_label = 'CoorDL'
disdl_label = r'$\bf{DisDP}$'
baseline_label = 'Pytorch'
line_width = 1.5
visual_map_plot_1 = {
    coordl_label: {'color': 'black', 'linestyle': '--', 'marker': '', 'linewidth': line_width},
    disdl_label: {'color': 'black', 'linestyle': ':', 'marker': '', 'linewidth': line_width},
    baseline_label: {'color': 'black', 'linestyle': '-', 'marker': '', 'linewidth': line_width},
}


#--------------------------------------------------------------------------------


# Create the plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(17.5, 3.7))  # Adjusted to fit within single column
# Bar width
bar_width = 0.25
training_speed = [5,10,15,20] #batches per second
cache_hits_no_cost_cap = [100,98,94,93] #cache hit ratio
cache_hits_10_dollar_cost_cap = [98,84,81,81] #cache hit ratio
cache_hits_5_dollar_cost_cap = [90,72,72,70] #cache hit ratio
x = np.arange(len(training_speed))

# Plot bars for cache hit rates under different cost caps at each training speed
ax1.bar(x - bar_width, cache_hits_no_cost_cap, bar_width, label='No Cost Cap', color='#FEA400',hatch='\\', edgecolor='black')
ax1.bar(x, cache_hits_10_dollar_cost_cap, bar_width, label='$10 Cost Cap', color='#005250',hatch='//', edgecolor='black')
ax1.bar(x + bar_width, cache_hits_5_dollar_cost_cap, bar_width, label='$5 Cost Cap', color='#4C8BB8', hatch='.', edgecolor='black')

# Set labels and title
ax1.set_xlabel('Training Speed (Batches/Second)', fontsize=12)
ax1.set_ylabel('Cache Hit Rate', fontsize=11)

# Set custom x-axis ticks for each training speed
ax1.set_xticks(x)
ax1.set_xticklabels(training_speed)

# Enable grid

# Legend
ax1.legend(loc='upper center', fontsize=9, ncol=3, frameon=True)
# Show the plot
# Apply thousands formatter to the y-axis
# ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

#--------------------------------------------------------------------------------
delta_t = np.random.normal(loc=0, scale=10, size=1000)  # Mean 0ms, std 10ms

# 2. CDF of Arrival Delays
sorted_delays = np.sort(delta_t)
cdf = np.arange(len(sorted_delays)) / len(sorted_delays)
ax2.plot(sorted_delays, cdf, marker='.', linestyle='none', label='CDF')
ax2.axvline(0, color='red', linestyle='dashed', label='Ideal Arrival')
ax2.set_xlabel("Delta T (ms)")
ax2.set_ylabel("Cumulative Probability")
ax2.legend()

#-------------------------------------------------------------------------------------

# Simulated time (in minutes)
time_steps = np.arange(0, 120, 1)  # 2 hours (120 minutes)
# Simulated demand (e.g., batch request rate)
batch_request_rate = 50 + 30 * np.sin(2 * np.pi * time_steps / 60) + np.random.randint(-5, 5, size=len(time_steps))
# Simulated prefetching scaler (adjusted based on demand)
# For example, you could have a simple formula like scaling based on demand.
prefetching_scaler = np.clip(batch_request_rate * 0.5, 10, 100)  # Prefetching scale based on demand

# Plot both batch request rate (demand) and prefetching scaler
ax3.plot(time_steps, batch_request_rate, label="Batch Request", color="steelblue", linewidth=2)
ax3.plot(time_steps, prefetching_scaler, label="Prefetch Request", color="darkorange", linestyle="--", linewidth=2)

# # Add title, labels, and legend
ax3.set_xlabel("Time (Minutes)")
ax3.set_ylabel("Requests / Second")
ax3.legend(frameon=True)

# Adjust layout for professional appearance
plt.tight_layout()
plt.show()
#-------------------------------------------------------------------------------------