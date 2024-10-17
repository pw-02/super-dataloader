import numpy as np
import matplotlib.pyplot as plt

# Dummy data
jobs = np.array([1, 2, 4, 8, 16, 32])
throughput = np.array([100, 190, 360, 640, 900, 1100])  # Throughput in batches/sec
prefetch_latency = np.array([1.2, 1.4, 1.6, 2.0, 2.5, 3.0])  # Prefetch latency in seconds
cache_hits = np.array([95, 90, 85, 80, 75, 70])  # Cache hit percentage
cache_misses = 100 - cache_hits  # Cache miss percentage
cost = np.array([50, 100, 150, 250, 400, 600])  # Cost in $
data_size = np.array([100, 200, 300, 400, 600, 800])  # Data size in GB

# Create a figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Subplot 1: Throughput vs. Number of Jobs
axs[0, 0].plot(jobs, throughput, marker='o', color='b')
axs[0, 0].set_title('Throughput vs. Number of Jobs')
axs[0, 0].set_xlabel('Number of Jobs')
axs[0, 0].set_ylabel('Throughput (Batches/s)')

# Subplot 2: Prefetch Latency vs. Number of Jobs
axs[0, 1].bar(jobs, prefetch_latency, color='orange')
axs[0, 1].set_title('Prefetch Latency vs. Number of Jobs')
axs[0, 1].set_xlabel('Number of Jobs')
axs[0, 1].set_ylabel('Prefetch Latency (s)')

# Subplot 3: Cache Hit/Miss Ratio vs. Number of Jobs (Stacked Bar)
axs[1, 0].bar(jobs, cache_hits, label='Cache Hits', color='green')
axs[1, 0].bar(jobs, cache_misses, bottom=cache_hits, label='Cache Misses', color='red')
axs[1, 0].set_title('Cache Hit/Miss Ratio vs. Number of Jobs')
axs[1, 0].set_xlabel('Number of Jobs')
axs[1, 0].set_ylabel('Percentage (%)')
axs[1, 0].legend()

# Subplot 4: Cost vs. Data Size (Scatter Plot)
scatter = axs[1, 1].scatter(data_size, cost, c=jobs, cmap='viridis', s=100)
axs[1, 1].set_title('Cost vs. Data Size')
axs[1, 1].set_xlabel('Data Size (GB)')
axs[1, 1].set_ylabel('Cost ($)')
cbar = plt.colorbar(scatter, ax=axs[1, 1])
cbar.set_label('Number of Jobs')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
