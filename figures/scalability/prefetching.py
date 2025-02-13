import matplotlib.pyplot as plt
import numpy as np

# Simulated data for demonstration
# X-axis: Training Demand (batches processed per second)
training_demand = np.linspace(1, 100, 10)  # 10 data points for batches per second

# Cache hit rate for different cost caps (prefetching capacity limits)
cache_hit_rate_cap1 = np.clip(90 - 0.015 * (training_demand - 1), 75, 90)  # Cap 1: High capacity
cache_hit_rate_cap2 = np.clip(90 - 0.03 * (training_demand - 1), 60, 85)   # Cap 2: Medium capacity
cache_hit_rate_cap3 = np.clip(90 - 0.05 * (training_demand - 1), 50, 80)   # Cap 3: Low capacity

# Create the plot
plt.figure(figsize=(5.83, 4.2))

# Plot the data as line plots for each cap
plt.plot(training_demand, cache_hit_rate_cap1, label='Cost Cap 1', color='green', marker='o', linestyle='-', linewidth=2, markersize=6)
plt.plot(training_demand, cache_hit_rate_cap2, label='Cost Cap 2', color='blue', marker='s', linestyle='-', linewidth=2, markersize=6)
plt.plot(training_demand, cache_hit_rate_cap3, label='Cost Cap 3', color='red', marker='^', linestyle='-', linewidth=2, markersize=6)

# Add labels and title
plt.xlabel('Training Demand (Batches/Second)', fontsize=12)
plt.ylabel('Cache Hit Rate (%)', fontsize=12)

# Add grid and customize appearance
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add legend
plt.legend()

# Tight layout for spacing
plt.tight_layout()

# Show the plot
plt.show()
