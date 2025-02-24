import matplotlib.pyplot as plt
import numpy as np

# Example data: Simulate how the batch request rate changes over time
time_intervals = np.arange(0, 100, 1)  # Time intervals from 0 to 100 (e.g., seconds or cycles)
num_requests = np.log(time_intervals + 1) * 100  # Simulate number of requests over time, log scaling
batch_request_rate = np.diff(num_requests)  # Rate at which batches are being requested (differences in request count)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot batch request rate over time (left axis)
ax1.plot(time_intervals[1:], batch_request_rate, color='blue', label='Batch Request Rate', linestyle='-', linewidth=2)
ax1.set_xlabel('Time (Intervals)', fontsize=12)
ax1.set_ylabel('Batch Request Rate', fontsize=11, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Add grid
ax1.grid(True, linestyle='--', alpha=0.6)

# Title
plt.title("Batch Request Rate Over Time as Number of Requests Increases", fontsize=14)

# Add legend
ax1.legend(loc='upper left', fontsize=9, frameon=True)

# Show the plot
plt.tight_layout()
plt.show()
