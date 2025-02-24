import numpy as np
import matplotlib.pyplot as plt

# Given time to process one batch in seconds for both jobs
time_per_batch_fast = 0.107508104  # seconds for the fast job
time_per_batch_slow = 0.507133094  # seconds for the slow job

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
batch_size_mb = 50  # each batch requires 50 MB
memory_usage_difference = batches_difference * batch_size_mb / 1024  # in GB, considering positive values only
memory_usage_difference[memory_usage_difference < 0] = 0  # Avoid negative memory usage

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(17, 3.5))

# 2. Processing Rate Between the Two Jobs (Vertical Bar Chart)
axs[0].bar(['Fast Job', 'Slow Job'], [processing_rate_fast, processing_rate_slow], color=['green', 'red'])
axs[0].set_title('Processing Rate (Batches per Hour)')
axs[0].set_ylabel('Batches Processed per Hour')
axs[0].grid(axis='y')



# 1. Difference in Batches Processed Per Hour
axs[1].plot(hours, batches_difference, marker='o', color='purple', label='Batches Difference')
axs[1].set_title('Difference in Batches Processed Per Hour')
axs[1].set_xlabel('Hours')
axs[1].set_ylabel('Difference in Number of Batches Processed')
axs[1].set_xticks(hours)
axs[1].grid()
axs[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')  # horizontal line at y=0 for reference
axs[1].legend()



# 3. Memory Usage Based on the Difference in Cached Batches
axs[2].plot(hours, memory_usage_difference, marker='o', color='blue', label='Memory Usage (GB)')
axs[2].set_title('Memory Usage Based on Cached Batches Difference')
axs[2].set_xlabel('Hours')
axs[2].set_ylabel('Memory Usage (GB)')
axs[2].set_xticks(hours)
axs[2].grid()
axs[2].legend()

plt.tight_layout()  # adjusts plot to fit into the figure area
plt.show()
