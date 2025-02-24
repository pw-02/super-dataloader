import matplotlib.pyplot as plt
import numpy as np

# Example data (batch size 1024 removed)
batch_sizes = [16, 32, 64, 128, 256]
retrieval_times_super = [0.08, 0.10, 0.20, 0.30, 0.40]
preprocessing_times_super = [0.04, 0.05, 0.10, 0.15, 0.20]
retrieval_times_pytorch = [0.10, 0.12, 0.22, 0.32, 0.45]
preprocessing_times_pytorch = [0.06, 0.08, 0.13, 0.18, 0.25]
retrieval_times_shade = [0.09, 0.11, 0.21, 0.31, 0.42]
preprocessing_times_shade = [0.05, 0.07, 0.12, 0.17, 0.23]

latency_super = [0.04, 0.05, 0.08, 0.12, 0.18]
latency_pytorch = [0.05, 0.06, 0.09, 0.14, 0.20]
latency_shade = [0.04, 0.05, 0.07, 0.10, 0.16]

throughput_super = [80, 100, 200, 300, 400]
throughput_pytorch = [70, 90, 190, 290, 380]
throughput_shade = [75, 95, 195, 295, 390]

# Create the figure and gridspec layout
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.5])

# Top Row: Combined Average Time per Batch with Stacked Bar Chart
ax0 = fig.add_subplot(gs[0, :])  # Spans both columns in the top row
bar_width = 0.25  # Width of each bar

# Calculate the positions of the bars
bar_positions = np.arange(len(batch_sizes))  # Evenly spaced positions
bar_positions_super = bar_positions - bar_width
bar_positions_pytorch = bar_positions
bar_positions_shade = bar_positions + bar_width

# Plot bars for SUPER
ax0.bar(bar_positions_super, retrieval_times_super, width=bar_width, label='SUPER - Retrieval Time', color='b', alpha=0.7)
ax0.bar(bar_positions_super, preprocessing_times_super, width=bar_width, bottom=retrieval_times_super, label='SUPER - Preprocessing Time', color='b', alpha=0.3)

# Plot bars for PyTorch
ax0.bar(bar_positions_pytorch, retrieval_times_pytorch, width=bar_width, label='PyTorch - Retrieval Time', color='g', alpha=0.7)
ax0.bar(bar_positions_pytorch, preprocessing_times_pytorch, width=bar_width, bottom=retrieval_times_pytorch, label='PyTorch - Preprocessing Time', color='g', alpha=0.3)

# Plot bars for Shade
ax0.bar(bar_positions_shade, retrieval_times_shade, width=bar_width, label='Shade - Retrieval Time', color='r', alpha=0.7)
ax0.bar(bar_positions_shade, preprocessing_times_shade, width=bar_width, bottom=retrieval_times_shade, label='Shade - Preprocessing Time', color='r', alpha=0.3)

ax0.set_xlabel('Batch Size')
ax0.set_ylabel('Time per Batch (s)')
ax0.set_title('Data Retrieval and Preprocessing Time per Batch')
ax0.legend()
ax0.grid(True)
ax0.set_xticks(bar_positions)
ax0.set_xticklabels(batch_sizes)

# Bottom Left: Latency of Data Retrieval
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(batch_sizes, latency_super, label='SUPER', marker='o', color='b')
ax1.plot(batch_sizes, latency_pytorch, label='PyTorch', marker='o', color='g')
ax1.plot(batch_sizes, latency_shade, label='Shade', marker='o', color='r')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Latency (s)')
ax1.set_title('Latency of Data Retrieval')
ax1.legend()
ax1.grid(True)
ax1.set_xticks(batch_sizes)  # Explicitly set tick positions
ax1.set_xticklabels(batch_sizes)  # Ensure labels are evenly spaced
ax1.set_xlim([min(batch_sizes) - 10, max(batch_sizes) + 10])  # Set x-axis limits for even spacing

# Bottom Right: Data Throughput
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(batch_sizes, throughput_super, label='SUPER', marker='o', color='b')
ax2.plot(batch_sizes, throughput_pytorch, label='PyTorch', marker='o', color='g')
ax2.plot(batch_sizes, throughput_shade, label='Shade', marker='o', color='r')
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Throughput (MB/s)')
ax2.set_title('Data Throughput')
ax2.legend()
ax2.grid(True)
ax2.set_xticks(batch_sizes)  # Explicitly set tick positions
ax2.set_xticklabels(batch_sizes)  # Ensure labels are evenly spaced
ax2.set_xlim([min(batch_sizes) - 10, max(batch_sizes) + 10])  # Set x-axis limits for even spacing

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('combined_data_loading_evaluation_even_spacing_lines.png')
plt.show()
