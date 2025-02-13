import matplotlib.pyplot as plt

# Sample data for demonstration
memory_config = [0, 5, 10, 15, 20, 25, 30]
cache_hit_ratio = [70, 94, 94.1, 94, 94,94, 88]  # Cache Hit %
cost = [0.1, 0.15, 0.19, 0.2, 0.5, 0.8, 0.8]  # Cost / Hour ($)

# Adjusted figure size for three figures in a row
fig, ax1 = plt.subplots(figsize=(5, 3))  # Width x Height in inches

# Plotting the cache hit ratio on the primary y-axis with line style and color
ax1.set_xlabel('Minibatch Request Rate (batches/second)', fontsize=10)
ax1.set_ylabel('Cache Hit %', fontsize=10)
line1, = ax1.plot(memory_config, cache_hit_ratio, 'k-', color='#007E7E', marker='o', linewidth=1.5, label='Cache Hit %')  # Solid line with circle markers
ax1.tick_params(axis='y', labelsize=10)

# Adding a secondary y-axis for cost with line style and color
ax2 = ax1.twinx()
ax2.set_ylabel('Cost / Hour ($)', fontsize=10)
line2, = ax2.plot(memory_config, cost, 'k--', color='#FEA400', marker='x', linewidth=1.5, label='Cost / Hour')  # Dashed line with x markers
ax2.tick_params(axis='y', labelsize=10)

# Add grid lines with consistent style
ax1.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

# Combine legends and set consistent font size
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=9)

# Add a title if needed
# plt.title('Cache Hit % and Cost vs. Minibatch Request Rate', fontsize=10)

# Save the plot with high resolution
# plt.savefig('Figure_11.png', bbox_inches='tight', dpi=300)

# Show plot
plt.show()
