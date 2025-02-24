import matplotlib.pyplot as plt
import numpy as np

# Define a function for generating convergence curves
def convergence_curve(x, scale=1.0, shift=0.0, rate=0.1):
    return scale * (1 - np.exp(-rate * x)) + shift

# Define a function for generating cost data
def cost_curve(x, base_cost=10, scale_cost=0.5):
    return base_cost + scale_cost * x

# Sample data
time_hours = np.linspace(0, 50, 500)  # Time from 0 to 50 hours, 500 points

# Generate convergence curves and cost data for different partitions
accuracy_partition1 = convergence_curve(time_hours, scale=0.7, shift=0.2, rate=0.1)
accuracy_partition2 = convergence_curve(time_hours, scale=0.75, shift=0.2, rate=0.15)
accuracy_partition3 = convergence_curve(time_hours, scale=0.8, shift=0.2, rate=0.2)
accuracy_partition4 = convergence_curve(time_hours, scale=0.85, shift=0.2, rate=0.25)
accuracy_partition5 = convergence_curve(time_hours, scale=0.9, shift=0.2, rate=0.3)

cost_partition1 = cost_curve(time_hours, base_cost=5, scale_cost=0.4)
cost_partition2 = cost_curve(time_hours, base_cost=6, scale_cost=0.5)
cost_partition3 = cost_curve(time_hours, base_cost=7, scale_cost=0.6)
cost_partition4 = cost_curve(time_hours, base_cost=8, scale_cost=0.7)
cost_partition5 = cost_curve(time_hours, base_cost=9, scale_cost=0.8)

# Create the subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharex=True)

# Plot accuracy curves on the first subplot
ax1.plot(time_hours, accuracy_partition1, label='Partition 1', color='blue')
ax1.plot(time_hours, accuracy_partition2, label='Partition 2', color='green')
ax1.plot(time_hours, accuracy_partition3, label='Partition 3', color='red')
ax1.plot(time_hours, accuracy_partition4, label='Partition 4', color='orange')
ax1.plot(time_hours, accuracy_partition5, label='Partition 5', color='purple')

# Set labels and title for the accuracy subplot
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Plot cost curves on the second subplot
ax2.plot(time_hours, cost_partition1, '--', label='Partition 1 Cost', color='blue')
ax2.plot(time_hours, cost_partition2, '--', label='Partition 2 Cost', color='green')
ax2.plot(time_hours, cost_partition3, '--', label='Partition 3 Cost', color='red')
ax2.plot(time_hours, cost_partition4, '--', label='Partition 4 Cost', color='orange')
ax2.plot(time_hours, cost_partition5, '--', label='Partition 5 Cost', color='purple')

# Set labels and title for the cost subplot
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Cost')
ax2.legend()
ax2.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
