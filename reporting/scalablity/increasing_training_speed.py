import matplotlib.pyplot as plt

# Data points
training_demand = [0.5, 0.4, 0.3, 0.2, 0.1]  # X-axis: batch processing time
super = [949.931554817107, 1191.07788627257, 1470.87042045695, 1932.77772713556, 2127.45548180037]  # Y-axis: training speed (example data)
baseline = [947, 1149, 1153, 1152, 1163]  # Y-axis: Ideal training speed (sample values)

# Plotting the line chart
plt.figure(figsize=(4.7, 2.5))  # Set figure size

# Plot actual training speed
plt.plot(training_demand, super, marker='o', linestyle='-', color='#005250', label=r'$\bf{SUPER}$')

# Plot ideal training speed
plt.plot(training_demand, baseline, marker='s', linestyle='--', color='#FEA400', label='Baseline')

# Adding labels and title
plt.xlabel('Batch Execution Time on GPU (Seconds)')
plt.ylabel('Training Speed (Samples/sec)')

# Adding grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Inverting the x-axis (optional: if smaller values mean higher demand)
plt.gca().invert_xaxis()

# Display the plot
plt.show()