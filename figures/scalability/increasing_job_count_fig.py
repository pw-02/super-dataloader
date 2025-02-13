import matplotlib.pyplot as plt
import numpy as np

# Data
concurrent_jobs = np.arange(1, 21)
coordl = np.array([136.5865588, 267.86081, 401.7902429, 535.7215442, 669.6590489, 803.5965539,
                   937.5354311, 1071.462984, 1205.394251, 1339.324019, 1473.263421, 1578.652709,
                   1710.202822, 1841.76132, 1973.320231, 2104.87912, 2236.436375, 2367.990649,
                   2499.546177, 2631.103609])
super_vals = np.array([292.8484753, 426.7897239, 595.4783372, 795.8563516, 1328.265121, 1860.721967,
                       2701.305051, 2846.832603, 3046.519497, 3220.134065, 4300.065485, 4431.628357,
                       4582.358744, 5022.062329, 5507.568325, 5990.47046, 6332.883061, 6556.549489,
                       6818.441795, 7171.929227])
ideal = super_vals  # Ideal follows "Super" values

# Plot
plt.figure(figsize=(6, 4))
plt.plot(concurrent_jobs, coordl, linestyle='-', color='#007E7E', label='CoorDL')  # Change color if needed
plt.plot(concurrent_jobs, ideal, linestyle='dashed', color='red', label='CoorDL')
plt.scatter(concurrent_jobs, super_vals, color='blue', label='Super')

# Labels, legend, and formatting
plt.xlabel("Concurrent Jobs")
plt.ylabel("Throughput (samples/sec)")
plt.xticks(np.arange(1, 21, 2))  # Set x-axis ticks in increments of 2
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Improve spacing
plt.tight_layout()
# Show plot
plt.show()
