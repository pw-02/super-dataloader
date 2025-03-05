import numpy as np
import matplotlib.pyplot as plt

# Dummy Data: Throughput, Cost, and Cost Efficiency for Baseline and Your Approach
batch_sizes = np.array([16, 32, 64, 128, 256, 512])
throughput_baseline = np.array([100, 180, 250, 300, 320, 330])  # Batches per second (Baseline)
cost_baseline = np.array([10, 20, 30, 50, 80, 100])  # Compute cost for Baseline ($)

throughput_your = np.array([120, 220, 330, 420, 460, 480])  # Batches per second (Your approach)
cost_your = np.array([8, 18, 28, 40, 65, 85])  # Compute cost for Your approach ($)

# Cost Efficiency (batches per dollar)
cost_efficiency_baseline = throughput_baseline / cost_baseline
cost_efficiency_your = throughput_your / cost_your

# Plotting Cost Efficiency vs Throughput (Scatter Plot)
plt.figure(figsize=(10, 6))
plt.scatter(throughput_baseline, cost_efficiency_baseline, color='r', label='Baseline', s=100)
plt.scatter(throughput_your, cost_efficiency_your, color='b', label='Your Approach', s=100)
plt.xlabel('Throughput (Batches/sec)')
plt.ylabel('Cost Efficiency (Batches/$)')
plt.title('Cost Efficiency vs Throughput')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting Cost Efficiency Comparison (Bar Chart)
labels = ['Your Approach', 'Baseline']
cost_efficiency_values = [np.mean(cost_efficiency_your), np.mean(cost_efficiency_baseline)]
plt.bar(labels, cost_efficiency_values, color=['b', 'r'])
plt.ylabel('Average Cost Efficiency (Batches/$)')
plt.title('Cost Efficiency Comparison')
plt.tight_layout()
plt.show()

# Plotting Throughput vs Compute Cost (Scatter Plot)
plt.figure(figsize=(10, 6))
plt.scatter(cost_baseline, throughput_baseline, color='r', label='Baseline', s=100)
plt.scatter(cost_your, throughput_your, color='b', label='Your Approach', s=100)
plt.xlabel('Compute Cost ($)')
plt.ylabel('Throughput (Batches/sec)')
plt.title('Throughput vs Compute Cost')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
