# import matplotlib.pyplot as plt

# # Example data
# configurations = ['Config A', 'Config B', 'Config C', 'Config D']
# costs = [10, 15, 20, 25]  # Example costs in dollars
# throughputs = [100, 150, 120, 90]  # Example throughput in samples per second

# # Scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(costs, throughputs, c='blue', marker='o')
# for i, config in enumerate(configurations):
#     plt.text(costs[i], throughputs[i], config, fontsize=12, ha='right')
# plt.xlabel('Cost ($)')
# plt.ylabel('Throughput (samples/sec)')
# plt.title('Cost vs. Throughput for Different Configurations')
# plt.grid(True)
# plt.show()


# import numpy as np

# # Example data
# configurations = ['Config A', 'Config B', 'Config C', 'Config D']
# costs = [10, 15, 20, 25]
# latencies = [100, 80, 120, 90]  # Example latency in milliseconds

# # Bar chart
# x = np.arange(len(configurations))
# width = 0.35

# fig, ax1 = plt.subplots(figsize=(10, 6))

# bars1 = ax1.bar(x - width/2, costs, width, label='Cost ($)', color='blue')
# ax2 = ax1.twinx()
# bars2 = ax2.bar(x + width/2, latencies, width, label='Latency (ms)', color='orange')

# ax1.set_xlabel('Configuration')
# ax1.set_ylabel('Cost ($)')
# ax2.set_ylabel('Latency (ms)')
# ax1.set_xticks(x)
# ax1.set_xticklabels(configurations)
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# plt.title('Cost and Latency by Configuration')
# plt.grid(True)
# plt.show()


# import seaborn as sns
# import pandas as pd

# # Example data
# data = {
#     'Configuration': ['Config A', 'Config B', 'Config C', 'Config D'],
#     'Cost ($)': [10, 15, 20, 25],
#     'Throughput (samples/sec)': [100, 150, 120, 90],
#     'Latency (ms)': [100, 80, 120, 90]
# }

# df = pd.DataFrame(data)
# df.set_index('Configuration', inplace=True)

# # Heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(df, annot=True, cmap='viridis', fmt='.2f')
# plt.title('Heatmap of Performance and Cost Across Configurations')
# plt.show()


# # Example data
# configurations = ['Config A', 'Config B', 'Config C', 'Config D']
# costs = [10, 15, 20, 25]
# throughputs = [100, 150, 120, 90]

# # Line plot
# plt.figure(figsize=(8, 6))
# plt.plot(costs, throughputs, marker='o', linestyle='-', color='green')
# plt.xlabel('Cost ($)')
# plt.ylabel('Throughput (samples/sec)')
# plt.title('Cost-Performance Curve')
# plt.grid(True)
# plt.show()


# import matplotlib.pyplot as plt

# # Example data
# configurations = ['Config A', 'Config B', 'Config C', 'Config D']
# base_cost = [5, 7, 8, 10]
# variable_cost = [5, 8, 12, 15]

# # Stacked bar chart
# plt.figure(figsize=(10, 6))
# bar1 = plt.bar(configurations, base_cost, color='lightblue', label='Base Cost')
# bar2 = plt.bar(configurations, variable_cost, bottom=base_cost, color='darkblue', label='Variable Cost')

# plt.xlabel('Configuration')
# plt.ylabel('Cost ($)')
# plt.title('Cost Breakdown by Configuration')
# plt.legend()
# plt.grid(True)
# plt.show()


# import matplotlib.pyplot as plt

# # Example data
# configurations = ['Config A', 'Config B', 'Config C', 'Config D']
# costs = [10, 15, 20, 25]  # Example costs in dollars
# throughputs = [100, 150, 120, 90]  # Example throughput in samples per second
# ratios = [t / c for t, c in zip(throughputs, costs)]

# # Scatter plot
# plt.figure(figsize=(8, 6))
# plt.bar(configurations, ratios, color='teal')
# plt.xlabel('Configuration')
# plt.ylabel('Throughput-to-Cost Ratio (samples/sec per $)')
# plt.title('Throughput-to-Cost Ratio for Different Configurations')
# plt.grid(True)
# plt.show()

# import matplotlib.pyplot as plt

# # Example data
# configurations = ['Config A', 'Config B', 'Config C', 'Config D']
# ratios = [10, 10, 6, 3.6]  # Example throughput-to-cost ratios

# # Bar chart
# plt.figure(figsize=(10, 6))
# plt.bar(configurations, ratios, color='coral')
# plt.xlabel('Configuration')
# plt.ylabel('Throughput-to-Cost Ratio (samples/sec per $)')
# plt.title('Throughput-to-Cost Ratio by Configuration')
# plt.grid(True)
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Example data
# configurations = ['Config A', 'Config B', 'Config C', 'Config D']
# costs = [10, 15, 20, 25]
# throughputs = [100, 150, 120, 90]
# ratios = [t / c for t, c in zip(throughputs, costs)]

# x = np.arange(len(configurations))

# # Dual y-axis plot
# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.bar(x - 0.2, costs, 0.4, label='Cost ($)', color='blue')
# ax1.set_xlabel('Configuration')
# ax1.set_ylabel('Cost ($)', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')

# ax2 = ax1.twinx()
# ax2.plot(x, throughputs, color='green', marker='o', label='Throughput (samples/sec)')
# ax2.set_ylabel('Throughput (samples/sec)', color='green')
# ax2.tick_params(axis='y', labelcolor='green')

# # Adding ratio as a line plot
# ax3 = ax1.twinx()
# ax3.plot(x, ratios, color='red', linestyle='--', marker='x', label='Throughput-to-Cost Ratio')
# ax3.set_ylabel('Throughput-to-Cost Ratio (samples/sec per $)', color='red')
# ax3.spines['right'].set_position(('outward', 60))
# ax3.tick_params(axis='y', labelcolor='red')

# ax1.set_xticks(x)
# ax1.set_xticklabels(configurations)
# fig.tight_layout()
# plt.title('Cost, Throughput, and Throughput-to-Cost Ratio by Configuration')
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Example data
batch_sizes = [32, 64, 128, 256, 512]
lambda_sizes = [512, 1024, 2048, 10000]
throughput_cost_ratio = {
    512: [20.0, 26.67, 40.0, 53.33,44.54],
    1024: [14.67, 19.09, 34.0, 48.57,55.4],
    2048: [12.0, 14.67, 25.14, 38.89,55.4],
    10000: [8.67, 11.5, 20.0, 32.73,51.1],
    'Redis': [14.67, 19.09, 28.33, 42.5, 44.5]
}

# Bar chart settings
bar_width = 0.15
index = np.arange(len(batch_sizes))  # Positions for batch sizes

# Plotting
plt.figure(figsize=(6, 3))

# Bar plots for each Lambda size
for i, lambda_size in enumerate(lambda_sizes):
    plt.bar(index + i * bar_width, throughput_cost_ratio[lambda_size], bar_width, label=f'{lambda_size} MB Lambda')

# Bar plot for ElastiCache Redis
plt.bar(index + len(lambda_sizes) * bar_width, throughput_cost_ratio['Redis'], bar_width, label='ElastiCache Redis', color='red', hatch='//')

# Customizing x-axis
plt.xlabel('Batch Size')
plt.ylabel('Throughput/Cost (samples/sec per $)')
# Adjust x-tick positions to center them under groups
xticks_pos = index + (len(lambda_sizes) / 2) * bar_width  # Center positions for x-ticks
plt.xticks(xticks_pos, batch_sizes)

plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

