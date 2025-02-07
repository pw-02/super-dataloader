import numpy as np
import matplotlib.pyplot as plt

# Simulated time-series data
time = np.arange(0, 100, 1)  # 100 time steps
actual_concurrency = np.random.randint(5, 25, size=len(time))  # Random usage
concurrency_cap = 15  # Fixed cap

# Time-Series Plot
plt.figure(figsize=(8, 4))
plt.plot(time, actual_concurrency, label="Actual Concurrency", color='blue')
plt.axhline(y=concurrency_cap, color='red', linestyle='--', label="Concurrency Cap")
plt.xlabel("Time")
plt.ylabel("Concurrency Level")
plt.legend()
plt.title("Concurrency Usage vs. Cap Over Time")
plt.show()

# Histogram of concurrency levels
plt.figure(figsize=(6, 4))
plt.hist(actual_concurrency, bins=10, edgecolor='black', alpha=0.7)
plt.axvline(x=concurrency_cap, color='red', linestyle='--', label="Concurrency Cap")
plt.xlabel("Concurrency Level")
plt.ylabel("Frequency")
plt.legend()
plt.title("Frequency of Concurrency Levels")
plt.show()

import seaborn as sns

# Simulated data
cost = [100, 150, 250, 350, 500]  # AWS Lambda cost at different caps
delay_percent = [30, 20, 10, 5, 2]  # % of delayed batches
concurrency_settings = ["Cap 10", "Cap 20", "Cap 30", "Cap 40", "Unlimited"]

# Scatter Plot - Cost vs Delay Tradeoff
plt.figure(figsize=(6, 4))
plt.scatter(cost, delay_percent, color='purple', label="Trade-off Points")
plt.plot(cost, delay_percent, linestyle="--", color="gray")  # Trendline
plt.xlabel("AWS Lambda Cost ($)")
plt.ylabel("% of Delayed Batches")
plt.title("Cost vs Delay Tradeoff")
plt.legend()
plt.show()

# Stacked Bar Chart - Cost Breakdown
compute_cost = [60, 90, 150, 210, 300]  # Part of total cost from execution
invocation_cost = [40, 60, 100, 140, 200]  # Cost from number of requests

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(concurrency_settings, compute_cost, label="Compute Cost", color='blue')
ax.bar(concurrency_settings, invocation_cost, bottom=compute_cost, label="Invocation Cost", color='orange')

plt.ylabel("Total Cost ($)")
plt.title("Cost Breakdown by Concurrency Cap")
plt.legend()
plt.show()
