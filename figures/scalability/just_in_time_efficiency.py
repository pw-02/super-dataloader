import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated arrival delays (Delta T in milliseconds)
np.random.seed(42)
delta_t = np.random.normal(loc=0, scale=10, size=1000)  # Mean 0ms, std 10ms
stall_threshold = -5  # Define stall as Delta T < -5ms

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogram of Arrival Delays
sns.histplot(delta_t, bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].axvline(0, color='red', linestyle='dashed', label='Ideal Arrival')
axes[0, 0].set_title("Histogram of Arrival Delays")
axes[0, 0].set_xlabel("Delta T (ms)")
axes[0, 0].set_ylabel("Number of Batches")
axes[0, 0].legend()

# 2. CDF of Arrival Delays
sorted_delays = np.sort(delta_t)
cdf = np.arange(len(sorted_delays)) / len(sorted_delays)
axes[0, 1].plot(sorted_delays, cdf, marker='.', linestyle='none', label='CDF')
axes[0, 1].axvline(0, color='red', linestyle='dashed', label='Ideal Arrival')
axes[0, 1].set_title("CDF of Arrival Delays")
axes[0, 1].set_xlabel("Delta T (ms)")
axes[0, 1].set_ylabel("Cumulative Probability")
axes[0, 1].legend()

# 3. Time-Series Plot of Arrival Delays
time_steps = np.arange(len(delta_t))
axes[1, 0].plot(time_steps, delta_t, marker='o', linestyle='-', markersize=3)
axes[1, 0].axhline(0, color='red', linestyle='dashed', label='Ideal Arrival')
axes[1, 0].axhline(stall_threshold, color='black', linestyle='dotted', label='Stall Threshold')
axes[1, 0].set_title("Time-Series of Arrival Delays")
axes[1, 0].set_xlabel("Training Step")
axes[1, 0].set_ylabel("Delta T (ms)")
axes[1, 0].legend()

# 4. Stall Rate vs. Prefetching Aggressiveness
aggressiveness = np.linspace(0.5, 2.0, 10)  # Scaling factor for prefetching
stall_rates = [np.mean(delta_t < (-5 * a)) for a in aggressiveness]
axes[1, 1].plot(aggressiveness, stall_rates, marker='o', linestyle='-')
axes[1, 1].set_title("Stall Rate vs. Prefetching Aggressiveness")
axes[1, 1].set_xlabel("Prefetching Aggressiveness (Scaling Factor)")
axes[1, 1].set_ylabel("Stall Rate (%)")

plt.tight_layout()
plt.show()
