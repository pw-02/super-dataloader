import numpy as np
import matplotlib.pyplot as plt

# Set LaTeX-style font for professional appearance
plt.rcParams.update({
    # "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # "legend.fontsize": 10,
})

# Simulated arrival delays (Delta T in milliseconds)
np.random.seed(42)
delta_t = np.random.normal(loc=0, scale=10, size=1000)  # Mean 0ms, std 10ms
stall_threshold = -5  # Define stall as Delta T < -5ms

# Simulated time (in minutes) and demand-driven Lambda concurrency
time_steps = np.arange(0, 120, 1)  # 2 hours (120 minutes)
batch_request_rate = 50 + 30 * np.sin(2 * np.pi * time_steps / 60) + np.random.randint(-5, 5, size=len(time_steps))
lambda_concurrency = np.clip(batch_request_rate * 0.8, 5, 100)  # Lambda scaling response

# 1. Histogram of Arrival Delays
plt.figure(figsize=(6, 4.2))  # Create a new figure
plt.hist(delta_t, bins=30, color="steelblue", edgecolor="black", linewidth=0.8, alpha=0.7)
plt.axvline(0, color='red', linestyle='dashed', linewidth=1.5, label='Ideal Arrival')
plt.title("Arrival Delays Distribution")
plt.xlabel("Delta T (s)")
plt.ylabel("Number of Batches")
plt.legend(frameon=True)
# plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------- #
# 2. Concurrent AWS Lambda Requests Scaling Over Time
plt.figure(figsize=(6, 4.2))  # Create a new figure
# Plot stacked area chart
plt.fill_between(time_steps, batch_request_rate, color="steelblue", alpha=0.7, label="Batch Request Rate", edgecolor="black", linewidth=1)
plt.fill_between(time_steps, lambda_concurrency, color='#F0F0F0', alpha=1, label="Lambda Concurrency", edgecolor="black", linewidth=1)

# Labels and aesthetics
plt.title("Demand-Driven Scaling of AWS Lambda Requests")
plt.xlabel("Time (Minutes)")
plt.ylabel("Requests per Minute")
plt.legend(frameon=False, loc="upper left")

# Grid for readability
# plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------- #
# 3. Placeholder for third plot (to be replaced with actual analysis)
plt.figure(figsize=(6, 4.2))  # Create a new figure
plt.title("Third Plot Placeholder")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
