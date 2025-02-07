import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated data (replace with real logs)
np.random.seed(42)
time_steps = np.arange(0, 1000, 10)  # Time in seconds
batch_request_rate = np.abs(np.sin(time_steps / 150) * 50 + np.random.normal(0, 5, len(time_steps)))  # Varying demand
lambda_concurrency = np.clip(batch_request_rate / 5 + np.random.normal(0, 1, len(time_steps)), 1, 50)  # Scaling Lambda

# --- 1. Time-Series Plot (Lambda Scaling) ---
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(time_steps, lambda_concurrency, label="Lambda Concurrency", color="blue", linewidth=2)
ax1.set_ylabel("Concurrent Lambda Executions", color="blue")
ax1.set_xlabel("Time (seconds)")
ax2 = ax1.twinx()
ax2.plot(time_steps, batch_request_rate, label="Batch Request Rate", color="red", linestyle="dashed", linewidth=2)
ax2.set_ylabel("Batch Request Rate (req/sec)", color="red")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.title("AWS Lambda Scaling vs. Batch Requests")
plt.show()

# --- 2. Scatter Plot (Request Rate vs. Lambda Concurrency) ---
plt.figure(figsize=(8, 5))
plt.scatter(batch_request_rate, lambda_concurrency, alpha=0.7, color="purple")
plt.xlabel("Batch Request Rate (req/sec)")
plt.ylabel("Concurrent Lambda Invocations")
plt.title("Scaling Relationship Between Request Rate and Lambda Concurrency")
plt.grid(True)
plt.show()

# --- 3. Histogram of Data Arrival Timing ---
arrival_delays = np.random.normal(0, 5, len(time_steps))  # Simulated arrival timing (replace with actual)
plt.figure(figsize=(8, 5))
plt.hist(arrival_delays, bins=30, color="green", alpha=0.7)
plt.axvline(0, color="black", linestyle="dashed", label="On-Time Boundary")
plt.xlabel("Arrival Delay (seconds)")
plt.ylabel("Frequency")
plt.title("Distribution of Batch Arrival Timing")
plt.legend()
plt.show()
