import numpy as np
import matplotlib.pyplot as plt

# Simulated time (in minutes)
time_steps = np.arange(0, 120, 1)  # 2 hours (120 minutes)

# Simulated demand (e.g., batch request rate)
batch_request_rate = 50 + 30 * np.sin(2 * np.pi * time_steps / 60) + np.random.randint(-5, 5, size=len(time_steps))

# Simulated prefetching scaler (adjusted based on demand)
# For example, you could have a simple formula like scaling based on demand.
prefetching_scaler = np.clip(batch_request_rate * 0.5, 10, 100)  # Prefetching scale based on demand

# Create a new figure for visualization
plt.figure(figsize=(10, 6))

# Plot both batch request rate (demand) and prefetching scaler
plt.plot(time_steps, batch_request_rate, label="Batch Request Rate (Demand)", color="steelblue", linewidth=2)
plt.plot(time_steps, prefetching_scaler, label="Prefetching Scaler", color="darkorange", linestyle="--", linewidth=2)

# Add title, labels, and legend
plt.title("Prefetching Scalability vs. Demand")
plt.xlabel("Time (Minutes)")
plt.ylabel("Requests per Minute / Scaling Factor")
plt.legend(frameon=True)

# Adjust layout for professional appearance
plt.tight_layout()
plt.show()
