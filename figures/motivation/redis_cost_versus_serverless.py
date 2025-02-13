import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Define cost per hour range
cost_per_hour = np.linspace(0.1, 50, 20)  # From $0.1 to $50 per hour

# Redis: Amount of data that can be cached per hour (assuming monthly Redis cost per GB)
redis_monthly_cost_per_gb = 50  # $50 per GB per month
redis_hourly_cost_per_gb = redis_monthly_cost_per_gb / 720  # Convert to cost per hour
redis_storage_gb = cost_per_hour / redis_hourly_cost_per_gb  # Storage capacity in GB

# Serverless cache: Number of requests possible per hour
serverless_cost_per_request = 0.000002  # $0.000002 per request
serverless_requests_per_hour = cost_per_hour / serverless_cost_per_request  # Requests per hour

# Create figure and dual-axis
fig, ax1 = plt.subplots(figsize=(6, 4))

# First y-axis (left) - Redis Cached Storage (Bar Plot)
bars = ax1.bar(cost_per_hour, redis_storage_gb, color="#005250", label="Redis Cache Capacity (GB)", width=1.25)
ax1.set_ylabel("Redis Cache Capacity (GB)", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)} GB"))

# Second y-axis (right) - Serverless Cache Requests per Hour (Line Plot)
ax2 = ax1.twinx()
line, = ax2.plot(cost_per_hour, serverless_requests_per_hour, color="gray", linewidth=3, label="Serverless Cache Requests per Hour")
ax2.set_ylabel("Serverless Cache Requests per Hour", color="black")
ax2.tick_params(axis="y", labelcolor="black")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M"))

# Labels and title
ax1.set_xlabel("Cost per Hour ($)")

# Combine legends from both axes
handles = [bars, line]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc="upper left")

# Show plot
plt.show()
