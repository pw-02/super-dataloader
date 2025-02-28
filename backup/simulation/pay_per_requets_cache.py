import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Number of requests per training epoch (ranging from small to large)
requests = np.linspace(1e3, 8e5, 200)  # From 1,000 to 800,000 requests

# Cost parameters
cost_per_request = 0.00003125 / 1.6666   # Pay-per-request cost per minibatch retrieval
# cost_per_request = 0.20 / 1e6
fixed_cache_cost = 7  # Fixed cost of persistent in-memory cache

# Compute total costs
pay_per_request_cost = cost_per_request * requests
persistent_cache_cost = np.full_like(requests, fixed_cache_cost)

# Find the break-even point
break_even_idx = np.where(pay_per_request_cost < fixed_cache_cost)[0][-1]
break_even_requests = requests[break_even_idx]
break_even_cost = pay_per_request_cost[break_even_idx]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(requests, pay_per_request_cost, label="InfiniStore (serverless)", linestyle='--', color='blue')
plt.plot(requests, persistent_cache_cost, label="ElastiCache", linestyle='-', color='red')

# Highlight break-even point
plt.scatter(break_even_requests, break_even_cost, color='black', zorder=3)

# Set x-axis limit to 1e6 (1 million requests)
# plt.xlim(1e3, 8e5)

# Format the x-axis to display values in thousands (K)
formatter_x = FuncFormatter(lambda x, _: f'{x * 1e-3:.0f}K')
plt.gca().xaxis.set_major_formatter(formatter_x)

# Format the y-axis to display values with a dollar sign
formatter_y = FuncFormatter(lambda x, _: f'${x:.0f}')
plt.gca().yaxis.set_major_formatter(formatter_y)

plt.xlabel("Requests/Epoch")
plt.ylabel("Cost/Epoch ($)")
# plt.title("Cost Comparison: Pay-Per-Request vs. Persistent Cache")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# plt.show()
plt.savefig('figure.png', dpi=300)  # High resolution for print quality

