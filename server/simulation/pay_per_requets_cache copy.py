import matplotlib.pyplot as plt

# Data
batch_size = [1, 16, 32, 64, 128, 256]
num_requests = [1281167, 80073, 40036, 20018, 10009, 5005]

batch_size = reversed(batch_size)
num_requests = reversed(num_requests)

# Plot
plt.figure(figsize=(10, 5))  # Suitable for single column width in papers
plt.plot(batch_size, num_requests, marker='o', linestyle='-', color='b')

# Labels and title
plt.xlabel("Batch Size")
plt.ylabel("Number of Requests")
plt.title("Batch Size vs Number of Requests")

# Optional: Log scale for better visualization if needed
# plt.yscale('log')  # Uncomment if you want to use log scale

# Grid and show plot
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
