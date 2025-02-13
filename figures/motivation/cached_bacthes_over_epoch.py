import pandas as pd
import matplotlib.pyplot as plt

# File path (Windows format)
file_path = r"C:\Users\pw\Desktop\super_results\analysis\cached_batches_over_time.csv"

# Load CSV file (adjust delimiter if needed)
df = pd.read_csv(file_path, delimiter="\t")  # Use "," if the file is comma-separated

# Extract relevant columns
epochs = df["Time (Minutes)"]
cached_batches = df["Number of Cached Batches"]
cached_size_gb = df["Total Cached Size GB (with transforms)"]

# Create figure and dual-axis
fig, ax1 = plt.subplots(figsize=(10, 5))

# First y-axis (left) - Number of Cached Batches (Bar Plot)
bars = ax1.bar(epochs, cached_batches, color="orange", label="Number of Cached Batches")
ax1.set_ylabel("Number of Cached Batches", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

# Second y-axis (right) - Total Cached Size GB (Line Plot)
ax2 = ax1.twinx()
line, = ax2.plot(epochs, cached_size_gb, color="gray", linewidth=2, label="Total Cached Size GB (with transforms)")
ax2.set_ylabel("Total Cached Size (GB)", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

# Labels and title
plt.title("Cached Batches vs Total Cached Size Over Epochs")
ax1.set_xlabel("Epoch")

# Combine legends from both axes
handles = [bars, line]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc="upper left")

# Show plot
plt.show()
