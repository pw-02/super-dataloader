import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Read the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\pw\projects\super-dataloader\sim_report.csv')

# Extract the data from the DataFrame
elapsed_time = df['elasped_time(hours)']
total_cached_batches = df['severless_cache_cost']
# total_cached_data_db = df['toal_cached_data_db']
# Create figure and dual-axis
fig, ax1 = plt.subplots(figsize=(8, 3))
# First y-axis (left) - Total Cached Batches (Line Plot)
line1, = ax1.plot(elapsed_time, total_cached_batches, color="#005250", linewidth=3, label="Total Cached Batches")
ax1.set_ylabel("Total Cache Cost", color="black")
ax1.tick_params(axis="y", labelcolor="black")
# ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
# ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

# # Second y-axis (right) - Total Cached Data (Line Plot)
# ax2 = ax1.twinx()
# line2, = ax2.plot(elapsed_time, total_cached_data_db, color="#005250", linewidth=3, label="Total Cached Data (GB)")
# ax2.set_ylabel("Total Cached Data (GB)", color="black")
# ax2.tick_params(axis="y", labelcolor="black")

# Labels and title
ax1.set_xlabel("Elapsed Time (hours)")

# # Combine legends from both axes
# handles = [line1, line2]
# labels = [h.get_label() for h in handles]
# ax1.legend(handles, labels, loc="upper right")

# Show plot
fig.tight_layout()
plt.show()