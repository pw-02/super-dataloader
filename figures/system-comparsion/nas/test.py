import matplotlib.pyplot as plt
import numpy as np

# Given data
cache_hit_percentage = {
    'CoordL': [0.5, 0.6, 0.7, 0.8, 0.9],
    'DisDP': [0.6, 0.7, 0.8, 0.9, 0.95],
    'Pytorch': [0.7, 0.8, 0.9, 0.95, 0.99]
}

# Prepare the data
systems = list(cache_hit_percentage.keys())
values = np.array(list(cache_hit_percentage.values()))

# Plotting the data
fig, ax = plt.subplots(figsize=(8, 6))

# Creating a bar for each system
ax.bar(systems, values.mean(axis=1), color=['blue', 'green', 'red'])

# Adding labels and title
ax.set_xlabel('System')
ax.set_ylabel('Average Cache Hit Percentage')
ax.set_title('Average Cache Hit Percentage for Different Systems')

# Display the plot
plt.tight_layout()
plt.show()
