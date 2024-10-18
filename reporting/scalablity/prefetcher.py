import time
import random
import matplotlib.pyplot as plt
import math
# Variables for simulation
total_time = 60  # Run simulation for 60 seconds
time_step = 0.5  # Interval for updates (seconds)
max_batches = 50  # Maximum batches that can be consumed per second

# Initialize lists to store time-series data
time_points = []
training_demand = []
prefetch_rate = []

# Function to simulate training demand over time
def simulate_training_speed(t):
    # Gradually increase and decrease speed using a sine wave pattern
    return int(max_batches * (0.5 + 0.5 * (1 + math.sin(t / 5))))

# Simulate prefetcher and training demand over time
for t in range(int(total_time / time_step)):
    current_time = t * time_step
    
    # Simulate training demand at this time step
    demand = simulate_training_speed(current_time)
    training_demand.append(demand)
    
    # Simulate prefetcher trying to match the demand (with some random delay)
    prefetch = max(0, demand + random.randint(-5, 5))  # Add noise
    prefetch_rate.append(prefetch)
    
    # Store the current time point
    time_points.append(current_time)
    
    # Adjust sleep to simulate processing time
    time.sleep(max(0.01, 1 / demand))  # Faster sleep means higher demand

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time_points, training_demand, label='Training Demand (batches/s)', color='red')
plt.plot(time_points, prefetch_rate, label='Prefetcher Rate (batches/s)', color='blue')
plt.xlabel('Time (seconds)')
plt.ylabel('Batches per second')
plt.title('Real-Time Adaptation of Prefetcher to Changing Training Demand')
plt.legend()
plt.grid(True)
plt.show()
