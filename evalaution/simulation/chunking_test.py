import time
from collections import deque

# Define jobs with different speeds
jobs = {"A": 5.0, "B": 0.5}  # Job A is twice as fast as Job B
chunks = [1,2]
num_epochs = 2
cache = {}  # Simulated cache {chunk_id: epoch}
queue = deque(chunks)  # Ordered chunk queue

# Tracking job positions
job_progress = {job: {"chunk": queue[0], "epoch": 1} for job in jobs}

# To track the number of batches in the cache over time
batch_size_per_chunk = 100  # Each chunk contains 100 batches
batches_in_cache_over_time = []

# Simulation loop
for t in range(100):  # Arbitrary time steps
    print(f"\nTime step {t+1}")

    # Process each job based on its speed
    for job, speed in jobs.items():
        if t % (1/speed) == 0:  # Simulate different speeds
            current_chunk = job_progress[job]["chunk"]
            current_epoch = job_progress[job]["epoch"]

            # Store processed batch in cache
            cache[(current_chunk, current_epoch)] = f"Processed by {job}"

            # Move to next chunk
            next_chunk = queue[(queue.index(current_chunk) + 1) % len(queue)]
            if next_chunk == 1:
                job_progress[job]["epoch"] += 1  # New epoch starts
            job_progress[job]["chunk"] = next_chunk
        else:
            print(f"Job {job} is idle")

    # Cache eviction: Remove a chunk when all jobs have processed it
    for chunk, epoch in list(cache.keys()):
        if all(job_progress[j]["epoch"] > epoch or job_progress[j]["chunk"] != chunk for j in jobs):
            print(f"Evicting chunk {chunk} of epoch {epoch} from cache")
            del cache[(chunk, epoch)]

    # Track the number of batches in the cache at this time step
    # Each chunk has 100 batches, so we multiply the number of chunks in cache by 100
    batches_in_cache = len(cache) * batch_size_per_chunk
    batches_in_cache_over_time.append(batches_in_cache)

    # # Display cache status
    # print("Cache:", cache)
    # time.sleep(0.5)  # Slow down simulation for readability

# After the loop, print the number of batches in cache over time
print("\nBatches in cache over time:", batches_in_cache_over_time)

# Plot the number of batches in the cache over time
import matplotlib.pyplot as plt
plt.plot(batches_in_cache_over_time)
plt.xlabel('Time Step')
plt.ylabel('Batches in Cache')
plt.title('Number of Batches in Cache Over Time')
plt.show()
