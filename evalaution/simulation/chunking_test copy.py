import heapq
import time
from collections import deque

# Define jobs with different speeds
jobs = {"A": 1.0, "B": 0.5}  # Job A is twice as fast as Job B
# Define chunks and their batches
chunks = {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}  # 5 chunks with 100 batches each
num_epochs = 2
cache = {}  # Simulated cache {chunk_id: epoch: {batch_id: processed_by}}
queue = deque(chunks.keys())  # Ordered chunk queue

# Tracking job positions and progress
job_progress = {job: {"chunk": queue[0], "epoch": 1, "batches_processed": 0} for job in jobs}
batch_cache = {chunk: set() for chunk in chunks}  # Tracks batches already processed in cache

# Function to simulate processing a batch
def process_batch(job, chunk, batch_id):
    """Simulate batch processing: check cache and process accordingly"""
    if batch_id not in batch_cache[chunk]:
        # Cache miss, process the batch
        print(f"Job {job} is processing batch {batch_id} from chunk {chunk}")
        batch_cache[chunk].add(batch_id)
        return False  # Cache miss
    else:
        # Cache hit, just reuse the batch
        print(f"Job {job} hit cache for batch {batch_id} from chunk {chunk}")
        return True  # Cache hit

# Simulation loop
for t in range(15):  # Arbitrary time steps
    print(f"\nTime step {t+1}")

    # Process each job based on its speed
    for job, speed in jobs.items():
        if t % (1/speed) == 0:  # Simulate different speeds
            current_chunk = job_progress[job]["chunk"]
            current_epoch = job_progress[job]["epoch"]
            batches_processed = job_progress[job]["batches_processed"]

            # Process the next batch in the current chunk
            batch_id = batches_processed + 1  # Process next batch in the chunk
            cache_hit = process_batch(job, current_chunk, batch_id)

            # Move to next batch in the chunk
            job_progress[job]["batches_processed"] += 1

            # If all batches in a chunk are processed, move to next chunk
            if job_progress[job]["batches_processed"] >= chunks[current_chunk]:
                next_chunk = queue[(queue.index(current_chunk) + 1) % len(queue)]
                if next_chunk == 1:
                    job_progress[job]["epoch"] += 1  # New epoch starts
                job_progress[job]["chunk"] = next_chunk
                job_progress[job]["batches_processed"] = 0

    # Cache eviction: Remove a chunk when all jobs have processed it
    for chunk in list(chunks.keys()):
        if all(job_progress[j]["epoch"] > current_epoch or job_progress[j]["chunk"] != chunk for j in jobs):
            print(f"Evicting chunk {chunk} from cache")
            batch_cache[chunk].clear()

    # Display cache status and jobs processing progress
    print("Cache:", batch_cache)
    for job in jobs:
        print(f"Job {job} progress: Chunk {job_progress[job]['chunk']}, Epoch {job_progress[job]['epoch']}, Batches Processed {job_progress[job]['batches_processed']}")
    
