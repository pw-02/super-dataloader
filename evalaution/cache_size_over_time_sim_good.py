import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Simulation parameters
batches_per_epoch = 10
total_epochs = 10
total_batches = batches_per_epoch * total_epochs
job_speeds = [0.125, 5.5]  # Time per batch (in seconds)
cached_batches = []
cache_accesses_counter = {}
cache_size_over_time = []  # Track cache size at each step
time_steps = []  # Track corresponding time values

class Job:
    def __init__(self, job_id, time_per_batch, total_batches):
        self.job_id = job_id
        self.time_per_batch = time_per_batch  # Now represents time taken per batch
        self.expected_duration_sec = total_batches * time_per_batch
        self.batches_to_process = list(range(1, total_batches + 1))
        self.batches_processed = 0
        self.next_available_time = 0  # Tracks when the job can process the next batch

    def __lt__(self, other):
        return self.next_available_time < other.next_available_time  # Priority queue sorting

    def process_next_batch(self):
        if not self.batches_to_process:
            return False  # No more batches to process

        next_batch = self.batches_to_process.pop(0)
        self.batches_processed += 1
        self.next_available_time += self.time_per_batch  # Schedule next batch processing time

        # Cache management
        if next_batch in cached_batches:
            cache_accesses_counter[next_batch] += 1
            if cache_accesses_counter[next_batch] >= len(jobs):  # Remove if all jobs accessed
                cached_batches.remove(next_batch)
        else:
            cached_batches.append(next_batch)
            cache_accesses_counter[next_batch] = 1

        # Track cache size over time
        time_steps.append(self.next_available_time)
        cache_size_over_time.append(len(cached_batches))

        # Check if we have reached the end of an epoch
        if self.batches_processed % batches_per_epoch == 0:
            print('job:', self.job_id, ',epoch:', self.batches_processed // batches_per_epoch, ',time:', self.next_available_time, ',cache size:', len(cached_batches))

        return True

def create_jobs(job_speeds: List[float]) -> List[Job]:
    return [Job(job_id, time_per_batch, total_batches) for job_id, time_per_batch in enumerate(job_speeds)]

def run():
    global jobs
    jobs = create_jobs(job_speeds)
    job_queue = []
    
    for job in jobs:
        heapq.heappush(job_queue, job)

    while job_queue:
        job = heapq.heappop(job_queue)  # Get the job with the earliest execution time
        
        if job.process_next_batch():
            heapq.heappush(job_queue, job)
        else:
            print(f"Job {job.job_id} has finished processing all batches after {job.next_available_time} seconds")
    
    # Print max cache size
    print(f"Max cache size: {max(cache_size_over_time)}")

if __name__ == "__main__":
    run()
