import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Simulation parameters
batches_per_epoch = 100
total_epochs = 1
total_batches = batches_per_epoch * total_epochs
job_speeds = [5, 0.25]  # Time per batch (in seconds)
cached_batches = []
cache_accesses_counter = {}

class Job:
    def __init__(self, job_id, time_per_batch, total_batches):
        self.job_id = job_id
        self.time_per_batch = time_per_batch  # Now represents time taken per batch
        self.expected_duration_sec = total_batches * time_per_batch
        self.batches_to_process = list(range(1, total_batches + 1))
        self.batches_processed = 0
        self.next_available_time = 0  # Tracks when the job can process the next batch

    def __str__(self):
        return f"Job {self.job_id} with time per batch {self.time_per_batch}s has processed {self.batches_processed} batches"

def create_jobs(job_speeds: List[float]) -> List[Job]:
    return [Job(job_id, time_per_batch, total_batches) for job_id, time_per_batch in enumerate(job_speeds)]

def run():
    jobs = create_jobs(job_speeds)
    max_job_duration = max(job.expected_duration_sec for job in jobs)
    
    # Simulation loop
    for time_step in range(int(max_job_duration) + 1):
        print(f"Processing batches at time step {time_step}")
        jobs_to_remove = []

        for job in jobs:
            if job.batches_to_process and time_step >= job.next_available_time:
                next_batch = job.batches_to_process.pop(0)
                job.batches_processed += 1
                job.next_available_time = time_step + job.time_per_batch  # Schedule next batch processing time

                if next_batch in cached_batches:
                    cache_accesses_counter[next_batch] = cache_accesses_counter.get(next_batch, 0) + 1
                    if cache_accesses_counter[next_batch] >= len(jobs):
                        cached_batches.remove(next_batch)
                else:
                    cached_batches.append(next_batch)
                    cache_accesses_counter[next_batch] = cache_accesses_counter.get(next_batch, 0) + 1

            if not job.batches_to_process:
                print(f"Job {job.job_id} has finished processing all batches after {time_step} seconds")
                jobs_to_remove.append(job)

        # Remove finished jobs
        for job in jobs_to_remove:
            jobs.remove(job)

    print(f"Total batches processed: {sum(job.batches_processed for job in jobs)}")
    print(f"Total batches cached: {len(cached_batches)}")

if __name__ == "__main__":
    run()
