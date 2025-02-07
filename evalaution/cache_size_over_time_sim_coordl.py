import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Simulation parameters
batches_per_epoch = 10
total_epochs = 10
total_batches = batches_per_epoch * total_epochs
job_speeds = [0.125, 5.5]  # Time per batch (in seconds)
cached_batches = {}
cache_size_over_time = []  # Track cache size at each step
time_steps = []  # Track corresponding time values

class Job:
    def __init__(self, job_id, time_per_batch, total_batches):
        self.job_id = job_id
        self.time_per_batch = time_per_batch  
        self.batches_to_process = list(range(1, total_batches + 1))
        self.batches_processed = 0
        self.next_available_time = time_per_batch 
        self.wait_time = 0  
        self.epoch_completion_time = 0  
        self.reached_end_of_epoch = False
        self.epoch_delays: Dict[int, float] = {}
        self.current_epoch = 0  

    def __lt__(self, other):
        return self.next_available_time < other.next_available_time  

    def process_next_batch(self):
        if not self.batches_to_process:
            return False  # No more batches to process, prevent popping an empt
        """Process a batch and update cache usage."""
        next_batch = self.batches_to_process.pop(0)
        self.batches_processed += 1
        self.next_available_time += self.time_per_batch  
        print(f"[Time {self.next_available_time:.2f}] Job {self.job_id} processed batch {next_batch}")

        # Cache management
        if next_batch in cached_batches:
            cached_batches[next_batch] += 1
            if cached_batches[next_batch] >= 2:  # Remove if both jobs accessed
                del cached_batches[next_batch]
        else:
            cached_batches[next_batch] = 1

        # Epoch tracking
        if self.batches_processed % batches_per_epoch == 0:
            self.reached_end_of_epoch = True
            self.epoch_completion_time = self.next_available_time
            self.current_epoch += 1

        # Track cache size over time
        time_steps.append(self.next_available_time)
        cache_size_over_time.append(len(cached_batches))

    def log_epoch_delay(self, last_epoch_time):
        """Record how much delay this job experienced waiting for others."""
        delay = last_epoch_time - self.epoch_completion_time
        self.epoch_delays[self.current_epoch] = delay
        self.next_available_time = last_epoch_time + self.time_per_batch
        print(f"Job {self.job_id} delayed by {delay:.2f} seconds in epoch {self.current_epoch}")

def create_jobs(job_speeds: List[float]) -> List[Job]:
    return [Job(job_id, time_per_batch, total_batches) for job_id, time_per_batch in enumerate(job_speeds)]

def run():
    jobs = create_jobs(job_speeds)
    job_queue: List[Job] = []
    for job in jobs:
        heapq.heappush(job_queue, job)

    training_finished = False
    while not training_finished:
        training_finished = True  # Assume training is done unless proven otherwise

        if job_queue:
            training_finished = False  # There are jobs still running
            job = heapq.heappop(job_queue)  # Get the job with the earliest available time

            # **Check if the job has finished all its batches**
            if len(job.batches_to_process) == 0:
                print(f"Job {job.job_id} has no more batches to process.")
                continue  # Skip processing and do not re-add it to the queue

            job.process_next_batch()

            if job.reached_end_of_epoch:
                print(f"Job {job.job_id} finished epoch {job.batches_processed // batches_per_epoch} at time {job.epoch_completion_time}")

                # Check if all jobs finished the epoch
                all_jobs_finished_epoch = all(j.reached_end_of_epoch for j in jobs)

                if all_jobs_finished_epoch:
                    last_job_to_finish_epoch_time = max(j.epoch_completion_time for j in jobs)
                    print(f"All jobs finished epoch {job.batches_processed // batches_per_epoch}")

                    for other_job in jobs:
                        other_job.reached_end_of_epoch = False
                        other_job.next_available_time = last_job_to_finish_epoch_time + other_job.time_per_batch
                        other_job.epoch_delays[other_job.current_epoch] = last_job_to_finish_epoch_time - other_job.epoch_completion_time
                        other_job.epoch_completion_time = 0

                    # Re-add jobs to the queue only if they have more batches to process
                    for job in jobs:
                        if len(job.batches_to_process) > 0:
                            heapq.heappush(job_queue, job)
            else:
                heapq.heappush(job_queue, job)

    print(f"Max cache size: {max(cache_size_over_time)}")


if __name__ == "__main__":
    run()
