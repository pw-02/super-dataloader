import random
from collections import deque
from threading import Lock
from threading import Lock, Thread
import time

total_batches = 27000

class StagingArea:
    def __init__(self):
        self.staging_area = deque()  # Holds minibatches in a queue
        self.minibatch_usage = {}  # Keeps track of how many times each minibatch has been processed
        self.lock = Lock()  # To ensure thread-safety when accessing the staging area
        self.evicted_count = 0  # To track the number of evicted minibatches
        self.max_count  = 0
    def add_minibatch(self, minibatch_id):
        with self.lock:
            # Only add the minibatch if it hasn't been added before
            if minibatch_id not in self.minibatch_usage:
                self.staging_area.append(minibatch_id)
                if len(self.staging_area) > self.max_count:
                    self.max_count = len(self.staging_area)
                    print(f"Max count: {self.max_count}")
                # Initialize its usage count if it's the first time it's being added
                self.minibatch_usage[minibatch_id] = 0  # Initialize count at 0, not 1

    def fetch_minibatch(self, processed_by):
        with self.lock:
            # Check for unprocessed mini-batches in the staging area
            for minibatch_id in list(self.staging_area):
                if minibatch_id not in processed_by:
                    # Increment the usage count for the minibatch
                    self.minibatch_usage[minibatch_id] += 1

                    # If all jobs have processed it, remove it from the staging area
                    if self.minibatch_usage[minibatch_id] == 4:  # 4 jobs total
                        self.evicted_count += 1
                        print(f"Evicted minibatch {minibatch_id} - {self.minibatch_usage[minibatch_id]},  total evictions: {self.evicted_count}, max count: {self.max_count}")
                        self.staging_area.remove(minibatch_id)
                    return minibatch_id
            return None  # No unprocessed minibatches found in staging area

    def get_size(self):
        # Returns the size of the staging area
        return len(self.staging_area)


class Job:
    def __init__(self, job_id, total_jobs, minibatches, staging_area, job_speed):
        self.job_id = job_id
        self.total_jobs = total_jobs
        self.minibatches = minibatches  # Each job has a unique list of 100 minibatches
        self.processed_by = set()  # To track processed minibatches
        self.staging_area = staging_area  # Shared cache (staging area)
        self.job_speed = job_speed  # Speed of the job in processing minibatches

    def create_minibatch(self):
        # Simulate job processing minibatches and adding them to the staging area
        for i in range(len(self.minibatches)):
            minibatch_id = self.minibatches[i]
            if minibatch_id not in self.processed_by:
                # Add only the first unprocessed minibatch to the staging area
                self.staging_area.add_minibatch(minibatch_id)
                break  # Stop after adding the first unprocessed minibatch


    def process_minibatches(self):
        # Simulate the processing of minibatches for the job
        while len(self.processed_by) <= total_batches :  # Each job processes all 400 minibatches
            minibatch_id = self.fetch_minibatch()
            if minibatch_id:
                pass
                # print(f"Job {self.job_id} processed minibatch {minibatch_id}")
            else:
                # print(f"Job {self.job_id} found no new minibatches to process, creating a new one")
                self.create_minibatch()
            self.processed_by.add(minibatch_id)
            #sleep for the job speed
            time.sleep(self.job_speed)


    def fetch_minibatch(self):
        # Fetch a minibatch from the staging area
        minibatch_id = self.staging_area.fetch_minibatch(self.processed_by)
        # if minibatch_id:
        #     # Mark the minibatch as processed by this job
        #     self.processed_by.add(minibatch_id)
        return minibatch_id

def simulate_jobs_processing():
    total_jobs = 4
    minibatches = [f"minibatch_{i+1}" for i in range(total_batches)]  # 400 minibatches in total
    
    # Shared staging area
    staging_area = StagingArea()

    jobs = []
    
    # Calculate minibatches per job (evenly split)
    minibatches_per_job = len(minibatches) // total_jobs
    remaining_minibatches = len(minibatches) % total_jobs
    job_speeds = [random.uniform(0.001, 5) for _ in range(total_jobs)]  # Random job speeds
    job_speeds = [0.0123, 0.0234, 0.0403, 0.05004]
       # Initialize jobs
    for job_id in range(total_jobs):
        # Assign minibatches to each job, splitting the batches evenly
        start_index = job_id * minibatches_per_job
        end_index = start_index + minibatches_per_job
        
        # If there are remaining minibatches, distribute them one by one to the first few jobs
        if job_id < remaining_minibatches:
            end_index += 1  # Add one extra minibatch to the first remaining jobs
        speed = job_speeds[job_id]
        job_minibatches = minibatches[start_index:end_index]
        job = Job(job_id, total_jobs, job_minibatches, staging_area, speed)
        jobs.append(job)

    # # Initialize jobs
    # for job_id in range(total_jobs):
    #     job_minibatches = minibatches[job_id * 100 : (job_id + 1) * 100]  # Assign 100 minibatches to each job
    #     job = Job(job_id, total_jobs, job_minibatches, staging_area)
    #     jobs.append(job)

    # Create and start threads for each job to run concurrently
    threads = []
    for job in jobs:
        thread = Thread(target=job.process_minibatches)
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


    # Track the final state of the staging area
    print(f"Final staging area size: {staging_area.get_size()}")
    print(f"Max size staging area: {staging_area.max_count}")
    # print(f"Final minibatch usage: {staging_area.minibatch_usage}")

if __name__ == "__main__":
    simulate_jobs_processing()
