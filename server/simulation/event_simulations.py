import random
from collections import deque
from threading import Lock, Thread
import time
import heapq

total_batches = 50

class Event:
    def __init__(self, time, event_type, job_id, minibatch_id):
        self.time = time  # Time at which event occurs
        self.event_type = event_type  # Type of event (e.g., processing, creation, eviction)
        self.job_id = job_id  # Job that generated the event
        self.minibatch_id = minibatch_id  # The minibatch associated with the event

    def __lt__(self, other):
        return self.time < other.time  # Compare events by time

class StagingArea:
    def __init__(self):
        self.staging_area = deque()  # Holds minibatches in a queue
        self.minibatch_usage = {}  # Keeps track of how many times each minibatch has been processed
        self.lock = Lock()  # To ensure thread-safety when accessing the staging area
        self.evicted_count = 0  # To track the number of evicted minibatches
        self.max_count  = 0

    def add_minibatch(self, minibatch_id):
        with self.lock:
            if minibatch_id not in self.minibatch_usage:
                self.staging_area.append(minibatch_id)
                if len(self.staging_area) > self.max_count:
                    self.max_count = len(self.staging_area)
                self.minibatch_usage[minibatch_id] = 0

    def fetch_minibatch(self, processed_by):
        with self.lock:
            for minibatch_id in list(self.staging_area):
                if minibatch_id not in processed_by:
                    self.minibatch_usage[minibatch_id] += 1
                    if self.minibatch_usage[minibatch_id] == 4:  # All jobs processed
                        self.evicted_count += 1
                        self.staging_area.remove(minibatch_id)
                    return minibatch_id
            return None

    def get_size(self):
        return len(self.staging_area)

class Job:
    def __init__(self, job_id, total_jobs, minibatches, staging_area, job_speed, event_queue):
        self.job_id = job_id
        self.total_jobs = total_jobs
        self.minibatches = minibatches
        self.processed_by = set()
        self.staging_area = staging_area
        self.job_speed = job_speed
        self.event_queue = event_queue

    def create_minibatch(self):
        for minibatch_id in self.minibatches:
            if minibatch_id not in self.processed_by:
                self.staging_area.add_minibatch(minibatch_id)
                break

    def process_minibatches(self):
        current_time = 0
        while len(self.processed_by) < total_batches:
            minibatch_id = self.fetch_minibatch()
            if minibatch_id:
                self.processed_by.add(minibatch_id)
            else:
                self.create_minibatch()

            current_time += self.job_speed
            event = Event(current_time, "process", self.job_id, minibatch_id)
            heapq.heappush(self.event_queue, event)
            time.sleep(self.job_speed)

    def fetch_minibatch(self):
        return self.staging_area.fetch_minibatch(self.processed_by)

def simulate_jobs_processing():
    total_jobs = 4
    minibatches = [f"minibatch_{i+1}" for i in range(total_batches)]
    
    staging_area = StagingArea()
    event_queue = []  # Min-heap to process events in order of time
    jobs = []

    job_speeds = [random.uniform(0.001, 5) for _ in range(total_jobs)]  # Job speeds

    for job_id in range(total_jobs):
        job_minibatches = minibatches[job_id * (total_batches // total_jobs):(job_id + 1) * (total_batches // total_jobs)]
        job = Job(job_id, total_jobs, job_minibatches, staging_area, job_speeds[job_id], event_queue)
        jobs.append(job)

    # Create and start threads for each job
    threads = []
    for job in jobs:
        thread = Thread(target=job.process_minibatches)
        threads.append(thread)
        thread.start()

    # Process events in order of time
    while event_queue:
        event = heapq.heappop(event_queue)
        if event.event_type == "process":
            print(f"Time {event.time}: Job {event.job_id} processed minibatch {event.minibatch_id}")
        
    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print(f"Final staging area size: {staging_area.get_size()}")
    print(f"Max size staging area: {staging_area.max_count}")

if __name__ == "__main__":
    simulate_jobs_processing()
