import time
from collections import defaultdict
import random

class Cache:
    def __init__(self, ttl):
        self.cache = {}
        self.ttl = ttl
        self.access_times = defaultdict(lambda: 0)
    
    def load(self, batch_id, data):
        """Load a batch into the cache."""
        self.cache[batch_id] = data
        self.access_times[batch_id] = time.time()
    
    def get(self, batch_id):
        """Retrieve a batch from the cache if it hasn't expired."""
        if batch_id in self.cache and (time.time() - self.access_times[batch_id]) < self.ttl:
            return self.cache[batch_id]
        return None
    
    def evict(self):
        """Remove expired batches from the cache."""
        current_time = time.time()
        to_evict = [batch_id for batch_id, last_access in self.access_times.items()
                    if (current_time - last_access) >= self.ttl]
        for batch_id in to_evict:
            del self.cache[batch_id]
            del self.access_times[batch_id]

def assign_batches(num_jobs, batch_set, step):
    """Assign batches to jobs in a round-robin fashion with rotation."""
    return [(i, (i + step) % len(batch_set)) for i in range(num_jobs)]

def simulate_training(num_training_steps, num_jobs, ttl, cost_per_request):
    """Simulate training with multiple jobs and a cache, considering different job speeds."""
    cache = Cache(ttl)
    
    # Define random processing speeds for each job
    job_speeds = [random.uniform(0.5, 2.0) for _ in range(num_jobs)]  # Processing times in seconds
    
    # Track cache hits, total accesses, and unique cache requests
    total_accesses = 0
    cache_hits = 0
    unique_requests = set()  # To track unique cache requests
    steps = 0  # Initialize steps
    
    # Simulate over multiple sets of batches
    for start_batch in range(0, num_training_steps, num_jobs):
        batch_set = list(range(start_batch, min(start_batch + num_jobs, num_training_steps)))
        
        print(f"Processing batch set: {batch_set}")
        
        # Incremental steps for each batch set
        for step in range(len(batch_set)):
            steps += 1
            assignments = assign_batches(num_jobs, batch_set, step)
            print(f"\nStep {steps}:")
            
            # Track job completion times
            completion_times = []
            
            # Each job processes the assigned batch
            for job_id, batch_index in assignments:
                batch_id = batch_set[batch_index]
                data = cache.get(batch_id)
                
                total_accesses += 1
                unique_requests.add(batch_id)  # Track unique requests
                
                if data:
                    cache_hits += 1
                    print(f"Job {job_id + 1} cache hit for Batch {batch_id}")
                else:
                    print(f"Job {job_id + 1} cache miss for Batch {batch_id}")
                    # Simulate loading the batch
                    data = f"Data for batch {batch_id}"
                    cache.load(batch_id, data)
                
                # Track when the job will finish processing
                processing_time = job_speeds[job_id]
                completion_time = time.time() + processing_time
                completion_times.append((completion_time, batch_id, job_id))
            
            # # Process jobs as their completion times arrive
            # for completion_time, batch_id, job_id in sorted(completion_times):
            #     # Wait until the job is done
            #     time.sleep(max(0, completion_time - time.time()))
            
            # Evict old batches from cache
            cache.evict()
        
    # Calculate and print cache hit percentage
    if total_accesses > 0:
        hit_percentage = (cache_hits / total_accesses) * 100
    else:
        hit_percentage = 0
    print(f"Finished processing batch set: {batch_set}")
    print(f"Total Accesses: {total_accesses}, Total Hits: {cache_hits}, Total Misses: {total_accesses - cache_hits}")

    print(f"Cache Hit Percentage: {hit_percentage:.2f}%")
    
    # Calculate and print cost
    num_unique_requests = len(unique_requests)
    cost = (num_unique_requests / 1_000_000) * cost_per_request
    print(f"Cache Cost: ${cost:.2f}\n")

# Configuration
NUM_TRAINING_STEPS = 10000  # Total number of batches
NUM_JOBS = 10  # Number of jobs
TTL = 15 * 60  # Cache TTL in seconds (15 minutes)
COST_PER_REQUEST = 0.20  # Cost per 1 million requests

# Run the simulation
simulate_training(NUM_TRAINING_STEPS, NUM_JOBS, TTL, COST_PER_REQUEST)
