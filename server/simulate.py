import time
import concurrent.futures
from typing import Tuple
from central_batch_manager import PrefetchManager, Dataset, CentralBatchManager, Batch
import logging

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Constants
TIME_ON_CACHE_HIT = 0.025
TIME_ON_CACHE_MISS = 5
NUM_JOBS = 4# Number of parallel jobs to simulate
LOOK_AHEAD = 50  # Number of batches each job will process
DELAY_BETWEEN_JOBS = 2  # Delay in seconds between the start of each job

BATCHES_PER_JOB = 500  # Number of batches each job will process
partitions_per_dataset = 1
# Shared instances initialized once
prefetcher = PrefetchManager('CreateVisionTrainingBatch', '10.0.28.76:6378', None, PREFETCH_TIME)
dataset = Dataset(data_dir='s3://sdl-cifar10/test/', batch_size=128, drop_last=False, num_partitions=partitions_per_dataset)
batch_manager = CentralBatchManager(dataset=dataset, look_ahead=LOOK_AHEAD, prefetch_service=prefetcher, cache_eviction_service=None)


def simulate_training_job(job_id: str) -> Tuple[str, int, int, float]:
    """
    Simulates a training job that fetches batches from a dataset with cache hits and misses.

    :param job_id: A unique identifier for the job.
    :return: A tuple containing job_id, number of cache hits, number of cache misses, and total duration.
    """

    cache_hits = 0
    cache_misses = 0
    previous_step_training_time = 0
    previous_step_is_cache_hit = False
    cached_missed_batch = False

    start_time = time.perf_counter()  # Start time for job duration measurement

    # Process each batch for the job
    for i in range(BATCHES_PER_JOB):
        batch = batch_manager.get_next_batch(job_id, previous_step_training_time, previous_step_is_cache_hit, TIME_ON_CACHE_HIT, cached_missed_batch)
        if batch.is_cached:
            previous_step_is_cache_hit = True
            cache_hits += 1
            time.sleep(TIME_ON_CACHE_HIT)
            cached_missed_batch = False
            previous_step_training_time = TIME_ON_CACHE_HIT
        else:
            cached_missed_batch = True
            previous_step_is_cache_hit = False
            cache_misses += 1
            time.sleep(TIME_ON_CACHE_MISS)
            previous_step_training_time = TIME_ON_CACHE_MISS
        
        hit_rate = cache_hits / (i + 1) if (i + 1) > 0 else 0
        if i % 1== 0:
            logger.info(f'Job {job_id}, {batch.batch_id}, Hits: {cache_hits}, Misses: {cache_misses}, Rate: {hit_rate:.2f}')

    # Stop prefetcher and compute total duration
    total_duration = time.perf_counter() - start_time
    batch_manager.handle_job_ended(job_id, previous_step_training_time, previous_step_is_cache_hit)
    return job_id, cache_hits, cache_misses, total_duration, hit_rate


if __name__ == "__main__":
    # Using ThreadPoolExecutor to run jobs in parallel
    job_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_JOBS) as executor:
        futures = []
        
        for job_id in range(1, NUM_JOBS + 1):
            # Submit job with a delay
            logger.info(f"Starting job {job_id} after {DELAY_BETWEEN_JOBS * (job_id - 1)} seconds delay.")
            future = executor.submit(simulate_training_job, str(job_id))
            futures.append(future)
            
            # Delay the start of the next job
            time.sleep(DELAY_BETWEEN_JOBS)

        # Collect and log the results of each job
        for future in concurrent.futures.as_completed(futures):
            job_id, cache_hits, cache_misses, total_duration, hit_rate = future.result()
            job_results.append((job_id, cache_hits, cache_misses, total_duration, hit_rate))
            # job_results.append(f"Results for Job {job_id}: Cache Hits = {cache_hits}, Cache Misses = {cache_misses}, Duration = {total_duration:.2f} seconds, Hit Rate = {hit_rate:.2f}")
            # job_results.append(f"Results for Job {job_id}: Cache Hits = {cache_hits}, Cache Misses = {cache_misses}, Duration = {total_duration:.2f} seconds")

    total_hits = sum(result[1] for result in job_results)
    total_misses = sum(result[2] for result in job_results)
    totatal_ratio = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
   
    for result in job_results:
        job_id, cache_hits, cache_misses, total_duration, hit_rate = result
        logger.info(f"Results for Job {job_id}: Cache Hits = {cache_hits}, Cache Misses = {cache_misses}, Duration = {total_duration:.2f} seconds, Hit Rate = {hit_rate}")
    
    logger.info(f"Total Cache Hits = {total_hits}, Total Cache Misses = {total_misses}, Total Hit Rate = {totatal_ratio:.2f}")
    
    time.sleep(1)
