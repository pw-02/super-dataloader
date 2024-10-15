import time
import concurrent.futures
from typing import Tuple
from central_batch_manager import Dataset, CentralBatchManager
import logging
from args import SUPERArgs

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Constants
MISS_WAIT_FOR_DATA_TIME = 0.659
HIT_WAIT_FOR_DATA_TIME = 0.0008
PREFETCH_TIME = 2
NUM_JOBS = 10 # Number of parallel jobs to simulate
DELAY_BETWEEN_JOBS = 0  # Delay in seconds between the start of each job
BATCHES_PER_JOB = 500  # Number of batches each job will process
GPU_TIME = 0.342
# PREPROCESS_TIME_ON_HIT = 0.001
# PREPROCESS_TIME_ON_MISS = 0.001

super_args:SUPERArgs = SUPERArgs(
            batch_size = 128,
            partitions_per_dataset = 1,
            lookahead_steps = 1000,
            serverless_cache_address = '',
            use_prefetching = True,
            use_keep_alive = False,
            prefetch_lambda_name = 'CreateVisionTrainingBatch',
            prefetch_cost_cap_per_hour=None,
            cache_evition_ttl_threshold = 1000,
            prefetch_simulation_time = PREFETCH_TIME,
            evict_from_cache_simulation_time = None,
            shuffle = False,
            drop_last = False,
            workload_kind = 'vision')

# im1k = 's3://imagenet1k-sdl/train/'
cf10 = 's3://sdl-cifar10/train/'
dataset = Dataset(data_dir=cf10, batch_size=128, drop_last=False, num_partitions=super_args.partitions_per_dataset, kind=super_args.workload_kind)
batch_manager = CentralBatchManager(dataset=dataset, args=super_args)


def simulate_training_job(job_id: str) -> Tuple[str, int, int, float]:
    """
    Simulates a training job that fetches batches from a dataset with cache hits and misses.

    :param job_id: A unique identifier for the job.
    :return: A tuple containing job_id, number of cache hits, number of cache misses, and total duration.
    """
    epoch = 1
    cache_hits = 0
    cache_misses = 0
    start_time = time.perf_counter()  # Start time for job duration measurement
    # Process each batch for the job
    for i in range(BATCHES_PER_JOB):
        batch = batch_manager.get_next_batch(job_id)
        if batch.is_cached:
            previous_step_wait_for_data_time = HIT_WAIT_FOR_DATA_TIME
            previous_step_is_cache_hit = True
            cache_hits += 1
            time.sleep(previous_step_wait_for_data_time + GPU_TIME)
            cached_missed_batch = False
        else:
            previous_step_wait_for_data_time = MISS_WAIT_FOR_DATA_TIME
            previous_step_is_cache_hit = False
            cache_misses += 1
            cached_missed_batch = False
            time.sleep(previous_step_wait_for_data_time + GPU_TIME)
           
        batch_manager.update_job_progess(job_id, batch.batch_id, previous_step_wait_for_data_time, previous_step_is_cache_hit, GPU_TIME, cached_missed_batch)
        hit_rate = cache_hits / (i + 1) if (i + 1) > 0 else 0
        if i % 1== 0 or not previous_step_is_cache_hit:
            logger.info(f'Epoch: {epoch}, Setp {i+1}, Job {job_id}, {batch.batch_id}, Hits: {cache_hits}, Misses: {cache_misses}, Rate: {hit_rate:.2f}')
        if i +1 == 391:
            epoch += 1
    # Stop prefetcher and compute total duration
    total_duration = time.perf_counter() - start_time
    batch_manager.job_ended(job_id)
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
    time.sleep(5)
