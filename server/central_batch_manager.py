import threading
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from sampling import BatchSampler, EndOfPartitionException
from dataset import Dataset, DatasetPartition
from batch import Batch, BatchSet
import time
from logger_config import logger
from utils import AverageMeter, find_optimal_prefetch_conncurrency, gen_partition_epoch_key
from aws_utils import get_average_lambda_duration, get_total_lambda_invocations, get_memory_allocation_of_lambda, compute_lambda_cost
import concurrent.futures
import boto3
import json
from botocore.config import Config
import queue
from datetime import datetime, timedelta, timezone
import redis
import copy
from itertools import cycle  # Import cycle from itertools
from typing import Iterator, Optional, Set
from job import DLTJob
from args import SUPERArgs
import math
class PrefetchService:
    def __init__(self, 
                 prefetch_lambda_name: str, 
                 cache_address: str, 
                 jobs:Dict[str, DLTJob], 
                 dataset: Dataset,
                 cost_threshold_per_hour: float = 10,
                 simulate_time: float = None):
        
        self.prefetch_lambda_execution_times = AverageMeter('Lambda Execution Time')
        self.prefetch_cycle_times = AverageMeter('Prefetch Cycle Time')
        self.prefetch_stop_event:threading.Event = threading.Event()
        self.prefetch_stop_event.set()
        self.cache_address:str = cache_address
        self.simulate_time:float = simulate_time
        self.lock = threading.Lock()
        self.jobs:Dict[str, DLTJob] = jobs
        self.prefetch_delay:float = 0
        self.cost_threshold_per_hour = cost_threshold_per_hour
        self.start_time = None
        self.start_time_utc = datetime.now(timezone.utc)
        self.dataset = dataset

        self.prefetch_lambda_name = prefetch_lambda_name
        self.prefetch_lambda_client = boto3.client('lambda', config= Config(max_pool_connections=50))
        self.prefetch_lambda_configured_memory = get_memory_allocation_of_lambda(self.prefetch_lambda_name)
        self.prefetch_lambda_invocations_count = 0

        # self.prefetch_lambda_execution_times.update(get_average_lambda_duration(self.prefetch_lambda_name))
        pass

    def start_prefetcher(self):
        self.start_time = time.perf_counter()
        self.prefetch_stop_event.clear()
        prefetch_thread = threading.Thread(target=self._prefetching_process)
        prefetch_thread.daemon = True
        prefetch_thread.start()

    def _compute_opttiomal_prefetch_lookahead(self, data_delay, job_gpu_time):

        #training job can process at batch in 'job_gpu_time' seconds, so in 'data_delay' seconds it can process..
        potential_batches_during_delay = data_delay / job_gpu_time #this is the number of batches that can could be processed duing the delay
        #o ensure there is no delay, the job should be able to process at least 'potential_batches' within delay time
        #if it takes the prefecher 'sef.prefecth_cycle_time' to prfetch a batch, then requrired concurrency is...
        batches_loaded_per_conccurency_unit =  data_delay / self.prefetch_cycle_times.avg
        required_concurrency = potential_batches_during_delay / batches_loaded_per_conccurency_unit 
        return required_concurrency

    def _prefetching_process(self):
        while not self.prefetch_stop_event.is_set():
            try:
                prefetch_cycle_started = time.perf_counter()
                prefetch_list: Set[Tuple[Batch, str]] = set()        
                #prefetch_cycle_duration = self.prefetch_cycle_times.avg if self.prefetch_cycle_times.count > 0 else self.simulate_time if self.simulate_time else 3
                for job in self.jobs.values():
                    if job.total_steps <= 1: #ignore first two steps for GPU warm up
                        continue

                    max_bacthes_per_second = math.ceil(1 / job.training_step_gpu_times.avg)
                    no_caching_batches_per_second =  math.floor(1 / job.dataload_time_on_miss.avg) if job.dataload_time_on_miss.count > 0 else 0
                    required_prefetch_bacthes_per_second = max_bacthes_per_second - no_caching_batches_per_second

                    if required_prefetch_bacthes_per_second < 1:
                        continue
                    
                    prefetch_cycle_duration = self.prefetch_cycle_times.avg + self.prefetch_delay if self.prefetch_cycle_times.count > 0 else self.simulate_time if self.simulate_time else 3
                    prefetch_conncurrency =  math.ceil(required_prefetch_bacthes_per_second * prefetch_cycle_duration)

                    #add in a check to see if the job is suffering from a data loading delay and benefit from prefetching
                    prefetch_counter, time_counter = 0, 0
                    # Fetch average times for cache hit and miss scenarios for the current job
                    avg_time_on_hit = job.training_step_times_on_hit.avg if job.training_step_times_on_hit.count > 0 else job.training_step_gpu_times.avg
                    avg_time_on_miss = job.training_step_times_on_miss.avg if job.training_step_times_on_miss.count > 0 else job.training_step_gpu_times.avg

                    if len(job.future_batches) < prefetch_conncurrency:
                        logger.info(f"Job '{job.job_id}' has {len(job.future_batches)} batches, less than the required prefetch concurrency of {prefetch_conncurrency}.")

                    # Iterate over future batches to determine access during the prefetch cycle duration
                    job_batches_snapshot = list(job.future_batches.values())
                    for batch in job_batches_snapshot:
                        if time_counter <= prefetch_cycle_duration:
                                # If accessed within the cycle, add its time to the counter
                                if batch.is_cached or batch.caching_in_progress:
                                    time_counter += avg_time_on_hit
                                else:
                                    time_counter += avg_time_on_miss
                                continue
                        
                        elif prefetch_counter >= prefetch_conncurrency:
                            break

                        else: 
                            prefetch_counter += 1
                            if not batch.is_cached and not batch.caching_in_progress:
                                batch.set_caching_in_progress(True)
                                payload = {
                                    'bucket_name': self.dataset.bucket_name,
                                    'batch_id': batch.batch_id,
                                    'batch_samples': self.dataset.get_samples(batch.indicies),
                                    'cache_address': self.cache_address,
                                    'task': 'prefetch',
                                }
                                prefetch_list.add((batch, json.dumps(payload)))
                  
                # Submit the prefetch list for processing
                if prefetch_list:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Create client if not already initialized
                        if self.prefetch_lambda_client is None:
                            self.prefetch_lambda_client = boto3.client('lambda', config= Config(max_pool_connections=50))

                        # Calculate the delay before invoking Lambdas
                        delay_time = self._compute_delay_to_satisfy_cost_threshold() * len(prefetch_list)

                        if delay_time > 0:
                            logger.info(f"Delaying prefetching by {delay_time:.5f} seconds to satisfy cost threshold.")
                            time.sleep(delay_time)
                        
                        if self.simulate_time:
                            time.sleep(self.simulate_time)

                        # Map each future to its corresponding (batch, payload) tuple
                        future_to_batch_payload = {executor.submit(self._prefetch_batch, payload): (batch, payload) for batch, payload in prefetch_list}

                        for future in concurrent.futures.as_completed(future_to_batch_payload):
                            batch, payload = future_to_batch_payload[future]
                            try:
                                response = future.result()
                                batch.set_caching_in_progress(in_progress=False)
                                
                                self.prefetch_lambda_execution_times.update(response['execution_time'])
                                # self.prefetch_lambda_invocations_count += 1
                                
                                if 'success' in response.keys() and response['success']:
                                    # print(f"Batch '{batch.batch_id}' has been prefetched.")
                                    batch.set_cache_status(is_cached=True)
                                else:
                                    batch.set_cache_status(is_cached=False)
                                    if 'message' in response.keys():
                                        logger.error(f"Error prefetching batch '{batch.batch_id}': {response['message']}")
                                    else:
                                        logger.error(f"Error prefetching batch '{batch.batch_id}'.")
                                # print(f'Invocation response: {response}')
                            except Exception as e:
                                logger.error(f"Error in prefetching batch: {e}", exc_info=True)
                    self.prefetch_lambda_invocations_count += len(prefetch_list)
                    self.prefetch_cycle_times.update(time.perf_counter() - prefetch_cycle_started + delay_time)
                    logger.info(f"Prefetch took: {self.prefetch_cycle_times.val:.4f}s for {len(prefetch_list)} batches. (Avg Prefetch Time: {self.prefetch_cycle_times.avg:.4f}s, Avg Lambda Time: {self.prefetch_lambda_execution_times.avg:.4f}s, Running Cost: ${self._compute_prefeteching_cost():.4f})")
            
                if len(self.jobs) == 0 or len(prefetch_list) == 0:
                    time.sleep(0.1)  # Sleep for a short while before checking again

            except Exception as e:
                logger.error(f"Unexpected error in prefetching process: {e}", exc_info=True)

    
    # def _prefetching_process(self):
    #     while not self.prefetch_stop_event.is_set():
    #         try:
    #             prefetch_cycle_started = time.perf_counter()
    #             prefetch_list: Set[Tuple[Batch, str]] = set()        
    #             #prefetch_cycle_duration = self.prefetch_cycle_times.avg if self.prefetch_cycle_times.count > 0 else self.simulate_time if self.simulate_time else 3
    #             for job in self.jobs.values():
    #                 if job.total_steps <= 1: #ignore first two steps for GPU warm up
    #                     continue

    #                 max_bacthes_per_second = math.ceil(1 / job.training_step_gpu_times.avg)
    #                 no_caching_batches_per_second =  math.floor(1 / job.dataload_time_on_miss.avg) if job.dataload_time_on_miss.count > 0 else 0
    #                 required_prefetch_bacthes_per_second = max_bacthes_per_second - no_caching_batches_per_second

    #                 if required_prefetch_bacthes_per_second < 1:
    #                     continue
                    
    #                 prefetch_cycle_duration = self.prefetch_cycle_times.avg + self.prefetch_delay if self.prefetch_cycle_times.count > 0 else self.simulate_time if self.simulate_time else 3
    #                 prefetch_conncurrency =  math.ceil(required_prefetch_bacthes_per_second * prefetch_cycle_duration)

    #                 #add in a check to see if the job is suffering from a data loading delay and benefit from prefetching
    #                 prefetch_counter, time_counter = 0, 0
    #                 # Fetch average times for cache hit and miss scenarios for the current job
    #                 avg_time_on_hit = job.training_step_times_on_hit.avg if job.training_step_times_on_hit.count > 0 else job.training_step_gpu_times.avg
    #                 avg_time_on_miss = job.training_step_times_on_miss.avg if job.training_step_times_on_miss.count > 0 else job.training_step_gpu_times.avg

    #                 if len(job.future_batches) < prefetch_conncurrency:
    #                     logger.info(f"Job '{job.job_id}' has {len(job.future_batches)} batches, less than the required prefetch concurrency of {prefetch_conncurrency}.")

    #                 # Iterate over future batches to determine access during the prefetch cycle duration
    #                 job_batches_snapshot = list(job.future_batches.values())
    #                 for batch in job_batches_snapshot:
    #                     # if time_counter <= prefetch_cycle_duration:
    #                     #         # If accessed within the cycle, add its time to the counter
    #                     #         if batch.is_cached or batch.caching_in_progress:
    #                     #             time_counter += avg_time_on_hit
    #                     #         else:
    #                     #             time_counter += avg_time_on_miss
    #                     #         continue
                        
    #                     if prefetch_counter >= prefetch_conncurrency:
    #                         break

    #                     else: 
    #                         if not batch.is_cached and not batch.caching_in_progress:
    #                             prefetch_counter += 1
    #                             batch.set_caching_in_progress(True)
    #                             payload = {
    #                                 'bucket_name': self.dataset.bucket_name,
    #                                 'batch_id': batch.batch_id,
    #                                 'batch_samples': self.dataset.get_samples(batch.indicies),
    #                                 'cache_address': self.cache_address,
    #                                 'task': 'prefetch',
    #                             }
    #                             prefetch_list.add((batch, json.dumps(payload)))
                  
    #             # Submit the prefetch list for processing
    #             if prefetch_list:
    #                 with concurrent.futures.ThreadPoolExecutor() as executor:
    #                     # Create client if not already initialized
    #                     if self.prefetch_lambda_client is None:
    #                         self.prefetch_lambda_client = boto3.client('lambda', config= Config(max_pool_connections=50))

    #                     # Calculate the delay before invoking Lambdas
    #                     delay_time = self._compute_delay_to_satisfy_cost_threshold() * len(prefetch_list)

    #                     if delay_time > 0:
    #                         logger.info(f"Delaying prefetching by {delay_time:.5f} seconds to satisfy cost threshold.")
    #                         time.sleep(delay_time)
                        
    #                     if self.simulate_time:
    #                         time.sleep(self.simulate_time)

    #                     # Map each future to its corresponding (batch, payload) tuple
    #                     future_to_batch_payload = {executor.submit(self._prefetch_batch, payload): (batch, payload) for batch, payload in prefetch_list}

    #                     for future in concurrent.futures.as_completed(future_to_batch_payload):
    #                         batch, payload = future_to_batch_payload[future]
    #                         try:
    #                             response = future.result()
    #                             batch.set_caching_in_progress(in_progress=False)
                                
    #                             self.prefetch_lambda_execution_times.update(response['execution_time'])
    #                             # self.prefetch_lambda_invocations_count += 1
                                
    #                             if 'success' in response.keys() and response['success']:
    #                                 # print(f"Batch '{batch.batch_id}' has been prefetched.")
    #                                 batch.set_cache_status(is_cached=True)
    #                             else:
    #                                 batch.set_cache_status(is_cached=False)
    #                                 if 'message' in response.keys():
    #                                     logger.error(f"Error prefetching batch '{batch.batch_id}': {response['message']}")
    #                                 else:
    #                                     logger.error(f"Error prefetching batch '{batch.batch_id}'.")
    #                             # print(f'Invocation response: {response}')
    #                         except Exception as e:
    #                             logger.error(f"Error in prefetching batch: {e}", exc_info=True)
    #                 self.prefetch_lambda_invocations_count += len(prefetch_list)
    #                 self.prefetch_cycle_times.update(time.perf_counter() - prefetch_cycle_started + delay_time)
    #                 logger.info(f"Prefetch took: {self.prefetch_cycle_times.val:.4f}s for {len(prefetch_list)} batches. (Avg Prefetch Time: {self.prefetch_cycle_times.avg:.4f}s, Avg Lambda Time: {self.prefetch_lambda_execution_times.avg:.4f}s, Running Cost: ${self._compute_prefeteching_cost():.4f})")
            
    #             if len(self.jobs) == 0 or len(prefetch_list) == 0:
    #                 time.sleep(0.1)  # Sleep for a short while before checking again

    #         except Exception as e:
    #             logger.error(f"Unexpected error in prefetching process: {e}", exc_info=True)

    def _prefetch_batch(self, payload):
            if self.simulate_time:
                return {'success': True, 'message': None, 'execution_time': self.simulate_time}
            
            request_started = time.perf_counter()
            response = self.prefetch_lambda_client.invoke( FunctionName=self.prefetch_lambda_name,
                                                InvocationType='RequestResponse',
                                                Payload=payload)
            
            response_data = json.loads(response['Payload'].read().decode('utf-8'))
            response_data['execution_time'] = time.perf_counter() - request_started
            return response_data
    
    def _get_average_request_duration(self):
        pass
        # if start_time is not None:
        #     return get_average_lambda_duration(self.name, start_time=start_time, end_time = datetime.now(timezone.utc)) #average duration since prefetching started
        # else:
        #     return get_average_lambda_duration(self.name, 
        #                                        start_time=datetime.now(timezone.utc)  - timedelta(hours=4), #average duration for the last 4 hours
        #                                        end_time = datetime.now(timezone.utc))
    
    def _compute_prefetch_lambda_request_rate(self):
        with self.lock:
            elapsed_time = time.perf_counter() - self.start_time  # Calculate elapsed time
            if elapsed_time > 0:
                request_rate = self.prefetch_lambda_invocations_count / elapsed_time  # Compute request rate
                return request_rate
            return 0.0
        
    def _compute_delay_to_satisfy_cost_threshold(self):
        """Calculate delay based on current cost and the predefined cost threshold."""
        if self.cost_threshold_per_hour is None:
            return 0
        else:
            #  avg_request_duration = self.prefetch_lambda.get_average_request_duration(self.start_time_utc)
            current_prefetch_cost = self._compute_prefeteching_cost()
            if current_prefetch_cost == 0:
                return 0
            # logger.info(f"Current prefetch cost: {current_prefetch_cost:.16f}, Requests: {self.prefetch_lambda_invocations_count}, Execution time: {self.prefetch_lambda_execution_times.sum:.2f} seconds")
            
            cost_per_request =  current_prefetch_cost / self.prefetch_lambda_invocations_count
            request_rate = self._compute_prefetch_lambda_request_rate() 
            requests_per_hour = request_rate * 3600  # Convert request rate to requests per hour
            # Calculate the maximum allowable requests per hour within the cost threshold
            max_requests_per_hour = self.cost_threshold_per_hour / cost_per_request
            # If the current request rate is within the threshold, no delay is needed
            if requests_per_hour <= max_requests_per_hour:
                return 0  # No delay needed
    
            # Calculate the required delay in hours between each request to stay within the cost threshold
            delay_per_request_hours = (1 / max_requests_per_hour) - (1 / requests_per_hour)
            # Convert delay from hours to seconds for practical use
            delay_per_request_seconds = delay_per_request_hours * 3600
            self.prefetch_delay = max(delay_per_request_seconds, 0)
            return max(delay_per_request_seconds, 0)
        
    def _compute_prefeteching_cost(self):
        current_prefetch_cost = compute_lambda_cost(self.prefetch_lambda_invocations_count, self.prefetch_lambda_execution_times.sum, self.prefetch_lambda_configured_memory)
        return current_prefetch_cost
    
    def stop_prefetcher(self):
            if not self.prefetch_stop_event.is_set():
                self.prefetch_stop_event.set()
                logger.info(f"Prefetcher stopped. Total requests: {self.prefetch_lambda_invocations_count}, Total execution time: {self.prefetch_lambda_execution_times.sum:.2f}s, Total cost: ${self._compute_prefeteching_cost():.4f}")

class CacheEvictionService:
    def __init__(self, cache_address: str,  jobs:Dict[str, DLTJob], keep_alive_time_threshold:int = 1000, simulate_keep_alvive: bool = False):
        self.cache_address = cache_address
        self.cache_eviction_stop_event = threading.Event()  # Event to signal stopping
        self.keep_alive_time_threshold = keep_alive_time_threshold
        self.jobs:Dict[str, DLTJob] = jobs
        self.cache_eviction_stop_event.set()
        self.simulate_keep_alvive = simulate_keep_alvive
        self.redis_client = None
        self.lock = threading.Lock()

    def start_cache_evictor(self):
        self.cache_eviction_stop_event.clear()
        keep_alive_thread = threading.Thread(target=self._keep_alive_process)
        keep_alive_thread.daemon = True
        keep_alive_thread.start()
     
    def stop_cache_evictor(self):
        if not self.cache_eviction_stop_event.is_set():
            self.cache_eviction_stop_event.set()
            logger.info(f"Cache eviction service stopped")

    def _keep_alive_process(self):
        while not self.cache_eviction_stop_event.is_set():  # Check for stop signal
            try:
                cache_host, cache_port = self.cache_address.split(":")

                if self.redis_client is None and not self.simulate_keep_alvive:
                    self.redis_client = redis.StrictRedis(host=cache_host, port=cache_port)

                #Take a snapshot
                with self.lock:
                    jobs_snapshot = list(self.jobs.values())

                for job in jobs_snapshot:
                    if job.total_steps <= 1:
                        continue
                    # job_batches_snapshot = list(job.future_batches.values())
                    for batch in job.future_batches.values():
                        
                        if batch.time_since_last_access() > self.keep_alive_time_threshold:
                            try:
                                if self.simulate_keep_alvive:
                                        batch.set_cache_status(is_cached=True)
                                        batch.set_last_accessed_time()
                                else:
                                    if self.redis_client.get(batch.batch_id):  # Check if the batch is still cached
                                            batch.set_last_accessed_time()  # Update the last accessed time
                                            batch.set_cache_status(is_cached=True)
                                    else:
                                            # If the batch is not in Redis, mark it as not cached
                                            batch.set_cache_status(is_cached=False)
                                            logger.warning(f"Batch '{batch.batch_id}' is not in cache, setting is_cached to False.")
                            except Exception as e:
                                    logger.error(f"Error keeping batch '{batch.batch_id}' alive: {e}")
                                    batch.set_cache_status(is_cached=False)
                time.sleep(5)  # Sleep for a short while before checking the queue again
            
            except Exception as e:
                logger.error(f"Unexpected error in cache eviction process: {e}", exc_info=True)


class CentralBatchManager:
    def __init__(self, dataset: Dataset, args: SUPERArgs):
        self.dataset = dataset
        self.look_ahead = args.lookahead_steps  #min(args.lookahead_steps, self.dataset.partitions[1].num_batches)
        self.jobs: Dict[str, DLTJob] = {}
        self.active_epoch_idx = 1
        self.active_partition_id = None
        # Create a cycle iterator for partitions
        self.epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = OrderedDict()  #first key is epoch id, second key is partition id, value is the batches
        self.batch_sampler = BatchSampler(
            partitions=dataset.partitions.values(), 
            batch_size=self.dataset.batch_size, 
            shuffle=args.shuffle, 
            drop_last=args.drop_last)
        self.evict_from_cache_simulation_time = args.evict_from_cache_simulation_time

        # Generate initial batches
        for _ in range(self.look_ahead):
            self._generate_new_batch()
        
        self.use_keep_alive = args.use_keep_alive
        # Initialize prefetch service
        self.prefetch_service: Optional[PrefetchService] = None
        if args.use_prefetching:
            self.prefetch_service = PrefetchService(
                prefetch_lambda_name=args.prefetch_lambda_name,
                cache_address=args.serverless_cache_address,
                jobs=self.jobs,
                dataset=self.dataset,
                cost_threshold_per_hour=args.prefetch_cost_cap_per_hour,
                simulate_time=args.prefetch_simulation_time)
        if self.use_keep_alive:
            #Initialize cache eviction service
            self.cache_eviction_service:CacheEvictionService = CacheEvictionService(
                cache_address=args.serverless_cache_address,
                jobs=self.jobs,
                keep_alive_time_threshold=args.cache_evition_ttl_threshold,
                simulate_keep_alvive= True if self.evict_from_cache_simulation_time is not None else False
                # simulate_time=args.evict_from_cache_simulation_time
            )     

        self.lock = threading.Lock()  # Lock for thread safety
        
        # if self.prefetch_service is not None and self.prefetch_service.prefetch_stop_event.is_set():  
        #         self.prefetch_service.start_prefetcher() #prefetcher is stopped, start it
            
        # if self.use_keep_alive and self.cache_eviction_service.cache_eviction_stop_event.is_set():
        #         self.cache_eviction_service.start_cache_evictor()


    def _generate_new_batch(self):
        next_batch:Batch = next(self.batch_sampler)

        if self.evict_from_cache_simulation_time:
            next_batch.evict_from_cache_simulation_time = self.evict_from_cache_simulation_time

        if self.active_partition_id != next_batch.partition_id:
            self.active_partition_id = next_batch.partition_id

        if self.active_epoch_idx != next_batch.epoch_idx:
            self.active_epoch_idx = next_batch.epoch_idx

        partition_batches = self.epoch_partition_batches.setdefault(next_batch.epoch_idx, {})
        partition_batch_set = partition_batches.setdefault(next_batch.partition_id, BatchSet(f'{next_batch.epoch_idx}_{next_batch.partition_id}'))
        partition_batch_set.batches[next_batch.batch_id] = next_batch

        for job in self.jobs.values():
            if partition_batch_set.id in job.active_batch_set_ids:
                job.future_batches[next_batch.batch_id] = next_batch
            else:
                job.future_batches[next_batch.batch_id] = next_batch
                job.active_batch_set_ids.add(partition_batch_set.id)
                # job.active_batch_set_id = partition_batch_set.id
        
    # def allocate_batches_to_job(self, job: DLTJob):

    #     if job.partition_id_cycle is None: #new job, lets start cycling partitons at the currently active partition
    #         partition_ids = list(self.dataset.partitions.keys())
    #         start_index = partition_ids.index(self.active_partition_id)
    #         reordered_ids = partition_ids[start_index:] + partition_ids[:start_index]
    #         job.partition_id_cycle = cycle(reordered_ids)
    #         job.started_partition_index = copy.deepcopy(self.active_partition_id)
        
    #     next_partition_id = next(job.partition_id_cycle)
    #     if next_partition_id == job.started_partition_index:
    #         job.epochs_completed_count += 1

    #     #now find the last batch set for this partition, and make sure it hasn't been processed by the job before
    #     for epoch_id in reversed(self.epoch_partition_batches.keys()):
    #         if next_partition_id in self.epoch_partition_batches[epoch_id]:
    #             batch_set = self.epoch_partition_batches[epoch_id][next_partition_id]
    #             job.future_batches.update(batch_set.batches)
    #             job.active_batch_set_id = batch_set.id
    #             break

    def allocate_batches_to_job(self, job: DLTJob):
        
        if job.partition_id_cycle is None: #new job, lets start cycling partitons at the currently active partition
                partition_ids = list(self.dataset.partitions.keys())
                start_index = partition_ids.index(self.active_partition_id)
                reordered_ids = partition_ids[start_index:] + partition_ids[:start_index]
                job.partition_id_cycle = cycle(reordered_ids)
                job.started_partition_index = copy.deepcopy(self.active_partition_id)
        
        next_partition_id = next(job.partition_id_cycle)
        if next_partition_id == job.started_partition_index:
            job.epochs_completed_count += 1

        epoch_ids = list(self.epoch_partition_batches.keys())
        #check if current epoch has at least look_ahead batches
        if len(self.epoch_partition_batches[epoch_ids[-1]]) >= self.look_ahead:
            if next_partition_id in self.epoch_partition_batches[epoch_id]:
                batch_set = self.epoch_partition_batches[epoch_id][next_partition_id]
                job.future_batches.update(batch_set.batches)
                job.active_batch_set_ids.add(batch_set.id)
                # job.active_batch_set_id = batch_set.id
        else:

            #we need to find the last 'look_ahead' batches in the global epoch_partition_batches
            while len(job.future_batches) < self.look_ahead:
                epoch_id = epoch_ids.pop()
                partition_ids = list(self.epoch_partition_batches[epoch_id].keys())
                for partition_id in partition_ids:
                    batch_set = self.epoch_partition_batches[epoch_id][partition_id]
                    job.active_batch_set_ids.add(batch_set.id)
                    for batch in reversed(batch_set.batches.values()):
                        job.future_batches[batch.batch_id] = batch
                        job.future_batches.move_to_end(batch.batch_id, last=False)


    def update_job_progess(self, 
                           job_id,
                           previous_step_batch_id,
                           previous_step_wait_for_data_time,
                           previous_step_is_cache_hit,
                           previous_step_gpu_time,
                           previous_batch_cached_on_miss):

     with self.lock:
        parts = previous_step_batch_id.split('_')
        epoch_id = int(parts[0])
        partition_id = int(parts[1])
        batch = self.epoch_partition_batches[epoch_id][partition_id].batches[previous_step_batch_id]
        if previous_step_is_cache_hit or previous_batch_cached_on_miss:
            batch.set_last_accessed_time()
            batch.set_cache_status(True)
        else:
            batch.set_cache_status(False)

        self.jobs[job_id].update_perf_metrics(previous_step_wait_for_data_time, previous_step_is_cache_hit, previous_step_gpu_time)



    def get_next_batch(self, job_id: str) -> Optional[Batch]:
        
        with self.lock:

            if self.prefetch_service is not None and self.prefetch_service.prefetch_stop_event.is_set():  
                self.prefetch_service.start_prefetcher() #prefetcher is stopped, start it
            
            if self.use_keep_alive and self.cache_eviction_service.cache_eviction_stop_event.is_set():
                self.cache_eviction_service.start_cache_evictor()
            
            if job_id not in self.jobs:
                logger.info(f"Registering job '{job_id}'")
            job = self.jobs.setdefault(job_id, DLTJob(job_id))
            
            if not job.future_batches or len(job.future_batches) < self.look_ahead:
            # if not job.future_batches or len(job.future_batches) == 0: #reached end of partition so start preparing for next one
                self.allocate_batches_to_job(job)
            
            next_batch:Batch = job.next_training_step_batch()
            
            if not next_batch.is_cached:
                next_batch.set_caching_in_progress(True)
            
            if not next_batch.has_been_accessed_before:
                next_batch.set_has_been_accessed_before(True)
                self._generate_new_batch()

            # logger.info(f"Job '{job_id}' given batch '{next_batch.batch_id}' from partition '{next_batch.partition_id}' in epoch '{next_batch.epoch_idx}'")
            return next_batch
        
    def job_ended(self, job_id):
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                logger.info(f"Job '{job_id}' ended. Total time: {job.total_training_time():.4f}s, Steps: {job.total_training_steps()}, Hits: {job.training_step_times_on_hit.count}, Misses: {job.training_step_times_on_miss.count}, Rate: {job.training_step_times_on_hit.count / job.total_training_steps():.4f}" )
                self.jobs.pop(job_id)

            if len(self.jobs) == 0:
                # logger.info("All jobs have ended. Stopping prefetcher.")
                if self.prefetch_service:
                    self.prefetch_service.stop_prefetcher()
    
if __name__ == "__main__":
    TIME_ON_CACHE_HIT = 0.5
    TIME_ON_CACHE_MISS = 8
    PREFETCH_TIME = 5
    BATCH_SIZE = 128
    super_args = SUPERArgs(
        lookahead_steps = 100,
        batch_size = BATCH_SIZE,
        use_prefetching = True,
        prefetch_lambda_name = 'CreateVisionTrainingBatch',
        serverless_cache_address = '10.0.28.76:6378',
        prefetch_cost_cap_per_hour = None,
        prefetch_simulation_time = PREFETCH_TIME,
        evict_from_cache_simulation_time=None,
        partitions_per_dataset = 1,
        cache_evition_ttl_threshold=2,
        use_keep_alive=False,
        drop_last=False,
        shuffle=False)

    dataset = Dataset(data_dir='s3://sdl-cifar10/train/', batch_size=super_args.batch_size, drop_last=super_args.drop_last, num_partitions=super_args.partitions_per_dataset)
    batch_manager = CentralBatchManager(dataset=dataset, args=super_args)
    
    job_id = '1'
    cache_hits = 0
    cache_misses = 0
    previous_step_total_time = 0
    previous_step_is_cache_hit = False
    cached_previous_batch = False

    end = time.perf_counter()

    for i in range(100):
        batch:Batch = batch_manager.get_next_batch(
            job_id=job_id,
            previous_step_total_time=previous_step_total_time, 
            previous_step_is_cache_hit=previous_step_is_cache_hit,
            previous_step_gpu_time=TIME_ON_CACHE_HIT,
            cached_previous_batch=cached_previous_batch)

        if batch.is_cached:
            previous_step_is_cache_hit = True
            cache_hits += 1
            cached_previous_batch = False
            time.sleep(TIME_ON_CACHE_HIT)
            previous_step_total_time =TIME_ON_CACHE_HIT
        else:
            previous_step_is_cache_hit = False
            cache_misses += 1
            cached_previous_batch = True
            time.sleep(TIME_ON_CACHE_MISS)
            previous_step_total_time =TIME_ON_CACHE_MISS
        
        hit_rate = cache_hits / (i + 1) if (i + 1) > 0 else 0
        print(f'Batch {i+1}: {batch.batch_id}, Cache Hits: {cache_hits}, Cache Misses:{cache_misses}, Hit Rate: {hit_rate:.2f}')
        
    batch_manager.job_ended(job_id, previous_step_total_time, previous_step_is_cache_hit,TIME_ON_CACHE_HIT, cached_previous_batch)

    time.sleep(30)