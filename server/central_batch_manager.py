import threading
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from sampling import BatchSampler, EndOfPartitionException
from dataset import Dataset, DatasetPartition
from batch import Batch
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
from typing import Iterator, Optional

class BatchSet:
    def __init__(self, id:str):
        self.id = id
        self.batches: Dict[str, Batch] = OrderedDict()
        self.batches_finalized = False
        self.mark_for_eviction = False

class PrefetchLambda:
    def __init__(self, prefetch_lambda_name: str, cache_address: str, simulate: bool = False):
        self.name = prefetch_lambda_name
        self.cache_address = cache_address
        self.client = boto3.client('lambda', config= Config(max_pool_connections=50))
        self.redis_client = None
        self.configured_memory = get_memory_allocation_of_lambda(self.name)
        logger.info(f"Configured memory for lambda '{self.name}': {self.configured_memory}")
        self.request_counter = 0
        self.simulate = simulate
        self.lock = threading.Lock()
        # self.request_durations = AverageMeter('Request Duration')
    
    def create_client(self):
        self.client = boto3.client('lambda', config= Config(max_pool_connections=50))

    def prefetch_batch(self, item: Tuple[Batch, Dict[str, str]]):
        batch, payload = item
        if self.simulate:
            with self.lock:
                self.request_counter += 1
            batch.set_cache_status(is_cached=True)
            batch.set_caching_in_progress(in_progress=False) 
            return {'success': True, 'errorMessage': None}

        response = self.client.invoke( FunctionName=self.name,
                                             InvocationType='RequestResponse',
                                             Payload=json.dumps(payload))
        
        response_data = json.loads(response['Payload'].read().decode('utf-8'))

        with self.lock:
            self.request_counter += 1
        
        if response_data['success']:
            batch.set_cache_status(is_cached=True)
            batch.set_caching_in_progress(in_progress=False)
            # print(f"Prefetched Batch '{batch.batch_id}', total prefetched: {self.pefetched_batches_counter}")
            return response_data
        else:
            batch.set_caching_in_progress(in_progress=False)
            batch.set_cache_status(is_cached=False)
            logger.error(f"Error prefetching batch '{batch.batch_id}': {response_data['errorMessage']}")
            return response_data

    def get_average_request_duration(self):
        pass
        # if start_time is not None:
        #     return get_average_lambda_duration(self.name, start_time=start_time, end_time = datetime.now(timezone.utc)) #average duration since prefetching started
        # else:
        #     return get_average_lambda_duration(self.name, 
        #                                        start_time=datetime.now(timezone.utc)  - timedelta(hours=4), #average duration for the last 4 hours
        #                                        end_time = datetime.now(timezone.utc))
    
    def compute_request_rate(self, start_time:float):
        with self.lock:
            elapsed_time = time.perf_counter() - start_time  # Calculate elapsed time
            if elapsed_time > 0:
                request_rate = self.request_counter / elapsed_time  # Compute request rate
                return request_rate
            return 0.0


class CacheEvicitonService:
    def __init__(self, cache_address: str, keep_alive_time_threshold:int = 1000):
        self.cache_address = cache_address
        self.keep_alive_stop_event = threading.Event()  # Event to signal stopping
        self.keep_alive_queue: queue.Queue[Tuple[float, List[Tuple['Batch', Dict[str, str]]]]] = queue.Queue()  # Queue to store lists batches to be prefetched
        self.keep_alive_time_threshold = keep_alive_time_threshold
    def start_cache_evictor(self):
        keep_alive_thread = threading.Thread(target=self._keep_alive_process)
        keep_alive_thread.daemon = True
        keep_alive_thread.start()
    
    def _keep_alive_process(self):
        while not self.keep_alive_stop_event.is_set():  # Check for stop signal
            try:
                cache_host, cache_port = self.cache_address.split(":")

                if self.redis_client is None:
                    self.redis_client = redis.StrictRedis(host=cache_host, port=cache_port)
                # Get a batch from the queue with a timeout to handle stopping
                queued_time, set_of_batches = self.keep_alive_queue.get(timeout=1)
                for batch, payload in set_of_batches:
                    if batch.time_since_last_access() > self.keep_alive_time_threshold:
                        try:
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
            except queue.Empty:
                # Handle the empty queue case
                time.sleep(4)  # Sleep for a short while before checking the queue again
    


class PrefetchManager:
    def __init__(self, prefetch_lambda_name: str, cache_address: str, cost_threshold_per_hour: float = 10, simulate_time: float = None):
        self.prefetch_execution_times = AverageMeter('Prefetch Execution Time')
        self.prefetch_execution_times.update(simulate_time)  # Default value for the first time
        self.prefetch_stop_event = threading.Event()  # Event to signal preftch stopping
        self.prefetch_queue = queue.Queue()  # Queue to store lists batches to be prefetched
        self.cache_address = cache_address
        self.prefetch_lambda = PrefetchLambda(prefetch_lambda_name, cache_address, simulate = True if simulate_time is not None else False)
        self.max_prefetch_concurrency = 10000
        self.simulate_time = simulate_time
        self.cost_threshold_per_hour = cost_threshold_per_hour # 0.00000020272686997451454 * 2000
        self.prefetch_stop_event.set()
        self.start_time = None
        self.start_time_utc = datetime.now(timezone.utc)
        self.prefetch_delay = 0
        self.lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor()  # Reuse the executor

    def start_prefetcher(self):
        self.prefetch_stop_event.clear()
        self.start_time = time.perf_counter()
        prefetch_thread = threading.Thread(target=self._start_prefetching_process)
        prefetch_thread.daemon = True
        prefetch_thread.start()
    
    def calculate_delay(self):
        """Calculate delay based on current cost and the predefined cost threshold."""
        if self.cost_threshold_per_hour is None:
            return 0
        else:
            #  avg_request_duration = self.prefetch_lambda.get_average_request_duration(self.start_time_utc)
            current_total_cost = compute_lambda_cost(self.prefetch_lambda.request_counter, self.prefetch_execution_times.avg, self.prefetch_lambda.configured_memory)
            if current_total_cost == 0:
                return 0
            
            cost_per_request =  current_total_cost / self.prefetch_lambda.request_counter
            request_rate = self.prefetch_lambda.compute_request_rate(self.start_time) 
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
    
    def _start_prefetching_process(self):
        # Initialize a ThreadPoolExecutor to be reused for all prefetching tasks
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while not self.prefetch_stop_event.is_set():  # Check for stop signal
                try:
                    # Create client if not already initialized
                    if self.prefetch_lambda.client is None:
                        self.prefetch_lambda.create_client()

                    # Get a batch from the queue with a timeout to handle stopping
                    queued_time, set_of_batches = self.prefetch_queue.get(timeout=1)

                    # Calculate the delay before invoking Lambdas
                    delay_time = self.calculate_delay() * len(set_of_batches)
                    
                    # Delay based on the calculated time to manage costs
                    if delay_time > 0:
                        time.sleep(delay_time)

                    # Simulate additional time if necessary
                    if self.simulate_time:
                        time.sleep(self.simulate_time)

                    # Submit prefetch tasks to the executor
                    futures = [executor.submit(self.prefetch_lambda.prefetch_batch, item) for item in set_of_batches]

                    # Process completed futures
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            response = future.result()
                            # Optionally process the response
                            # print(f'Invocation response: {response}')
                        except Exception as e:
                            print(f'Invocation error: {e}')
                            logger.error(f"Error in prefetching batch: {e}", exc_info=True)
                    
                    # Calculate and log the time taken for prefetching
                    time_since_queued = time.perf_counter() - queued_time
                    self.prefetch_execution_times.update(time_since_queued - delay_time)
                    logger.info(f"Prefetching took: {time_since_queued:.4f} seconds for {len(set_of_batches)} batches. Expected time: {self.prefetch_execution_times.avg + delay_time}")

                except queue.Empty:
                    # Handle the empty queue case
                    time.sleep(0.1)  # Sleep for a short while before checking the queue again
                except Exception as e:
                    # General exception handling for unexpected issues
                    print(f"Unexpected error in prefetching process: {e}")
                    logger.error(f"Unexpected error in prefetching process: {e}", exc_info=True)


    # def _start_prefetching_process(self):

    #     while not self.prefetch_stop_event.is_set():  # Check for stop signal
    #         try:
    #             if self.prefetch_lambda.client is None:
    #                 self.prefetch_lambda.create_client()

    #             # Get a batch from the queue with a timeout to handle stopping
    #             queued_time, set_of_batches = self.prefetch_queue.get(timeout=1)
    #              # Calculate the delay before invoking Lambdas

    #             delay_time = self.calculate_delay() * len(set_of_batches)
    #             time.sleep(delay_time)  # Delay based on the calculated time to manage costs

    #             if self.simulate_time:
    #                 time.sleep(self.simulate_time)

    #             with concurrent.futures.ThreadPoolExecutor() as executor:
    #                 futures = [executor.submit(self.prefetch_lambda.prefetch_batch, item) for item in set_of_batches]

    #                 for future in concurrent.futures.as_completed(futures):
    #                     try:
    #                         response = future.result()
    #                         # print(f'Invocation response: {response}')
    #                     except Exception as e:
    #                         print(f'Invocation error: {e}')
                
    #             time_since_queued = time.perf_counter() - queued_time
    #             self.prefetch_execution_times.update(time_since_queued - delay_time)
    #             logger.info(f"Prefetching took: {time_since_queued:.4f} seconds for {len(set_of_batches)} batches. Expected time: {self.prefetch_execution_times.avg + delay_time}")
    #         except queue.Empty:
    #             # Handle the empty queue case
    #             time.sleep(0.1)  # Sleep for a short while before checking the queue again
    
    def stop_prefetcher(self):
        if not self.prefetch_stop_event.is_set():
            self.prefetch_stop_event.set()
            logger.info(f"Prefetcher stopped, total requests: {self.prefetch_lambda.request_counter}, Total execution time: {self.prefetch_execution_times.sum:.2f} seconds, Total cost: ${self.compute_total_cost():.16f}")
    
    def compute_total_cost(self):
        current_total_cost = compute_lambda_cost(self.prefetch_lambda.request_counter, self.prefetch_execution_times.avg, self.prefetch_lambda.configured_memory)
        return current_total_cost

class DLTJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.partition_id_cycle: Optional[Iterator[int]] = None
        self.epochs_completed_count = -1
        self.started_partition_index = None
        self.active_batch_set_id = None

        # self.active_epoch = initial_epoch
        self.total_steps = -1
        self.future_batches: OrderedDict[str, Batch] = OrderedDict()
        self.cache_hit_window = deque(maxlen=50)  # Sliding window for recent hits
        self.training_step_times_on_hit = AverageMeter('Training Step Time on Hit')
        self.training_step_times_on_miss =  AverageMeter('Training Step Time on Miss')
        self.training_step_gpu_times =  AverageMeter('training_step_gpu_times')
        self.current_batch:Batch = None
        self.cycle_bacthes = []

    def __repr__(self):
        return (f"Job(job_id={self.job_id}, current_epoch={self.active_epoch}, "
                f"current_index={self.total_steps})")
    
    def total_training_time(self):
        return self.training_step_times_on_hit.sum + self.training_step_times_on_miss.sum
    
    def total_training_steps (self):
        return self.training_step_times_on_hit.count + self.training_step_times_on_miss.count
    
    def update_perf_metrics(self, training_step_time: float, is_cache_hit: bool, previous_step_gpu_time:float):
        if previous_step_gpu_time > 0:
            self.training_step_gpu_times.update(previous_step_gpu_time)

        if training_step_time > 0: 
            if is_cache_hit:
                self.training_step_times_on_hit.update(training_step_time)
            else:
                self.training_step_times_on_miss.update(training_step_time)
    
    def next_training_step_batch(
        self,
        dataset: Dataset,
        prefetch_service: PrefetchManager = None,
        keep_alive_service: CacheEvicitonService = None,
        prefetch_delay: float = 0,
    ) -> Batch:
        self.total_steps += 1

        # Prefetching Logic
        if self.total_steps >= 1 and prefetch_service and not self.cycle_bacthes:
            prefetch_list = []
            prefetch_cycle_duration = prefetch_service.prefetch_execution_times.avg + prefetch_delay

            training_step_time_on_hit = (
                self.training_step_gpu_times.avg
                if self.training_step_times_on_hit.count == 0
                else self.training_step_times_on_hit.avg
            )
            training_step_time_on_miss = (
                prefetch_cycle_duration
                if self.training_step_times_on_miss.count == 0
                else self.training_step_times_on_miss.avg
            )
            optimal_prefetch_concurrency = find_optimal_prefetch_conncurrency(
                prefetch_cycle_duration, training_step_time_on_hit
            )

            # Prefetch and Cache Management
            prefetch_counter, time_counter = 0, 0
            for batch in self.future_batches.values():
                if time_counter <= prefetch_cycle_duration:
                    self.cycle_bacthes.append(batch.batch_id)
                    time_counter += (
                        training_step_time_on_hit if batch.is_cached else training_step_time_on_miss
                    )
                
                elif prefetch_counter < optimal_prefetch_concurrency:
                    prefetch_counter += 1
                    if not batch.is_cached and not batch.caching_in_progress:
                        batch.set_caching_in_progress(True)
                        payload = {
                            'bucket_name': dataset.bucket_name,
                            'batch_id': batch.batch_id,
                            'batch_samples': dataset.get_samples(batch.indicies),
                            'cache_address': prefetch_service.cache_address,
                            'task': 'prefetch',
                        }
                        prefetch_list.append((batch, payload))
                else:
                    break

            if len(prefetch_list) > 0:
                prefetch_service.prefetch_queue.put((time.perf_counter(), prefetch_list))

        # Keep-Alive Logic
        if keep_alive_service:
            keep_alive_list = [
                batch for batch in self.future_batches.values()
                if batch.is_cached and batch.time_since_last_access() > 1000
            ]
            if keep_alive_list:
                keep_alive_service.keep_alive_queue.put((time.perf_counter(), keep_alive_list))

        # Find the next training batch to process
        next_training_batch = None
        for batch_id, batch in list(self.future_batches.items()):
            if batch.is_cached or not batch.caching_in_progress:
                next_training_batch = self.future_batches.pop(batch_id)
                break

        # If no suitable batch found, get the first available one
        if not next_training_batch:
            next_training_batch = self.future_batches.pop(next(iter(self.future_batches)))

        # Update cycle batches
        if next_training_batch.batch_id in self.cycle_bacthes:
            self.cycle_bacthes.remove(next_training_batch.batch_id)

        return next_training_batch










    # def next_training_step_batch(self,
    #                              dataset: Dataset, 
    #                              prefetch_service:PrefetchManager = None, 
    #                              keep_alive_service: CacheEvicitonService = None,
    #                              prefetch_delay:float=0) -> Batch:
        






    #     self.total_steps += 1
    #     next_training_batch = None

    
    #     # If prefetching is enabled, compute prefetching logic
    #     if self.total_steps >= 1 and prefetch_service and len(self.cycle_bacthes)==0:
    #         prefetch_cycle_duration = prefetch_service.prefetch_execution_times.avg
    #         prefetch_list = []
    #         prefetch_cycle_duration += prefetch_delay
    #         time_counter = 0
    #         prefetch_counter = 0

    #         # Calculate optimal concurrency for prefetching
    #         training_step_time_on_hit = (
    #             self.training_step_gpu_times.avg
    #             if self.training_step_times_on_hit.count == 0
    #             else self.training_step_times_on_hit.avg
    #         )
    #         training_step_time_on_miss = (
    #             prefetch_cycle_duration
    #             if self.training_step_times_on_miss.count == 0
    #             else self.training_step_times_on_miss.avg
    #         )
    #         optimal_prefetch_concurrency = find_optimal_prefetch_conncurrency(prefetch_cycle_duration, training_step_time_on_hit)

    #         # Plan prefetching
    #         for batch_id, batch in self.future_batches.items():
    #             if time_counter <= prefetch_cycle_duration:
    #                 self.cycle_bacthes.append(batch.batch_id)
    #                 time_counter += training_step_time_on_hit if batch.is_cached else training_step_time_on_miss
    #             elif prefetch_counter < optimal_prefetch_concurrency:
    #                 if not batch.is_cached and not batch.caching_in_progress:
    #                     batch.set_caching_in_progress(True)
    #                     payload = {
    #                         'bucket_name': dataset.bucket_name,
    #                         'batch_id': batch.batch_id,
    #                         'batch_samples': dataset.get_samples(batch.indicies),
    #                         'cache_address': prefetch_service.cache_address,
    #                         'task': 'prefetch'
    #                     }
    #                     prefetch_list.append((batch, payload))
    #                     prefetch_counter += 1
    #             else:
    #                 break
    #         prefetch_service.prefetch_queue.put((time.perf_counter(), prefetch_list))
        
    #      # Identify batches to keep alive
    #     if keep_alive_service:
    #         keep_alive_list: List[Batch] = [
    #             batch for batch_id, batch in self.future_batches.items()
    #             if batch.is_cached and batch.time_since_last_access() > 1000
    #         ]
    #         keep_alive_service.keep_alive_queue.put((time.perf_counter(), keep_alive_list))
        
    #         # Find the next training batch to process
    #     for batch_id, batch in self.future_batches.items():
    #         if batch.is_cached or not batch.caching_in_progress:
    #             self.future_batches.pop(batch_id)
    #             next_training_batch = batch
    #             break
    #     if next_training_batch is None:
    #         next_training_batch = self.future_batches.pop(next(iter(self.future_batches)))

    #     if next_training_batch.batch_id in self.cycle_bacthes:
    #         self.cycle_bacthes.remove(next_training_batch.batch_id)

    #     return next_training_batch
               

            
            
    # def next_training_step_batch(self, cycle_duration:float, dataset: Dataset, cache_address: str, prefetch_delay:float=0) -> Optional[Batch]:
    #     self.total_steps += 1

    #     if self.total_steps == 0:
    #         return self.future_batches.popitem(last=False)[1], None, None #return the first batch in the future batches
        
    #     if self.batches_in_cycle: 
    #         return self.batches_in_cycle.pop(0), None, None
    #     else: 
    #         #cycele is empty so start a new cycle
    #         time_counter = 0
    #         prefetch_counter = 0
    #         cycle_duration = cycle_duration + prefetch_delay
    #         if self.training_step_times_on_miss.count == 0:
    #             training_step_time_on_miss = cycle_duration
    #         else:
    #             training_step_time_on_miss = self.training_step_times_on_miss.avg

    #         if self.training_step_times_on_hit.count == 0:
    #             training_step_time_on_hit = self.training_step_gpu_times.avg
    #         else:
    #             training_step_time_on_hit = self.training_step_times_on_hit.avg

    #         #the optimal prefetch size is the number of batches that can be processed in the prefetch cycle duration under 100% cache hit rate
    #         optimal_prefetch_conncurrency = find_optimal_prefetch_conncurrency(cycle_duration, training_step_time_on_hit)
            
    #         prefetch_list:List[Batch] = []
    #         keep_alive_list:List[Batch] = []

    #         for batch_id, batch in list(self.future_batches.items()):  
    #             #queue up the batches that will be processed in the cycle (prefetch) duration  
    #             if  time_counter <= cycle_duration:
    #                 if batch.is_cached:
    #                     time_counter += training_step_time_on_hit
    #                     self.batches_in_cycle.append(batch)
    #                 #if the batch is in progress, wait for a short time to see if it will be cached, if not, it will be included in the next cycle
    #                 # elif batch.caching_in_progress:
    #                 #     time.sleep(0.1)
    #                 #     if batch.is_cached:
    #                 #         time_counter += training_step_time_on_hit
    #                 #         self.batches_in_cycle.append(batch)
    #                 #     else:
    #                 #         time_counter += training_step_time_on_miss
    #                 #         self.batches_in_cycle.append(batch)
    #                 else:
    #                     time_counter += training_step_time_on_miss
    #                     self.batches_in_cycle.append(batch)
                    
    #                 self.future_batches.pop(batch_id) #remove the batch from the future batches since it has been queued up for processing
    #                 continue
                
    #             elif prefetch_counter < optimal_prefetch_conncurrency:
    #                 prefetch_counter += 1
    #                 if not batch.is_cached and not batch.caching_in_progress:
    #                     batch.set_caching_in_progress(in_progress=True)
    #                     payload = {
    #                         'bucket_name': dataset.bucket_name,
    #                         'batch_id': batch.batch_id,
    #                         'batch_samples': dataset.get_samples(batch.indicies),
    #                         'cache_address': cache_address,
    #                         'task': 'prefetch'
    #                         }
    #                     prefetch_list.append((batch,payload))
    #                 continue
    #             else:
    #                 if batch.last_accessed_time > 1000: #check if the batch has been accessed in the last 1000 seconds
    #                     keep_alive_list.append(batch)
            
    #         #start prefetching the batches

    #         return self.batches_in_cycle.pop(0), prefetch_list, keep_alive_list


class CentralBatchManager:

    def __init__(self, dataset: Dataset, 
                 look_ahead: int = 50, 
                 cache_eviction_service:CacheEvicitonService = None,
                 prefetch_service:PrefetchManager = None):
        
        self.dataset = dataset
        self.look_ahead = look_ahead
        self.prefetch_service:PrefetchManager = prefetch_service
        self.cache_eviction_service:CacheEvicitonService = cache_eviction_service
        self.lock = threading.Lock()
        self.jobs: Dict[str, DLTJob] = {}
        self.active_epoch_idx = 1
        self.active_partition_id = None

        # Create a cycle iterator for partitions
        self.epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = OrderedDict()  #first key is epoch id, second key is partition id, value is the batches
        self.batch_sampler = BatchSampler(partitions=dataset.partitions.values(), batch_size=self.dataset.batch_size, shuffle=False, drop_last=False)
        for _ in range(min(look_ahead, self.dataset.partitions[1].num_batches)):
            self._generate_new_batch()
        pass
    
    def _generate_new_batch(self):
        next_batch:Batch = next(self.batch_sampler)
        if self.active_partition_id != next_batch.partition_id:
            self.active_partition_id = next_batch.partition_id

        if self.active_epoch_idx != next_batch.epoch_idx:
            self.active_epoch_idx = next_batch.epoch_idx

        partition_batches = self.epoch_partition_batches.setdefault(next_batch.epoch_idx, {})
        partition_batch_set = partition_batches.setdefault(next_batch.partition_id, BatchSet(f'{next_batch.epoch_idx}_{next_batch.partition_id}'))
        partition_batch_set.batches[next_batch.batch_id] = next_batch

        for job in self.jobs.values():
            if job.active_batch_set_id == partition_batch_set.id:
                job.future_batches[next_batch.batch_id] = next_batch
        
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

        #now find the last batch set for this partition, and make sure it hasn't been processed by the job before
        for epoch_id in reversed(self.epoch_partition_batches.keys()):
            if next_partition_id in self.epoch_partition_batches[epoch_id]:
                batch_set = self.epoch_partition_batches[epoch_id][next_partition_id]
                job.future_batches.update(batch_set.batches)
                job.active_batch_set_id = batch_set.id
                break

    def get_next_batch(self, job_id: str, previous_step_training_time:float, previous_step_is_cache_hit:bool, previous_step_gpu_time:float, cached_batch:bool) -> Optional[Batch]:
        with self.lock:

            if self.prefetch_service is not None and self.prefetch_service.prefetch_stop_event.is_set():  
                self.prefetch_service.start_prefetcher() #prefetcher is stopped, start it
                
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if cached_batch or previous_step_is_cache_hit:
                    job.current_batch.set_last_accessed_time()
                    job.current_batch.set_cache_status(True)
                elif not previous_step_is_cache_hit and job.current_batch.is_cached:
                    job.current_batch.set_cache_status(False)
                job.update_perf_metrics(previous_step_training_time, previous_step_is_cache_hit, previous_step_gpu_time) #update the performance metrics for the job   
            else:
                job = DLTJob(job_id) #new job
                self.jobs[job_id] = job
                logger.info(f"Job '{job_id}' registered.")

            if not job.future_batches or len(job.future_batches) == 0: #reached end of partition so start preparing for next one
                self.allocate_batches_to_job(job)
            
            next_batch = job.next_training_step_batch(dataset=self.dataset, 
                                                      prefetch_service=self.prefetch_service,
                                                      keep_alive_service=self.cache_eviction_service)
            if next_batch.is_cached:
                next_batch.set_last_accessed_time()
            else:
                next_batch.set_caching_in_progress(True)
            
            if not next_batch.has_been_accessed_before:
                next_batch.set_has_been_accessed_before(True)
                self._generate_new_batch()

            job.current_batch = next_batch
            return next_batch
        
    def handle_job_ended(self, job_id: str, previous_step_training_time: float, previous_step_is_cache_hit: bool):
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.update_perf_metrics(previous_step_training_time, previous_step_is_cache_hit, previous_step_training_time)
                # logger.info(f"Job '{job_id}' has ended. Total training time: {job.total_training_time()}s, Total training steps: {job.total_training_steps()},total cache hits: {job.training_step_times_on_hit.count},total cache misses: {job.training_step_times_on_miss.count},cache hit rate: {job.training_step_times_on_hit.count / job.total_training_steps()}" )
                self.jobs.pop(job_id)
            if len(self.jobs) == 0:
                logger.info("All jobs have ended. Stopping prefetcher.")
                if self.prefetch_service:
                    self.prefetch_service.stop_prefetcher()
    



if __name__ == "__main__":
    TIME_ON_CACHE_HIT = 0.025
    TIME_ON_CACHE_MISS = 5
    PREFETCH_TIME = 4.5
    prefetcher = PrefetchManager(
        prefetch_lambda_name='CreateVisionTrainingBatch',
        cache_address= '10.0.28.76:6378',
        cost_threshold_per_hour= None, 
        simulate_time=PREFETCH_TIME)

    partitions_per_dataset = 1
    dataset = Dataset(data_dir='s3://sdl-cifar10/test/', batch_size=128, drop_last=False, num_partitions=partitions_per_dataset)
    batch_manager = CentralBatchManager(dataset=dataset, look_ahead=50, prefetch_service=prefetcher, cache_eviction_service=None)

    job_id = '1'
    cache_hits = 0
    cache_misses = 0
    previous_step_training_time = 0
    previous_step_is_cache_hit = False

    end = time.perf_counter()

    for i in range(500):
        batch:Batch = batch_manager.get_next_batch(job_id, previous_step_training_time, previous_step_is_cache_hit, TIME_ON_CACHE_HIT, False)
        if batch.is_cached:
            previous_step_is_cache_hit = True
            cache_hits += 1
            time.sleep(TIME_ON_CACHE_HIT)
            previous_step_training_time =TIME_ON_CACHE_HIT
        else:
            previous_step_is_cache_hit = False
            cache_misses += 1
            time.sleep(TIME_ON_CACHE_MISS)
            previous_step_training_time =TIME_ON_CACHE_MISS
        hit_rate = cache_hits / (i + 1) if (i + 1) > 0 else 0
        print(f'Batch {i+1}: {batch.batch_id}, Cache Hits: {cache_hits}, Cache Misses:{cache_misses}, Hit Rate: {hit_rate:.2f}')
        
    batch_manager.handle_job_ended(job_id, previous_step_training_time, previous_step_is_cache_hit)
    # batch_manager.prefetch_service.stop_prefetcher()
    print(f"total duration: {time.perf_counter() - end}")

    time.sleep(1)

    # print(f"Job Prefetch Size: {batch_manager.jobs[job_id].prefetch_size}")