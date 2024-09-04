import threading
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from data_objects.sampling import BatchSampler, EndOfEpochException
from data_objects.dataset import Dataset
from data_objects.batch import Batch
import time
from logger_config import logger
from utils.utils import AverageMeter, find_optimal_prefetch_conncurrency
from utils.aws_utils import get_average_lambda_duration, get_total_lambda_invocations, get_memory_allocation_of_lambda, compute_lambda_cost
import concurrent.futures
import boto3
import json
from botocore.config import Config
import queue
from datetime import datetime, timedelta, timezone
import redis


class PrefetchLambda:
    def __init__(self, prefetch_lambda_name: str, cache_address: str, simulated: bool = False):
        self.name = prefetch_lambda_name
        self.cache_address = cache_address
        self.client = boto3.client('lambda', config= Config(max_pool_connections=50))
        self.redis_client = None
        self.configured_memory = get_memory_allocation_of_lambda(self.name)
        logger.info(f"Configured memory for lambda '{self.name}': {self.configured_memory}")
        self.request_counter = 0
        self.simulated = simulated
        self.lock = threading.Lock()
    
    def create_client(self):
        self.client = boto3.client('lambda', config= Config(max_pool_connections=50))

    def prefetch_batch(self, item: Tuple[Batch, Dict[str, str]]):
        if self.simulated:
            batch, payload = item
            time.sleep(4)
            with self.lock:
                self.request_counter += 1
            batch.set_cache_status(is_cached=True)
            batch.set_caching_in_progress(in_progress=False) 
            return {'success': True, 'errorMessage': None}


        batch, payload = item
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

    def get_average_request_duration(self,start_time=None):
        if start_time is not None:
            return get_average_lambda_duration(self.name, start_time=start_time, end_time = datetime.now(timezone.utc)) #average duration since prefetching started
        else:
            return get_average_lambda_duration(self.name, 
                                               start_time=datetime.now(timezone.utc)  - timedelta(hours=4), #average duration for the last 4 hours
                                               end_time = datetime.now(timezone.utc))
    
    def compute_request_rate(self, start_time:float):
        with self.lock:
            elapsed_time = time.perf_counter() - start_time  # Calculate elapsed time
            if elapsed_time > 0:
                request_rate = self.request_counter / elapsed_time  # Compute request rate
                return request_rate
            return 0.0
    


class PrefetchService:
    def __init__(self, prefetch_lambda_name: str, cache_address: str, cost_threshold_per_hour: float = 10, simulate_prefetch: bool = False):
        self.prefetch_execution_times = AverageMeter('Prefetch Execution Time')
        self.prefetch_stop_event = threading.Event()  # Event to signal preftch stopping
        self.keep_alive_stop_event = threading.Event()  # Event to signal preftch stopping
        self.prefetch_queue = queue.Queue()  # Queue to store lists batches to be prefetched
        self.keep_alive_queue: queue.Queue[Tuple[float, List[Tuple['Batch', Dict[str, str]]]]] = queue.Queue()  # Queue to store lists batches to be prefetched
        self.cache_address = cache_address
        self.prefetch_lambda = PrefetchLambda(prefetch_lambda_name, cache_address, simulate_prefetch)
        self.max_prefetch_concurrency = 10000

        avg_lambda_duration_over_the_past_day = self.prefetch_lambda.get_average_request_duration(start_time=datetime.now(timezone.utc) - timedelta(days=1))
        if avg_lambda_duration_over_the_past_day is not None and avg_lambda_duration_over_the_past_day > 0:
            logger.info(f"Average execution time for lambda '{prefetch_lambda_name}': {avg_lambda_duration_over_the_past_day}")
            self.prefetch_execution_times.update(avg_lambda_duration_over_the_past_day)
        else:
            self.prefetch_execution_times.update(4)  # Default value for the first time

        self.lock = threading.Lock()
        self.keep_alive_time = 1000
        self.prefetch_stop_event.set()
        self.cost_threshold_per_hour = cost_threshold_per_hour # 0.00000020272686997451454 * 2000
        self.start_time = None
        self.start_time_utc = datetime.now(timezone.utc)
        self.prefetch_delay = 0

    def start_prefetcher(self):
        self.prefetch_stop_event.clear()
        self.start_time = time.perf_counter()
        prefetch_thread = threading.Thread(target=self._start_prefetching_process)
        prefetch_thread.daemon = True
        prefetch_thread.start()
    
    def start_keep_aliver(self):
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
                    if batch.last_accessed_time > self.keep_alive_time :
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
        while not self.prefetch_stop_event.is_set():  # Check for stop signal
            try:
                if self.prefetch_lambda.client is None:
                    self.prefetch_lambda.create_client()

                # Get a batch from the queue with a timeout to handle stopping
                queued_time, set_of_batches = self.prefetch_queue.get(timeout=1)
                 # Calculate the delay before invoking Lambdas

                delay_time = self.calculate_delay() * len(set_of_batches)
                time.sleep(delay_time)  # Delay based on the calculated time to manage costs

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.prefetch_lambda.prefetch_batch, item) for item in set_of_batches]
                
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            response = future.result()
                            # print(f'Invocation response: {response}')
                        except Exception as e:
                            print(f'Invocation error: {e}')
                
                time_since_queued = time.perf_counter() - queued_time
                self.prefetch_execution_times.update(time_since_queued - delay_time)
                logger.info(f"Prefetching took: {time_since_queued} seconds for {len(set_of_batches)} batches. Expected time: {self.prefetch_execution_times.avg + delay_time}")
            
            except queue.Empty:
                # Handle the empty queue case
                time.sleep(0.1)  # Sleep for a short while before checking the queue again
    
    def stop_prefetcher(self):
        self.prefetch_stop_event.set()
        logger.info(f"Prefetcher stopped, total requests: {self.prefetch_lambda.request_counter}, Total execution time: {self.prefetch_execution_times.sum:.2f} seconds, Total cost: ${self.compute_total_cost():.16f}")
    
    def compute_total_cost(self):
        current_total_cost = compute_lambda_cost(self.prefetch_lambda.request_counter, self.prefetch_execution_times.avg, self.prefetch_lambda.configured_memory)
        return current_total_cost

class DLTJob:
    def __init__(self, job_id: str, initial_epoch: int):
        self.job_id = job_id
        self.epochs_completed_count = -1
        self.active_epoch = initial_epoch
        self.total_steps = -1
        self.future_batches: OrderedDict[str, Batch] = OrderedDict()
        self.cache_hit_window = deque(maxlen=50)  # Sliding window for recent hits
        self.training_step_times_on_hit = AverageMeter('Training Step Time on Hit')
        self.training_step_times_on_miss =  AverageMeter('Training Step Time on Miss')
        self.training_step_gpu_times =  AverageMeter('training_step_gpu_times')
        self.batches_in_cycle = []
        self.current_batch:Batch = None

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
            
    def next_training_step_batch(self, cycle_duration:float, dataset: Dataset, cache_address: str, prefetch_delay:float=0) -> Optional[Batch]:
        self.total_steps += 1

        if self.total_steps == 0:
            return self.future_batches.popitem(last=False)[1], None, None #return the first batch in the future batches

        if self.batches_in_cycle: 
            return self.batches_in_cycle.pop(0), None, None
        else: 
            #cycele is empty so start a new cycle
            time_counter = 0
            prefetch_counter = 0
            cycle_duration = cycle_duration + prefetch_delay
            if self.training_step_times_on_miss.count == 0:
                training_step_time_on_miss = cycle_duration
            else:
                training_step_time_on_miss = self.training_step_times_on_miss.avg

            if self.training_step_times_on_hit.count == 0:
                training_step_time_on_hit = self.training_step_gpu_times.avg
            else:
                training_step_time_on_hit = self.training_step_times_on_hit.avg

            #the optimal prefetch size is the number of batches that can be processed in the prefetch cycle duration under 100% cache hit rate
            optimal_prefetch_conncurrency = find_optimal_prefetch_conncurrency(cycle_duration, training_step_time_on_hit)
            
            prefetch_list:List[Batch] = []
            keep_alive_list:List[Batch] = []

            for batch_id, batch in list(self.future_batches.items()):  
                #queue up the batches that will be processed in the cycle (prefetch) duration  
                if  time_counter <= cycle_duration:
                    if batch.is_cached:
                        time_counter += training_step_time_on_hit
                        self.batches_in_cycle.append(batch)
                    #if the batch is in progress, wait for a short time to see if it will be cached, if not, it will be included in the next cycle
                    # elif batch.caching_in_progress:
                    #     time.sleep(0.1)
                    #     if batch.is_cached:
                    #         time_counter += training_step_time_on_hit
                    #         self.batches_in_cycle.append(batch)
                    #     else:
                    #         time_counter += training_step_time_on_miss
                    #         self.batches_in_cycle.append(batch)
                    else:
                        time_counter += training_step_time_on_miss
                        self.batches_in_cycle.append(batch)
                    
                    self.future_batches.pop(batch_id) #remove the batch from the future batches since it has been queued up for processing
                    continue
                
                elif prefetch_counter < optimal_prefetch_conncurrency:
                    prefetch_counter += 1
                    if not batch.is_cached and not batch.caching_in_progress:
                        batch.set_caching_in_progress(in_progress=True)
                        payload = {
                            'bucket_name': dataset.bucket_name,
                            'batch_id': batch.batch_id,
                            'batch_samples': dataset.get_samples(batch.indicies),
                            'cache_address': cache_address,
                            'task': 'prefetch'
                            }
                        prefetch_list.append((batch,payload))
                    continue
                else:
                    if batch.last_accessed_time > 1000: #check if the batch has been accessed in the last 1000 seconds
                        keep_alive_list.append(batch)
            
            #start prefetching the batches

            return self.batches_in_cycle.pop(0), prefetch_list, keep_alive_list


class CentralBatchManager:
    def __init__(self, dataset: Dataset, look_ahead: int = 50, prefetch_concurrency: int = 10, prefetch_service:PrefetchService = None):

        self.dataset = dataset
        self.look_ahead = look_ahead
        self.lock = threading.Lock()
        self.jobs: Dict[str, DLTJob] = {}
        self.epoch_batches: Dict[int, OrderedDict[str, Batch]] = {}
        self.current_epoch = 1
        self.sampler = BatchSampler(size=len(self.dataset), epoch_id=self.current_epoch,  batch_size=self.dataset.batch_size, shuffle=False, drop_last=False)
        self.epoch_batches[self.current_epoch] = OrderedDict()
        self.batches_accessed = set()    
        self.prefetch_service:PrefetchService = prefetch_service
        # self.prefetch_concurrency = prefetch_concurrency
        self._initialize_batches()
        
    def _initialize_batches(self):
        while len(self.epoch_batches[self.current_epoch]) < self.look_ahead:
            try:
                batch = next(self.sampler)
                self.epoch_batches[self.current_epoch][batch.batch_id] = batch
            except EndOfEpochException:
                break
    
    def _generate_new_batch(self):
        try:
            new_batch = next(self.sampler)
            self.epoch_batches[self.current_epoch][new_batch.batch_id] = new_batch
            for job in self.jobs.values():
                if job.active_epoch == self.current_epoch:
                    job.future_batches[new_batch.batch_id] = new_batch
        except EndOfEpochException:
            self.current_epoch += 1
            self.epoch_batches[self.current_epoch] = OrderedDict()
            self.sampler.reset(self.current_epoch)
            self._generate_new_batch()


    def handle_job_ended(self, job_id: str, previous_step_training_time: float, previous_step_is_cache_hit: bool):
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.update_perf_metrics(previous_step_training_time, previous_step_is_cache_hit, previous_step_training_time)
                # logger.info(f"Job '{job_id}' has ended. Total training time: {job.total_training_time()}s, Total training steps: {job.total_training_steps()},total cache hits: {job.training_step_times_on_hit.count},total cache misses: {job.training_step_times_on_miss.count},cache hit rate: {job.training_step_times_on_hit.count / job.total_training_steps()}" )
                self.jobs.pop(job_id)
            if len(self.jobs) == 0:
                logger.info("All jobs have ended. Stopping prefetcher.")
                self.prefetch_service.stop_prefetcher()

    def get_next_batch(self, job_id: str, previous_step_training_time:float, previous_step_is_cache_hit:bool, previous_step_gpu_time:float, cached_batch:bool) -> Optional[Batch]:
        with self.lock:

            if self.prefetch_service.prefetch_stop_event.is_set(): #prefetcher has been stopped, start it again
                self.prefetch_service.start_prefetcher()
       
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if cached_batch or previous_step_is_cache_hit:
                    job.current_batch.set_last_accessed_time()
                    job.current_batch.set_cache_status(True)
                elif not previous_step_is_cache_hit and job.current_batch.is_cached:
                    job.current_batch.set_cache_status(False)
                job.update_perf_metrics(previous_step_training_time, previous_step_is_cache_hit, previous_step_gpu_time) #update the performance metrics for the job
            else:
                #new job
                job = DLTJob(job_id, self.current_epoch)
                self.jobs[job_id] = job
                logger.info(f"Job '{job_id}' registered with initial epoch '{self.current_epoch}'.")
            
            # if not job.future_batches or len(job.future_batches) ==0: #reaching end of epoch so start preparing for next epoch
            if not job.future_batches or len(job.future_batches) < self.look_ahead: #reaching end of epoch so start preparing for next epoch
                job.active_epoch = self.current_epoch
                job.future_batches.update(self.epoch_batches[self.current_epoch])
                job.epochs_completed_count += 1
            
            next_batch, prefetch_list, keep_alive_list = job.next_training_step_batch(cycle_duration=self.prefetch_service.prefetch_execution_times.avg, 
                                                                                      dataset=self.dataset, 
                                                                                      cache_address=self.prefetch_service.cache_address,
                                                                                      prefetch_delay=self.prefetch_service.prefetch_delay)
            if prefetch_list:
                self.prefetch_service.prefetch_queue.put((time.perf_counter(), prefetch_list))

            if next_batch.is_cached:
                next_batch.set_last_accessed_time()
            
            if not next_batch.has_been_accessed_before:
                next_batch.set_has_been_accessed_before(True)
                self._generate_new_batch()

            job.current_batch = next_batch
                
            return next_batch



if __name__ == "__main__":
    TIME_ON_CACHE_HIT = 0.25
    TIME_ON_CACHE_MISS = 1.5

    prefetcher = PrefetchService('CreateVisionTrainingBatch', '10.0.28.76:6378', None)

    dataset = Dataset('s3://sdl-cifar10/test/', 128, False, 1)
    batch_manager = CentralBatchManager(dataset,50,10,prefetcher)

    job_id = '1'
    cache_hits = 0
    cache_misses = 0
    previous_step_training_time = 0
    previous_step_is_cache_hit = False

    end = time.perf_counter()

    for i in range(50):
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
        

    batch_manager.prefetch_service.stop_prefetcher()
    print(f"total duration: {time.perf_counter() - end}")

    # print(f"Job Prefetch Size: {batch_manager.jobs[job_id].prefetch_size}")