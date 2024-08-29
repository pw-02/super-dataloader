import threading
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from data_objects.sampling import BatchSampler, EndOfEpochException
from data_objects.dataset import Dataset
from data_objects.batch import Batch
import time
from logger_config import logger
from utils.utils import AverageMeter, find_optimal_prefetch_conncurrency
from utils.aws_utils import get_average_lambda_duration
import concurrent.futures
import boto3
import json
from botocore.config import Config

# TIME_ON_CACHE_HIT = 1.25
# TIME_ON_CACHE_MISS = 5.25
# PREFETCH_EXECUTION_TIME = 5

class DLTJob:
    def __init__(self, job_id: str, initial_epoch: int):
        self.job_id = job_id
        self.epochs_completed_count = -1
        self.active_epoch = initial_epoch
        self.current_index = -1
        self.future_batches: OrderedDict[str, Batch] = OrderedDict()
        self.cache_hit_window = deque(maxlen=50)  # Sliding window for recent hits
        self.training_step_times_on_hit = AverageMeter('Training Step Time on Hit')
        self.training_step_times_on_miss =  AverageMeter('Training Step Time on Miss')
        self.training_step_gpu_times =  AverageMeter('training_step_gpu_times')

        #assign some default values for now
        # self.training_step_times_on_hit.update(5)
        # self.training_step_times_on_miss.update(0.5)


    def __repr__(self):
        return (f"Job(job_id={self.job_id}, current_epoch={self.active_epoch}, "
                f"current_index={self.current_index})")
    
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


class PrefetchLambda:
    def __init__(self, lambda_name: str):
        self.lambda_name = lambda_name
        self.avg_execution_time = get_average_lambda_duration(lambda_name)
        self.request_counter = 0
        self.lock = threading.Lock()

        print(f"Average execution time for lambda '{lambda_name}': {self.avg_execution_time}")
 
        self.lambda_client = boto3.client('lambda', config= Config(max_pool_connections=50))

    def update_avg_execution_time(self):
        self.avg_execution_time = get_average_lambda_duration(self.lambda_name)
    
    def prefetch_batch(self, payload_info: Tuple[Batch, Dict[str, str]]):
        
        batch, payload = payload_info
        payload['task'] ='prefetch'
        batch.caching_in_progress = True
        if  self.lambda_client is None:
            self.lambda_client = boto3.client('lambda') 

        response = self.lambda_client.invoke( FunctionName=self.lambda_name,
                                             InvocationType='RequestResponse',
                                             Payload=json.dumps(payload))
        response_data = json.loads(response['Payload'].read().decode('utf-8'))

        if response_data['success']:
            with self.lock:
                self.request_counter += 1
            batch.caching_in_progress = False
            batch.is_cached = True
            print(f"Batch '{batch.batch_id}' has been prefetched, request counter: {self.request_counter}")
            return response_data
        else:
            batch.caching_in_progress = False
            batch.is_cached = False
            logger.error(f"Error prefetching batch '{batch.batch_id}': {response_data['errorMessage']}")
            return response_data
    
        

class CentralBatchManager:
    def __init__(self, dataset: Dataset, look_ahead: int = 50, prefetch_concurrency: int = 10, 
                 prefetch_lambda_name: str = 'CreateVisionTrainingBatch',
                 cache_address: str = '10.0.19.221:6378 '):
        self.dataset = dataset
        self.look_ahead = look_ahead
        self.lock = threading.Lock()
        self.jobs: Dict[str, DLTJob] = {}
        self.epoch_batches: Dict[int, OrderedDict[str, Batch]] = {}
        self.current_epoch = 1
        self.sampler = BatchSampler(size=len(self.dataset), epoch_id=self.current_epoch,  batch_size=self.dataset.batch_size, shuffle=False, drop_last=False)
        self.epoch_batches[self.current_epoch] = OrderedDict()
        self.batches_accessed = set()
        self.prefetch_lambda = PrefetchLambda(prefetch_lambda_name)
        self.cache_address = cache_address
        self.prefetch_start_event = threading.Event()  # Event to signal preftch stopping
        self.prefetch_stop_event = threading.Event()  # Event to signal preftch stopping
        self.prefetch_concurrency = prefetch_concurrency
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


    def _select_batches_to_prefetch(self) -> List[Batch]:
        self.prefetch_lambda.update_avg_execution_time()
        to_prefetch = []
        for job in self.jobs.values():
            
            if job.training_step_times_on_hit.avg >= job.training_step_times_on_miss.avg:
                # No need to prefetch if cache hit time is less than cache miss time
                continue
            if job.training_step_times_on_miss.count == 0 and (job.training_step_times_on_hit.count == 0 or job.training_step_times_on_miss.avg == 0):
                # No need to prefetch if no cache misses and no cache hits
                continue
            if job.training_step_times_on_hit.count == 0:
                optimal_prefetch_size = find_optimal_prefetch_conncurrency(self.prefetch_lambda.avg_execution_time, job.training_step_gpu_times.avg)
                avg_execution_time = job.training_step_gpu_times.avg
            else:
                optimal_prefetch_size = find_optimal_prefetch_conncurrency(self.prefetch_lambda.avg_execution_time, job.training_step_times_on_hit.avg)
                avg_execution_time = job.training_step_times_on_hit.avg
            total_time = 0
            batches_job_will_access_in_next_cycle = []
            job_prefetch_list:List[Batch] = []
            # Compute how many batches the job will access in the next self.prefetch time seconds
            for future_batch_id, future_batch in job.future_batches.items():
                if len(to_prefetch) >= self.prefetch_concurrency:
                    break
                if total_time >= avg_execution_time:
                    break
                if future_batch.is_cached:
                    batches_job_will_access_in_next_cycle.append(future_batch_id)
                    total_time += job.training_step_times_on_hit.avg
                else:
                    batches_job_will_access_in_next_cycle.append(future_batch_id)
                    total_time += job.training_step_times_on_miss.avg

            # We cannot prefetch the batches_job_will_access_in_next_cycle since it takes >= than self.prefetch_time,
            # but we can prefetch for the next cycle
            for future_batch_id, future_batch in job.future_batches.items():
                if len(job_prefetch_list) >= optimal_prefetch_size:
                    break
                if future_batch_id not in batches_job_will_access_in_next_cycle:
                    job_prefetch_list.append(future_batch)

            # Mark batches for prefetching
            for batch in job_prefetch_list:
                if not batch.is_cached and not batch.caching_in_progress:
                    batch.caching_in_progress = True
                    payload = {
                            'bucket_name': self.dataset.bucket_name,
                            'batch_id': batch.batch_id,
                            'batch_samples': self.dataset.get_samples(batch.indicies),
                            'cache_address': self.cache_address
                            }
        
                    to_prefetch.append((batch, payload))
        if len(to_prefetch) >= 1:
            logger.info(f" prefetch size: {optimal_prefetch_size}, lambda time: {self.prefetch_lambda.avg_execution_time}, hit: {job.training_step_times_on_hit.avg}, miss: {job.training_step_times_on_miss.avg}, prefetch_count: {self.prefetch_lambda.request_counter}")
     
        return to_prefetch, self.prefetch_lambda.avg_execution_time

    def _start_proactive_prefetching(self):
        while not self.prefetch_stop_event.is_set():  # Check for stop signal
            to_prefetch, expected_time = self._select_batches_to_prefetch()
            if len(to_prefetch) <= 0:
                time.sleep(0.1)
            else:
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.prefetch_lambda.prefetch_batch, payload_info) for payload_info in to_prefetch]

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            response = future.result()
                            # print(f'Invocation response: {response}')
                        except Exception as e:
                            print(f'Invocation error: {e}')   
                
                actual_time_taken = time.perf_counter() - start
                print(f"Prefetching took: {actual_time_taken} seconds for {len(to_prefetch)} batches. Expected time: {expected_time}")
                # remaining_time = expected_time - actual_time_taken
                # if remaining_time > 0:
                #     print(f"Sleeping for {remaining_time:.2f} seconds to match expected time.")
                #     time.sleep(remaining_time)
        
        print("Prefetching stopped")

    def stop_prefetching(self):
        # Signal the thread to stop
        self.prefetch_stop_event.set()


    def start_prefetching_thread(self):
        prefetch_thread = threading.Thread(target=self._start_proactive_prefetching)
        prefetch_thread.daemon = True  # This allows the thread to exit when the main program exits
        prefetch_thread.start()

    def handle_job_ended(self, job_id: str, previous_step_training_time: float, previous_step_is_cache_hit: bool):
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.update_perf_metrics(previous_step_training_time, previous_step_is_cache_hit, previous_step_training_time)
                logger.info(f"Job '{job_id}' has ended. Total training time: {job.total_training_time()}s, Total training steps: {job.total_training_steps()},total cache hits: {job.training_step_times_on_hit.count},total cache misses: {job.training_step_times_on_miss.count},cache hit rate: {job.training_step_times_on_hit.count / job.total_training_steps()}" )
                
                self.jobs.pop(job_id)


    def get_next_batch(self, job_id: str, previous_step_training_time:float, previous_step_is_cache_hit:bool, previous_step_gpu_time:float) -> Optional[Batch]:
        with self.lock:    
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.update_perf_metrics(previous_step_training_time, previous_step_is_cache_hit, previous_step_gpu_time) #update the performance metrics for the job
            else:
                job = DLTJob(job_id, self.current_epoch)
                self.jobs[job_id] = job
                logger.info(f"Job '{job_id}' registered with initial epoch '{self.current_epoch}'.")
            
            # if not job.future_batches or len(job.future_batches) ==0: #reaching end of epoch so start preparing for next epoch
            if not job.future_batches or len(job.future_batches) < self.look_ahead: #reaching end of epoch so start preparing for next epoch
                job.active_epoch = self.current_epoch
                job.future_batches.update(self.epoch_batches[self.current_epoch])
                job.epochs_completed_count += 1
            
            if not self.prefetch_start_event.is_set(): # Check if prefetching already is in progress or not
                 self.prefetch_start_event.set()
                 self.start_prefetching_thread()
                 time.sleep(0.5)

            for batch_id, batch in list(job.future_batches.items()):     
                with batch.lock:
                   # batch is cached or caching is not in progress so process it
                    if batch.is_cached or not batch.caching_in_progress:
                        # Batch is cached, so it's processed normally
                        job.future_batches.pop(batch_id)

                        if not batch.has_been_accessed_before:
                            batch.has_been_accessed_before = True
                            self._generate_new_batch()

                        return batch
                    else:
                        continue
                    
        #should never reach here but just in case..
        batch = job.future_batches.pop(batch_id)
        if not batch.has_been_accessed_before:
            batch.has_been_accessed_before = True
            self._generate_new_batch()
        return batch
    

# if __name__ == "__main__":
#     dataset = Dataset('s3://sdl-cifar10/train/', 128, False, 1)
#     batch_manager = CentralBatchManager(dataset)
#     job_id = '1'
#     cache_hits = 0
#     cache_misses = 0

#     previous_step_training_time = 0
#     previous_step_is_cache_hit = False
#     end = time.perf_counter()
#     for i in range(5):
#         batch = batch_manager.get_next_batch(job_id, previous_step_training_time, previous_step_is_cache_hit)
#         if batch.is_cached:
#             previous_step_is_cache_hit = True
#             cache_hits += 1
#             time.sleep(TIME_ON_CACHE_HIT)
#             previous_step_training_time =TIME_ON_CACHE_HIT
#         else:
#             previous_step_is_cache_hit = True
#             cache_misses += 1
#             time.sleep(TIME_ON_CACHE_MISS)
#             previous_step_training_time =TIME_ON_CACHE_MISS

#             # time.sleep(3)
#         hit_rate = cache_hits / (i + 1) if (i + 1) > 0 else 0
#         print(f'Batch {i+1}: {batch.batch_id}, Cache Hits: {cache_hits}, Cache Misses:{cache_misses}, Hit Rate: {hit_rate:.2f}')
#     batch_manager.stop_prefetching()
#     time.sleep(10)
#     print(f"total duration: {time.perf_counter() - end}")

#     # print(f"Job Prefetch Size: {batch_manager.jobs[job_id].prefetch_size}")