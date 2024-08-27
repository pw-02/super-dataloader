import threading
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from sampling import BatchSampler, EndOfEpochException
from dataset import Dataset
from batch import Batch
import time
from logger_config import logger
from utils import AverageMeter

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

    def __repr__(self):
        return (f"Job(job_id={self.job_id}, current_epoch={self.active_epoch}, "
                f"current_index={self.current_index})")
    
    def total_training_time(self):
        return self.training_step_times_on_hit.sum + self.training_step_times_on_miss.sum
    
    def total_training_steps (self):
        return self.training_step_times_on_hit.count + self.training_step_times_on_miss.count
    
    def update_perf_metrics(self, training_step_time: float, is_cache_hit: bool):
        if training_step_time > 0: 
            if is_cache_hit:
                self.training_step_times_on_hit.update(training_step_time)
            else:
                self.training_step_times_on_miss.update(training_step_time)


class CentralBatchManager:
    def __init__(self, dataset: Dataset, look_ahead: int = 200, batch_size: int = 128, seed: int = 42):
        self.dataset = dataset
        self.look_ahead = look_ahead
        self.batch_size = batch_size
        self.seed = seed
        self.lock = threading.Lock()
        self.jobs: Dict[str, DLTJob] = {}
        self.epoch_batches: Dict[int, OrderedDict[str, Batch]] = {}
        self.current_epoch = 1
        self.sampler = BatchSampler(size=len(self.dataset), epoch_id=self.current_epoch, 
                                    batch_size=batch_size, shuffle=False, drop_last=False)
        
        self.epoch_batches[self.current_epoch] = OrderedDict()
        self.batches_accessed = set()
        self._initialize_batches()
        self.prefetch_event = threading.Event() # Event to indicate prefetching in progress
        self.prefetch_event.set()  # Initially set to indicate no prefetch in progress

        self.prefetch_concurrency = 15
        self.prefetch_time = None  # execution time for a single invocation of prefetch aws lambda function

    def _initialize_batches(self):
        while len(self.epoch_batches[self.current_epoch]) < self.look_ahead:
            try:
                batch = next(self.sampler)
                self.epoch_batches[self.current_epoch][batch.batch_id] = batch
            except EndOfEpochException:
                break
    def _adjust_prefetch_concurrency(self, job: DLTJob):
        # Adjust the prefetch size cap based on the hit rate in the sliding window
        average_hit_rate = sum(job.cache_hit_window) / len(job.cache_hit_window) if job.cache_hit_window else 0

        if average_hit_rate >= job.cache_hit_threshold:
            # Decrease cap if hit rate is high
            self.prefetch_concurrency = max(1, self.prefetch_concurrency - 1)
        else:
            self.prefetch_concurrency = min(self.prefetch_concurrency + 1, 100)  # Example max cap of 100

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
    
    def _adjust_prefetch_size(self, job: DLTJob):
        # Adjust the prefetch size cap based on the hit rate in the sliding window
        average_hit_rate = sum(job.cache_hit_window) / len(job.cache_hit_window) if job.cache_hit_window else 0

        if average_hit_rate >= job.cache_hit_threshold:
            # Decrease cap if hit rate is high
            self.prefetch_size = max(0, self.prefetch_size - 1)  # Ensure cap is at least 1
        else:
            self.prefetch_size = min(self.prefetch_size + 1, self.prefetch_size_cap)  # Example max cap
    
    def update_job_metrics(self, job_id: str, training_step_time: float, is_cache_hit: bool):
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if is_cache_hit:
                    job.training_step_times_on_hit.update(training_step_time)
                else:
                    job.training_step_times_on_miss.update(training_step_time)
    
    def handle_job_ended(self, job_id: str, previous_step_training_time: float, previous_step_is_cache_hit: bool):
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.update_perf_metrics(previous_step_training_time, previous_step_is_cache_hit)
                logger.info(f"Job '{job_id}' has ended. Total training time: {job.total_training_time()}s, Total training steps: {job.total_training_steps()},total cache hits: {job.training_step_times_on_hit.count},total cache misses: {job.training_step_times_on_miss.count},cache hit rate: {job.training_step_times_on_hit.count / job.total_training_steps()}" )
                
                self.jobs.pop(job_id)
    
    def get_next_batch(self, job_id: str, previous_step_training_time:float, previous_step_is_cache_hit:bool) -> Optional[Batch]:
        with self.lock:
            
            if job_id in self.jobs:
                job = self.jobs[job_id]
                # we have seen this job before so it might have some stats for us
            else:
                job = DLTJob(job_id, self.current_epoch)
                self.jobs[job_id] = job
                logger.info(f"Job '{job_id}' registered with initial epoch '{self.current_epoch}'.")
           
            job.update_perf_metrics(previous_step_training_time, previous_step_is_cache_hit)
            # job = self.jobs.setdefault(job_id, DLTJob(job_id, self.current_epoch))
            
            if not job.future_batches or len(job.future_batches) ==0: #reaching end of epoch so start preparing for next epoch
            # if not job.future_batches or len(job.future_batches) < self.look_ahead: #reaching end of epoch so start preparing for next epoch
                job.active_epoch = self.current_epoch
                job.future_batches.update(self.epoch_batches[self.current_epoch])
                job.epochs_completed_count += 1

            for batch_id, batch in list(job.future_batches.items()):

                with batch.lock:

                   # batch is cached or caching is not in progress so process it
                    if batch.is_cached or not batch.caching_in_progress:
                        # Batch is cached, so it's processed normally
                        job.future_batches.pop(batch_id)
                        # Now find the next 'prefetch_size' batches that are not cached or not in progress
                        
                        if self.prefetch_event.is_set(): # Check if prefetching already is in progress or not
                            to_prefetch = []
                            for future_batch_id, future_batch_id in job.future_batches.items():
                                if len(to_prefetch) >= self.prefetch_concurrency:
                                    break
                                if not future_batch_id.is_cached and not future_batch_id.caching_in_progress:
                                    future_batch_id.caching_in_progress = True
                                    to_prefetch.append(future_batch_id)
                            print(self.prefetch_concurrency, len(to_prefetch))  
                            if self.prefetch_concurrency != len(to_prefetch):
                                pass
                            self._prefetch_batches(to_prefetch)
                            
                        if not batch.has_been_accessed_before:
                            batch.has_been_accessed_before = True
                            self._generate_new_batch()

                        if batch.is_cached:
                            job.cache_hit_window.append(1)
                        else:
                            job.cache_hit_window.append(0)

                        # self._adjust_prefetch_concurrency(job)
                        return batch
                    else:
                        continue
                    
        #should never reach here but just in case prefetch dies or something
        batch = job.future_batches.pop(batch_id)
        if not batch.has_been_accessed_before:
            batch.has_been_accessed_before = True
            self._generate_new_batch()
        return batch
    
   
    def _prefetch_batches(self, to_prefetch: List[Batch]):
        def prefetch():
            # print(f"Prefetching batches: {[batch.batch_id for batch in to_prefetch]}")
            time.sleep(5)  # Simulate the time it takes to prefetch the batches
            for batch in to_prefetch:
                with batch.lock:
                    batch.is_cached = True
                    batch.caching_in_progress = False
            # print(f"Prefetching complete for batches: {[batch.batch_id for batch in to_prefetch]}")
            self.prefetch_event.set()

         # Clear the event to indicate that prefetching is in progress    
        self.prefetch_event.clear()
        # Start the prefetching in a separate thread
        threading.Thread(target=prefetch).start()
    

if __name__ == "__main__":
    dataset = Dataset('s3://sdl-cifar10/train/', 128, False, 1)
    batch_manager = CentralBatchManager(dataset)
    job_id = '1'
    cache_hits = 0
    cache_misses = 0
    end = time.perf_counter()
    for i in range(1000):
        batch = batch_manager.get_next_batch(job_id)
        if batch.is_cached:
            cache_hits += 1
            time.sleep(0.25)
        else:
            cache_misses += 1
            time.sleep(4.45)

            # time.sleep(3)
        hit_rate = cache_hits / (i + 1) if (i + 1) > 0 else 0
        print(f'Batch {i+1}: {batch.batch_id}, Cache Hits: {cache_hits}, Cache Misses:{cache_misses}, Hit Rate: {hit_rate:.2f}')
    print(f"total duration: {time.perf_counter() - end}")

    # print(f"Job Prefetch Size: {batch_manager.jobs[job_id].prefetch_size}")