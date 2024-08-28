import threading
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from sampling import BatchSampler, EndOfEpochException
from dataset import Dataset
from batch import Batch
import time
from logger_config import logger
from utils import AverageMeter, find_optimal_prefetch_conncurrency

TIME_ON_CACHE_HIT = 1.25
TIME_ON_CACHE_MISS = 5.25
PREFETCH_EXECUTION_TIME = 5

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

        #assign some default values for now
        self.training_step_times_on_hit.update(TIME_ON_CACHE_HIT)
        self.training_step_times_on_miss.update(TIME_ON_CACHE_MISS)


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
        self.sampler = BatchSampler(size=len(self.dataset), epoch_id=self.current_epoch,  batch_size=batch_size, shuffle=False, drop_last=False)
        
        self.epoch_batches[self.current_epoch] = OrderedDict()
        self.batches_accessed = set()
        self._initialize_batches()

        self.preftech_stop_event = threading.Event()  # Event to signal preftch stopping
        self.prefetch_concurrency = 1000
        self.prefetch_time = PREFETCH_EXECUTION_TIME  # execution time for a single invocation of prefetch aws lambda function
        self.first_job = True

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
        to_prefetch = []
        for job in self.jobs.values():
            if job.training_step_times_on_hit.avg >= job.training_step_times_on_miss.avg:
                # No need to prefetch if cache hit time is less than cache miss time
                continue

            optimal_prefetch_size = find_optimal_prefetch_conncurrency(self.prefetch_time, job.training_step_times_on_hit.avg)

            total_time = 0
            batches_job_will_access_in_next_cycle = []
            job_prefetch_list = []
            # Compute how many batches the job will access in the next self.prefetch time seconds
            for future_batch_id, future_batch in job.future_batches.items():
                if len(to_prefetch) >= self.prefetch_concurrency:
                    break
                if total_time >= self.prefetch_time:
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
                    to_prefetch.append(batch)

        return to_prefetch

    def _execute_prefetch(self, to_prefetch: List[Batch]):
            # Simulate prefetch time
            time.sleep(self.prefetch_time)  # Simulate the time it takes to prefetch the batches
            for batch in to_prefetch:
                with batch.lock:
                    batch.is_cached = True
                    batch.caching_in_progress = False

    def _start_proactive_prefetching(self):
        while not self.preftech_stop_event.is_set():  # Check for stop signal
            to_prefetch = self._select_batches_to_prefetch()

            if len(to_prefetch) > 0:
                self._execute_prefetch(to_prefetch)
            else:
                time.sleep(0.1)
        
        print("Prefetching stopped")

    def stop_prefetching(self):
        # Signal the thread to stop
        self.preftech_stop_event.set()


    def start_prefetching_thread(self):
        prefetch_thread = threading.Thread(target=self._start_proactive_prefetching)
        prefetch_thread.daemon = True  # This allows the thread to exit when the main program exits
        prefetch_thread.start()


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

            # job.update_perf_metrics(previous_step_training_time, previous_step_is_cache_hit)

            # if not job.future_batches or len(job.future_batches) ==0: #reaching end of epoch so start preparing for next epoch
            if not job.future_batches or len(job.future_batches) < self.look_ahead: #reaching end of epoch so start preparing for next epoch
                job.active_epoch = self.current_epoch
                job.future_batches.update(self.epoch_batches[self.current_epoch])
                job.epochs_completed_count += 1
            
            # if self.prefetch_event.is_set(): # Check if prefetching already is in progress or not
            #
            if self.first_job:
                self.first_job = False
                self.start_prefetching_thread()
                time.sleep(1)

            # if self.prefetch_event.is_set(): # Check if prefetching already is in progress or not
            #     self._start_proactive_prefetching()

            for batch_id, batch in list(job.future_batches.items()):     
                with batch.lock:
                   # batch is cached or caching is not in progress so process it
                    if batch.is_cached or not batch.caching_in_progress:
                        # Batch is cached, so it's processed normally
                        job.future_batches.pop(batch_id)
                        # Now find the next 'prefetch_size' batches that are not cached or not in progress

                        if not batch.has_been_accessed_before:
                            batch.has_been_accessed_before = True
                            self._generate_new_batch()

                        return batch
                    else:
                        continue
                    
        #should never reach here but just in case prefetch dies or something
        batch = job.future_batches.pop(batch_id)
        if not batch.has_been_accessed_before:
            batch.has_been_accessed_before = True
            self._generate_new_batch()
        return batch
    

if __name__ == "__main__":
    dataset = Dataset('s3://sdl-cifar10/train/', 128, False, 1)
    batch_manager = CentralBatchManager(dataset)
    job_id = '1'
    cache_hits = 0
    cache_misses = 0

    previous_step_training_time = 0
    previous_step_is_cache_hit = False
    end = time.perf_counter()
    for i in range(5):
        batch = batch_manager.get_next_batch(job_id, previous_step_training_time, previous_step_is_cache_hit)
        if batch.is_cached:
            previous_step_is_cache_hit = True
            cache_hits += 1
            time.sleep(TIME_ON_CACHE_HIT)
            previous_step_training_time =TIME_ON_CACHE_HIT
        else:
            previous_step_is_cache_hit = True
            cache_misses += 1
            time.sleep(TIME_ON_CACHE_MISS)
            previous_step_training_time =TIME_ON_CACHE_MISS

            # time.sleep(3)
        hit_rate = cache_hits / (i + 1) if (i + 1) > 0 else 0
        print(f'Batch {i+1}: {batch.batch_id}, Cache Hits: {cache_hits}, Cache Misses:{cache_misses}, Hit Rate: {hit_rate:.2f}')
    batch_manager.stop_prefetching()
    time.sleep(10)
    print(f"total duration: {time.perf_counter() - end}")

    # print(f"Job Prefetch Size: {batch_manager.jobs[job_id].prefetch_size}")