import threading
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from sampling import BatchSampler, EndOfEpochException
from dataset import Dataset
from batch import Batch
import time


class Job:
    def __init__(self, job_id: str, initial_epoch: int):
        self.job_id = job_id
        self.epochs_completed_count = -1
        self.active_epoch = initial_epoch
        self.current_index = -1
        self.future_batches: OrderedDict[str, Batch] = OrderedDict()
        
        self.prefetch_size = 1
        self.prefetch_size_cap = 100  # Maximum prefetch size
        self.prefetch_event = threading.Event()
        self.prefetch_event.set()  # Initially set to indicate no prefetch in progress
        self.target_hit_rate = 1.0 # Desired cache hit rate in percentage
        self.recent_requests = deque(maxlen=50)  # Rolling window to track last x requests
        self.initial_prefetch_size = self.prefetch_size
        self.prefetch_size_adjustment_interval = 3  # Initial interval in seconds to adjust concurrency
        self.max_interval = 300  # Maximum interval in seconds
        self.min_interval = 10  # Minimum interval in seconds
        self.stability_threshold = 3  # Number of stable intervals needed to increase the adjustment interval
        self.stability_counter = 0  # Tracks consecutive stable intervals

    def __repr__(self):
        return (f"Job(job_id={self.job_id}, current_epoch={self.active_epoch}, "
                f"current_index={self.current_index})")

class CentralBatchManager:
    def __init__(self, dataset: Dataset, look_ahead: int = 50, batch_size: int = 128, seed: int = 42):
        self.dataset = dataset
        self.look_ahead = look_ahead
        self.batch_size = batch_size
        self.seed = seed
        self.lock = threading.Lock()
        self.jobs: Dict[str, Job] = {}
        self.epoch_batches: Dict[int, OrderedDict[str, Batch]] = {}
        self.current_epoch = 1
        self.sampler = BatchSampler(size=len(self.dataset), epoch_id=self.current_epoch, 
                                    batch_size=batch_size, shuffle=True, drop_last=True)
        self.epoch_batches[self.current_epoch] = OrderedDict()
        self.batches_accessed = set()
        self._initialize_batches()
        self.request_counter = 0

    def _initialize_batches(self):
        while len(self.epoch_batches[self.current_epoch]) < self.look_ahead:
            try:
                batch = next(self.sampler)
                self.epoch_batches[self.current_epoch][batch.batch_id] = batch
            except EndOfEpochException:
                break
    
    def _get_recent_hit_rate(self, job: Job) -> float:
        return sum(job.recent_requests) / len(job.recent_requests) if job.recent_requests else 0

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
    
    def _adjust_prefetch_size(self, job: Job):
        
        if self.request_counter % job.prefetch_size_adjustment_interval == 0:
            # Adjust the prefetch size cap based on the hit rate in the sliding window
            current_hit_rate = self._get_recent_hit_rate(job)
            print(f'window hite rate: {current_hit_rate}')
            if current_hit_rate < job.target_hit_rate:
                if current_hit_rate < 0.5:
                    job.prefetch_size = max(job.prefetch_size+1,min(job.prefetch_size * 2, job.prefetch_size_cap)) # Example max cap
                    job.stability_counter = 0  # Reset stability counter if adjustment is needed
                else:
                    job.prefetch_size = min(job.prefetch_size + 1, job.prefetch_size_cap) # Example max cap
                    job.stability_counter = 0  # Reset stability counter if adjustment is needed
              
            elif current_hit_rate >= job.target_hit_rate: #+ job.hysteresis_margin
                job.prefetch_size = max(0, job.prefetch_size - 1)  # Ensure cap is at least 1
                job.stability_counter = 0  # Reset stability counter if adjustment is needed
            else:
                # Hit rate is within the target range
                job.stability_counter += 1
            
            # Gradually increase the adjustment interval if the hit rate is stable
            if job.stability_counter >= job.stability_threshold:
                job.prefetch_size_adjustment_interval = min(job.max_interval, job.prefetch_size_adjustment_interval * 2)
                job.stability_counter = 0  # Reset stability counter after increasing interval

       
           
    def get_next_batch(self, job_id: str) -> Optional[Batch]:
        with self.lock:
            self.request_counter += 1

            job = self.jobs.setdefault(job_id, Job(job_id, self.current_epoch))

            if not job.future_batches:
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
                        
                        if job.prefetch_event.is_set(): # Check if prefetching is in progress
                            to_prefetch = []
                            for next_batch_id, next_batch in job.future_batches.items():
                                if len(to_prefetch) >= job.prefetch_size:
                                    break
                                if not next_batch.is_cached and not next_batch.caching_in_progress:
                                    next_batch.caching_in_progress = True
                                    to_prefetch.append(next_batch)       
                            self._prefetch_batches(to_prefetch, job)
                            
                        if not batch.has_been_accessed_before:
                            batch.has_been_accessed_before = True
                            self._generate_new_batch()

                        if batch.is_cached:
                            job.recent_requests.append(1)
                        else:
                            job.recent_requests.append(0)

                        self._adjust_prefetch_size(job)

                        return batch
                    else:
                        job.recent_requests.append(0)
                        # self._adjust_prefetch_size(job)

                        # if job.prefetch_size < job.prefetch_size_cap:
                        #     job.prefetch_size += 1
                        continue
                    
        #should never reach here but just in case prefetch dies or something
        batch = job.future_batches.pop(batch_id)
        if not batch.has_been_accessed_before:
            batch.has_been_accessed_before = True
            self._generate_new_batch()
        return batch
    
   
    def _prefetch_batches(self, to_prefetch: List[Batch], job: Job):
        def prefetch():
            # print(f"Prefetching batches: {[batch.batch_id for batch in to_prefetch]}")
            time.sleep(7)  # Simulate the time it takes to prefetch the batches
            for batch in to_prefetch:
                batch.is_cached = True
                batch.caching_in_progress = False
            # print(f"Prefetching complete for batches: {[batch.batch_id for batch in to_prefetch]}")
            job.prefetch_event.set()

         # Clear the event to indicate that prefetching is in progress    
        job.prefetch_event.clear()
        # Start the prefetching in a separate thread
        threading.Thread(target=prefetch).start()
    

if __name__ == "__main__":
    dataset = Dataset('s3://sdl-cifar10/train/', 16, False, 1)
    batch_manager = CentralBatchManager(dataset)
    job_id = '1'
    cache_hits = 0
    cache_misses = 0
    for i in range(2500):
        batch = batch_manager.get_next_batch(job_id)
        if batch.is_cached:
            cache_hits += 1
            time.sleep(0.2)
        else:
            cache_misses += 1
            time.sleep(3)
        hit_rate = cache_hits / (i + 1) if (i + 1) > 0 else 0
        print(f'Batch {i+1}: {batch.batch_id}, Cache Hits: {cache_hits}, Cache Misses:{cache_misses}, Hit Rate: {hit_rate:.2f}, Prefetch Size: {batch_manager.jobs[job_id].prefetch_size}')

    # print(f"Job Prefetch Size: {batch_manager.jobs[job_id].prefetch_size}")