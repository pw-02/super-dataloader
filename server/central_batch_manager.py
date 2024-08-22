import threading
from collections import deque
from sampling import BatchSampler, EndOfEpochException
from typing import List, Tuple, Dict, Optional
from dataset import Dataset

class Job:
    def __init__(self, job_id: str, initial_epoch: int):
        self.job_id = job_id
        self.current_epoch = initial_epoch
        self.current_index = -1
        # self.completed_batches = set()  # Track completed batches

    def update_progress(self):
        """Update the job's progress by moving to the next batch index."""
        self.current_index += 1
        # self.completed_batches.add(batch.batch_id)

    def next_batch_index(self):
        """Return the index of the next batch."""
        return self.current_index + 1

    def __repr__(self):
        return f"Job(job_id={self.job_id}, current_epoch={self.current_epoch}, current_index={self.current_index})"


class CentralBatchManager:
    def __init__(self, dataset: Dataset, look_ahead: int = 2, batch_size: int = 128, seed: int = 42):
        self.dataset = dataset
        self.look_ahead = look_ahead
        self.batch_size = batch_size
        self.seed = seed
        self.lock = threading.Lock()
        self.jobs: Dict[str, Job] = {}  # Tracks jobs
        self.epoch_batches: Dict[int, List] = {}  # Batches for each epoch
        self.current_epoch = 1
        self.sampler = BatchSampler(size=len(self.dataset), epoch_id=self.current_epoch, batch_size=batch_size, shuffle=True, drop_last=True)
        self.epoch_batches[self.current_epoch] = []
        self.batches_accessed = set()  # Track accessed batch IDs
        self._initialize_batches()
        pass

    def _initialize_batches(self):

        while len(self.epoch_batches[self.current_epoch]) < self.look_ahead:
            try:
                batch = next(self.sampler)
                self.epoch_batches[self.current_epoch].append(batch)
            except EndOfEpochException:
                break  # If there are no more batches, stop initialization

    def _generate_new_batch(self) -> Optional[List[Tuple[str, int]]]:
        """Generate a new batch of samples using the BatchSampler."""
        try:
            new_batch = next(self.sampler)
            self.epoch_batches[self.current_epoch].append(new_batch)
            self.batches_accessed.add(new_batch.batch_id)
            # return next(self.sampler)
        except EndOfEpochException:
            # self.sampler.reset()
            self.current_epoch += 1
            self.epoch_batches[self.current_epoch] = []
            self.sampler.reset(self.current_epoch)
            return self._generate_new_batch()

    def get_next_batch(self, job_id: str) -> Optional[List[Tuple[str, int]]]:
        """Return the next available batch for a job and maintain the look-ahead buffer."""
        with self.lock:
            if job_id not in self.jobs:
                self.jobs[job_id] = Job(job_id, self.current_epoch)

            job = self.jobs[job_id]
            
            if job.next_batch_index() >= len(self.epoch_batches[job.current_epoch]):
                job.current_epoch += 1
                job.current_index = -1
                
            # Get the next batch and update job's progress
            batch = self.epoch_batches[job.current_epoch][job.next_batch_index()]
            job.update_progress()
       
            # Generate a new global batch for future if it's the first time the batch is processed
            if batch not in self.batches_accessed:
                self._generate_new_batch()

            return batch


if __name__ == "__main__":
    dataset = Dataset('s3://sdl-cifar10/test/', 32, False, 1)
    batch_manager = CentralBatchManager(dataset)
    job_id = '1'
    for i in range(85):
       batch = batch_manager.get_next_batch(job_id)
       print(f'Batch {i+1}: {batch.batch_id}')
    pass
