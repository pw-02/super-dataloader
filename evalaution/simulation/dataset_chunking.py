import heapq
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import csv
import os
import random
import string

# class UnifiedSampler:
#     def __init__(self, num_batches_per_epoch, chunk_size_as_percentage = 1):
#         self.num_batches_per_epoch = num_batches_per_epoch
#         self.num_chunks_per_epoch = int(1 / chunk_size_as_percentage)
#         self.num_batches_per_chunk = int(num_batches_per_epoch * chunk_size_as_percentage)
#         self.epoch_counter = 0
#         self.chunk_counter = 0
#         self.active_chunks = {}
#         self.create_new_chunk()

#     def create_new_chunk(self):

#         #genrate a bunch of rando ids for the next chunk
#         list_of_items_for_the_next_chunk = [''.join(random.choices(string.ascii_letters, k=8)) for _ in range(self.num_batches_per_chunk)]
#         self.active_chunks[f'{self.epoch_counter}_{self.chunk_counter}'] = list_of_items_for_the_next_chunk
#         self.chunk_counter += 1
#         if self.chunk_counter >= self.num_chunks_per_epoch:
#             self.chunk_counter = 0
#             self.epoch_counter += 1
    

#     def get_next_chunk(self, chunks_processed_by_jobs):
#          #check if job has processed all the chunks in active_chunks and if so create a new chunk
#         unprocessed_chunk = None
#         for chunk_id, chunk in self.active_chunks.items():
#             if chunk_id not in chunks_processed_by_jobs:
#                 unprocessed_chunk = chunk_id
#                 break
#         if unprocessed_chunk is None:
#             self.create_new_chunk()
#             unprocessed_chunk = self.chunk_counter
#         return unprocessed_chunk, self.active_chunks[unprocessed_chunk]



class Cache:
    def __init__(self,num_jobs):
        self.cache = {}
        self.num_jobs = num_jobs
        self.cache_size_over_time = {}

    def get_or_insert_batch(self, requested_batch, current_time):
        cache_hit = False
        if requested_batch not in self.cache:
            self.cache[requested_batch] = 1
        else:
            self.cache[requested_batch] += 1
            if self.cache[requested_batch] >= self.num_jobs:  # Remove if all jobs accessed
                del self.cache[requested_batch]
            cache_hit = True

        self.cache_size_over_time[current_time] = len(self.cache)
        return cache_hit
    

class MLJob:
    def __init__(self, job_id, time_per_batch, total_batches_to_process,num_batches_per_chunk, cache:Cache):
        self.job_id = job_id
        self.time_per_batch = time_per_batch
        self.total_batches_to_process = total_batches_to_process
        self.batches_processed = 0
        self.next_available_time = 0
        self.batch_list = []
        self.num_batches_per_chunk = num_batches_per_chunk
        self.cache = cache
        self.cache_hits = 0
        self.cache_misses = 0
        self.current_chunk_id = None
        self.processed_chunks = []
        self.current_chunk = 0
        self.current_epoch = 1
        self.current_batch = 1

    def job_is_eneded(self):
        return self.batches_processed >= self.total_batches_to_process
    
    def __lt__(self, other):
        return self.next_available_time < other.next_available_time  
    
    def process_next_batch(self):
        if self.job_is_eneded():
            return False
        
        is_cache_hit = self.cache.get_or_insert_batch((self.current_chunk,self.current_epoch, self.current_batch), 
                                                      self.next_available_time)
        if is_cache_hit:
            self.cache_hits += 1
            self.next_available_time += self.time_per_batch
        else:
            self.cache_misses += 1
            # self.next_available_time += 2*self.time_per_batch
            self.next_available_time += self.time_per_batch

        if self.current_batch == self.num_batches_per_chunk:
            self.current_batch = 1
            next_chunk = queue[(queue.index(current_chunk) + 1) % len(queue)]
            if next_chunk == 1:
                self.current_epoch += 1  # New epoch starts
            self.current_chunk = next_chunk
        
        # Move to next chunk
        next_chunk = queue[(queue.index(current_chunk) + 1) % len(queue)]
         
        if len(self.batch_list) == 0:
            if self.current_chunk_id is not None:
                self.processed_chunks.append(self.current_chunk_id)
            self.current_chunk_id, self.batch_list = self.sampler.get_next_chunk(self.processed_chunks)
        
        next_batch = self.batch_list.pop(0)
        self.batches_processed += 1
  


def run():

    # Define the dataset
    dataset_size = 1000
    batch_size = 10
    batches_per_epoch = dataset_size // batch_size
    toal_epochs = 10
    total_batches_to_process = batches_per_epoch * toal_epochs

    sampler = UnifiedSampler(batches_per_epoch, chunk_size_as_percentage=1) # 10% of the dataset


   
    cache = Cache(num_jobs=2)
    job_speeds = [5, 1]

    jobs = [MLJob(job_id,
                  time_per_batch, 
                  total_batches_to_process,
                  sampler,
                  cache) for job_id, time_per_batch in enumerate(job_speeds)] 
    
    job_queue: List[MLJob] = []
    for job in jobs:
        heapq.heappush(job_queue, job)
    
    training_finished = False
    while not training_finished:
        training_finished = True  # Assume training is done unless proven otherwise
        if job_queue:
            training_finished = False  # There are jobs still running
            job = heapq.heappop(job_queue)  # Get the job with the earliest available time

            if job.job_is_eneded():
                logging.info(f"Job {job.job_id} has no more batches to process.")
                continue
            job.process_next_batch()
            heapq.heappush(job_queue, job)
    
    for job in jobs:
        throughput = job.batches_processed / job.next_available_time
        logging.info(f"Throughput for Job {job.job_id}: {throughput:.2f} batches per second")


if __name__ == "__main__":
    run()


