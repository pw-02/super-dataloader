import time
import json
import threading
from typing import Dict, List, Set
from args import SUPERArgs
from logger_config import logger
from dataset import Dataset
from job import MLTrainingJob
from sampling import BatchSampler, EndOfEpochException
from utils import TokenBucket, format_timestamp, remove_trailing_slash
from aws_utils import AWSLambdaClient
from concurrent.futures import ThreadPoolExecutor
from batch import Batch
from epoch import Epoch

class SUPERCoordinator:
    def __init__(self, args:SUPERArgs, dataset:Dataset):
        self.args: SUPERArgs = args
        self.dataset:Dataset = dataset
        self.epoch_partitions: Dict[int, List[Epoch]] = {}
        self.active_epoch_partition:Epoch = None
        self.jobs: Dict[int, MLTrainingJob] = {}
        self.lambda_client:AWSLambdaClient = AWSLambdaClient()
        self.prefetch_batches_stop_event = threading.Event()
        self.token_bucket:TokenBucket = TokenBucket(capacity=args.max_lookahead_batches, refill_rate=0)
        self.executor = ThreadPoolExecutor(max_workers=args.max_prefetch_workers)  # Adjust max_workers as neede
        self.workload_kind = self.args.workload_kind
        
    def start_prefetcher_service(self):
        """Starts the prefetching process."""
        try:
            prefetch_thread = threading.Thread(target=self.prefetch, daemon=True, name='prefetch-thread')
            prefetch_thread.start()
        except Exception as e:
            logger.error(f"Error starting prefetching service: {e}")
    
    def start_keep_batches_alive_service(self):
        """Starts the prefetching process."""
        try:
            prefetch_thread = threading.Thread(target=self.monitor_active_batches, daemon=True, name='monitor-batches-thread')
            prefetch_thread.start()
        except Exception as e:
            logger.error(f"Error starting prefetching service: {e}")


    def prefetch(self):
        """Prefetches data and handles batch processing."""
        epoch_idx = 0
        while not self.prefetch_batches_stop_event.is_set():
            epoch_idx += 1
            for partition_id, partition in self.dataset.partitions.items():
                self.active_epoch_partition = Epoch(f'{epoch_idx}_{partition_id}', partition_id)
                self.epoch_partitions.setdefault(partition.partition_id, []).append(self.active_epoch_partition)
                batch_sampler = BatchSampler(len(partition), self.args.batch_size, self.active_epoch_partition.epoch_id, self.args.shuffle, self.args.drop_last)
                # Process batches until the end of the epoch
                while True:
                    try:
                        self.token_bucket.wait_for_tokens()
                        next_batch = next(batch_sampler)
                        self.active_epoch_partition.add_batch(next_batch)
                        # Submit the batch for prefetching
                        self.executor.submit(self.prefetch_batch, next_batch)
                    except EndOfEpochException:
                        # End of epoch, break the loop
                        break
                # Mark the active epoch as finalized
                self.active_epoch_partition.batches_finalized = True
                self.token_bucket.refill(1) #refill here because the pevious token wasn't used due to end of epoch

    def create_new_job(self, job_id, data_dir):
        if job_id in self.jobs:
            message = f"Job with id '{job_id}' already registered. Skipping."
            success  = False
        elif remove_trailing_slash(data_dir).casefold() != remove_trailing_slash(self.dataset.data_dir).casefold():
            success  = False
            message = f"Failed to register job with id '{job_id}' because data dir '{data_dir}' was not found in SUPER."
        else:
            self.jobs[job_id] = MLTrainingJob(job_id)
            message = f"New job with id '{job_id}' successfully registered."
            success  = True   
        return success, message
    
    def allocate_next_partition_epoch_to_job(self, job: MLTrainingJob):
        # Check if there are no epoch partitions remaining in the job
        if len(job.partition_epochs_remaining) == 0:
            # Assign active epoch to the job
            job.reset_partition_epochs_remaining(self.dataset.partitions)
            self.active_epoch_partition.queue_up_batches_for_job(job.job_id)
            job.current_partition_epoch = self.active_epoch_partition
        else:
            # Set the current index to the starting integer
            current_index = self.active_epoch_partition.partition_id
            
            # Initialize a variable to track the found epoch
            found_epoch = None
            
            # Iterate in reverse with wrap-around
            for _ in range(len(self.dataset.partitions)):
                # Check if the current index is in job.epoch_partitions_remaining
                if current_index in job.partition_epochs_remaining:
                    last_epoch = self.epoch_partitions[current_index][-1]
                    
                    # Check if the last epoch ID is not in the job's epoch history
                    if last_epoch.epoch_id not in job.epoch_history:
                        # Assign the found epoch and break the loop
                        found_epoch = last_epoch
                        last_epoch.queue_up_batches_for_job(job.job_id)
                        job.current_partition_epoch = last_epoch
                        break
                
                # Move to the previous index, wrapping around if necessary
                current_index = (current_index - 2) % len(self.dataset.partitions) + 1
            
            # If no suitable epoch was found, handle the case as needed
            if found_epoch is None:
                raise ValueError("No suitable partition epoch found.")


    def next_batch_for_job(self, job_id, num_batches_requested = 1): 
        job:MLTrainingJob = self.jobs[job_id]
        job.update_batch_processing_rate()
        if job.current_partition_epoch is None:
            self.allocate_next_partition_epoch_to_job(job)

        elif job.current_partition_epoch.batches_finalized and job.current_partition_epoch.pending_batch_accesses[job.job_id].empty():
            #job finished its current partition epoch
            job.partition_epochs_remaining.remove(job.current_partition_epoch.partition_id)
            job.epoch_history.append(job.current_partition_epoch.epoch_id)
            self.allocate_next_partition_epoch_to_job(job)

        while job.current_partition_epoch.pending_batch_accesses[job.job_id].qsize() < 1:
            time.sleep(0.01)

        num_items = min(job.current_partition_epoch.pending_batch_accesses[job.job_id].qsize(), num_batches_requested)
        next_batches = []

        for i in range(0, num_items):
            batch_id = job.current_partition_epoch.pending_batch_accesses[job.job_id].get()
            batch:Batch = job.current_partition_epoch.batches[batch_id]
            if batch.is_first_access():
              self.token_bucket.refill(1)

            batch.set_last_accessed_time()        
            next_batches.append(batch)
        return next_batches
           
    def get_dataset_info(self, data_dir):
        num_files = len(self.dataset)
        num_chunks = self.dataset.num_batches
        chunk_size = self.args.batch_size
        return num_files, num_chunks, chunk_size
    
    def prefetch_batch(self, next_batch: Batch, attempt=1, mode='prefetch'):
        if attempt > 5:
            return False
        try:
            payload = {
                'bucket_name': self.dataset.bucket_name,
                'batch_id': next_batch.batch_id,
                'batch_samples': self.dataset.get_samples(next_batch.indicies),
                'workload_kind': self.args.workload_kind,
                'task': mode,
                'cache_address': self.args.cache_address
            }
            response = self.lambda_client.invoke_function(
                self.args.create_batch_lambda, json.dumps(payload), self.args.simulate_mode
            )

            if response['success']:
                logger.info(f"Invoked lambda function for batch {next_batch.batch_id}, Mode: {mode}, Duration: {response['duration']:.3f}s")
                next_batch.set_cache_status(is_cached=True)
                next_batch.set_last_accessed_time()
            else:
                logger.error(f"Failed to invoke lambda function for batch: {next_batch.batch_id}, Message: {response['message']}, Request Duration: {response['duration']:.3f}s, Attempt: {attempt}")
                self.prefetch_batch(next_batch, attempt + 1, mode)
            return True
        except Exception as e:
            logger.error(f"Error in prefetch_batch: {type(e).__name__} - {e}")
            return False
        
    # def prefetch_batch(self, next_batch:Batch, attempt = 1, mode = 'prefetch'):
    #     if attempt >  5:
    #         return False
    #     try:
    #         payload = {
    #             'bucket_name': self.dataset.bucket_name,
    #             'batch_id': next_batch.batch_id,
    #             'batch_samples': self.dataset.get_samples(next_batch.indicies),
    #             'workload_kind':  self.args.workload_kind,
    #             'task':mode,
    #             'cache_address': self.args.cache_address
    #             }
    #         response = self.lambda_client.invoke_function(self.args.create_batch_lambda,json.dumps(payload), self.args.simulate_mode)

    #         if response['success'] == True:
    #             logger.info(f"Invoked lambda fucntion for batch {next_batch.batch_id}, Model: {mode}, Duration: {response['duration']:.3f}s")
    #             next_batch.set_cache_status(is_cached=True)
    #             next_batch.set_last_accessed_time()
    #         else:
    #             logger.error(f"Failed to Invoke lambda fucntion for batch: {next_batch.batch_id}, Message: {response['message']}, Request Duration: {response['duration']:.3f}s, Attempt: {attempt}")
    #             attempt +=1
    #             self.prefetch_batch(next_batch,attempt)
    #         return True
    #     except Exception as e:
    #         logger.error(f"Error in prefetch_batch: {e}")
    #         return False
        
    def stop_workers(self):
        self.prefetch_batches_stop_event.set()
        #self.executor.shutdown(wait=False)
    
    def handle_job_ended(self, job_id):
        self.jobs[job_id].is_active = False
        self.deactivate_inactive_epochs()
        self.jobs.pop(job_id)
    

    def monitor_active_batches(self):
        while not self.prefetch_batches_stop_event.is_set():
            time.sleep(10)
            active_epochs_set:Set[Epoch] = set()
            for job in self.jobs.values():
                if job.is_active and job.current_partition_epoch is not None:
                    active_epochs_set.add(job.current_partition_epoch)
            active_epochs_set.add(self.active_epoch_partition)

            for epoch in active_epochs_set:
                batches: Dict[str, Batch] = epoch.batches
                for batch in batches.values():
                    if batch.last_accessed_time is not None:
                         time_since_last_accessed = time.time() - batch.last_accessed_time
                         if time_since_last_accessed > self.args.keep_alive_ping_iterval:
                            self.prefetch_batch(batch,attempt=1, mode='keep_alive')


    def test_prefetch_rate(self, num_items_to_process = 10, num_workers = 10):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Use a thread pool to invoke the Lambda function for each item in the iterator
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            start_time = time.time()            
            # Create a list to hold the futures (tasks)
            futures = []
            item_count = 0  # Initialize item count
            for item in self.batch_sampler:
                # Submit a task to invoke the Lambda function for the current item
                futures.append(executor.submit(self.prefetch_batch, item))
                item_count += 1
                # Stop processing once 40 items have been processed
                if item_count >= num_items_to_process:
                    break
            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                # Optionally, process the result further if needed
                print(f"Lambda response: {result}")
            
            # End timing
            end_time = time.time()
            
        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Print the elapsed time
        print(f"Elapsed time for processing {num_items_to_process} items from the iterator: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    super_args:SUPERArgs = SUPERArgs()
    dataset = Dataset('')
    coordinator = SUPERCoordinator(super_args, dataset)
    # Start worker threads
    coordinator.start_workers()
    time.sleep(2)
    try:
        job1 = 1
        coordinator.create_new_job(job1, 's3://sdl-cifar10/train/')

        for i in range(0, 10):
            batch = coordinator.next_batch_for_job(job1)
            for b in batch:
                logger.info(f'Job {job1}, Batch_index {i+1}, Batch_id {b.batch_id}')
            time.sleep(1)
        
         # Infinite loop to keep the program running
        while True:
            # Do nothing or perform some tasks here
            pass

            # coordinator.run_batch_access_predictions()
    finally:
        # Stop worker threads before exiting
        coordinator.stop_workers()
