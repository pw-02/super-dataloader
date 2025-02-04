import threading
from collections import deque, OrderedDict
from typing import List, Optional, Dict, Tuple
from dataset import CoorDLDataset
# from batch import Batch, BatchSet
import time
from logger_config import logger
import json
from botocore.config import Config
import redis
from typing import Iterator, Optional, Set
from args import CoorDLArgs
from typing import OrderedDict as TypingOrderedDict
from aws_utils import S3Url
import functools
import boto3
from boto3.exceptions import botocore
from utils import create_unique_id
from torch.utils.data import RandomSampler
import torch

class CoorDLBatch():
    def __init__(self, batch_indicies, batch_id, epoch_id):
        self.indicies: List[int] = batch_indicies
        self.epoch_id:int = epoch_id
        self.batch_id:str = batch_id
        self.access_count:int = 0
        self.has_been_accessed_before = False
        self.caching_in_progress:bool = False
        self.lock = threading.Lock()
        self.is_cached:bool = False
    
    def set_cache_status(self, is_cached:bool):
        """Set the cache status and handle cache eviction timer."""
        with self.lock:
            self.is_cached = is_cached
    
    def set_caching_in_progress(self, in_progress:bool):
        with self.lock:
            self.caching_in_progress = in_progress

class CoorDLBatchSet:
    def __init__(self, id:str):
        self.id = id
        self.batches: Dict[str, CoorDLBatch] = OrderedDict()

class CoorDLJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.epochs_completed_count = 0
        self.future_batches: OrderedDict[str, CoorDLBatch] = OrderedDict()    
        self.current_batch:CoorDLBatch = None
        self.lock = threading.Lock()
        self.step_idx = None

    def get_total_batches_assigned_to_job(self):
        return len(self.future_batches)
    
    def next_training_step_batch(self):
        with self.lock:
            next_training_batch = None
           # First pass: Find the first cached batch
            for batch_id, batch in list(self.future_batches.items()):
                    if batch.is_cached:
                        next_training_batch = self.future_batches.pop(batch_id)  # Cached batch found
                        break
            if not next_training_batch:
                for batch_id, batch in list(self.future_batches.items()):
                    if not batch.caching_in_progress:
                            next_training_batch = self.future_batches.pop(batch_id)  # Cached batch found
                            break   
            self.current_batch = next_training_batch
            return next_training_batch

class CoorDLDataset():
    def __init__(self, data_dir: str, 
                 batch_size: int, 
                 drop_last: bool, kind = 
                 'vision', 
                 max_dataset_size = None):
        
        # Load samples from data directory
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.samples = self.load_paired_s3_object_keys(data_dir, False, True)
        self.bucket_name =  S3Url(data_dir).bucket
         # Calculate the number of batches
        if self.drop_last:
            self.num_batches = len(self) // self.batch_size
        else:
            self.num_batches = (len(self) + self.batch_size - 1) // self.batch_size

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
                for class_index, blob_class in enumerate(self.samples)
                for blob in self.samples[blob_class]]
    
    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())
    
    def is_image_file(self, path: str):
        return any(path.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'])

    def load_paired_s3_object_keys(self, s3_uri:str, images_only:bool, use_index_file:bool = True, max_dataset_size = None):
        paired_samples = {}
        s3url = S3Url(s3_uri)
        s3_client = boto3.client('s3')
        if max_dataset_size:
            index_file_key = f"{s3url.key}_paired_index_{max_dataset_size}GB.json"
        else:
            index_file_key = f"{s3url.key}_paired_index.json"

        # index_file_key = s3url.key + '_paired_index.json'
        if use_index_file:
            try:
                index_object = s3_client.get_object(Bucket=s3url.bucket, Key=index_file_key)
                file_content = index_object['Body'].read().decode('utf-8')
                paired_samples = json.loads(file_content)
                return paired_samples
            except botocore.exceptions.ClientError as e:
                print(f"Error reading index file '{index_file_key}': {str(e)}")

        # If use_index_file is False or encounter errors with index file, build paired_samples from S3 objects
        paginator = s3_client.get_paginator('list_objects_v2')
        total_size_gb = 0
        pages = paginator.paginate(Bucket=s3url.bucket, Prefix=s3url.key)
        for page in pages:
            if max_dataset_size and total_size_gb >= max_dataset_size:
                    break
            for blob in page.get('Contents', []):
                if max_dataset_size and total_size_gb >= max_dataset_size:
                    break

                blob_path = blob.get('Key')
                if blob_path.endswith("/"):
                    continue  # Ignore folders
                
                stripped_path = self.remove_prefix(blob_path, s3url.key).lstrip("/")
                if stripped_path == blob_path:
                    continue  # No matching prefix, skip
                
                if images_only and not self.is_image_file(blob_path):
                    continue  # Skip non-image files
                
                if 'index.json' in blob_path:
                    continue

                blob_class = stripped_path.split("/")[0]
                blobs_with_class = paired_samples.get(blob_class, [])
                blobs_with_class.append(blob_path)
                paired_samples[blob_class] = blobs_with_class
                total_size_gb += blob['Size'] / 1024 / 1024 / 1024

        if use_index_file and len(paired_samples) > 0:
            index_object = s3_client.put_object(
                Bucket=s3url.bucket, 
                Key=index_file_key, 
                  Body=json.dumps(paired_samples, indent=4).encode('UTF-8'))    
        return paired_samples
    
    def remove_prefix(self, s: str, prefix: str) -> str:
        if not s.startswith(prefix):
            return s
        return s[len(prefix) :]


class CoorDLBatchManager:
    def __init__(self, dataset: CoorDLDataset, args:CoorDLArgs):
        self.dataset = dataset
        self.jobs: Dict[str, CoorDLJob] = {}        
        self.epoch_idx = 1
        self.epoch_batches: Dict[int, Dict [int, CoorDLBatchSet]] = OrderedDict()  #first key is epoch id, second key is partition id, value is the batches
        self.batch_size = args.batch_size
        self.drop_last = args.drop_last
        # Create a generator with a fixed seed
        generator = torch.Generator()
        generator.manual_seed(42)  # Fix seed for reproducibility

        self.sampler = RandomSampler(dataset, generator=generator)
        self.epoch_batches[self.epoch_idx] = self.genereate_bacthes_for_epoch()
        self.cache_host, self.cache_port = args.cache_address.split(":")
        # self.cache_client:redis.StrictRedis = redis.StrictRedis(host=self.cache_host, port=int(self.cache_port))
        self.cache_client:redis.StrictRedis = redis.StrictRedis(host=self.cache_host, port=int(self.cache_port), ssl=True)

        self.lock = threading.Lock()  # Lock for thread safety
    
    def genereate_bacthes_for_epoch(self):
        batch_list = {}
        batch_indices = []
        batch_count = 0
        for idx in self.sampler:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                batch_count += 1
                batch_id = f"{self.epoch_idx}_{batch_count}_{create_unique_id(batch_indices, 16)}"
                next_batch = CoorDLBatch(batch_indices, batch_id, self.epoch_idx)
                batch_list[batch_id] = next_batch
                batch_indices = []

        # Handle drop_last behavior
        if batch_indices and not self.drop_last:
            batch_id = f"{self.epoch_idx}_{batch_count}_{create_unique_id(batch_indices, 16)}"
            batch_count += 1
            next_batch = CoorDLBatch(batch_indices, batch_id, self.epoch_idx)
            batch_list[batch_id] = next_batch
        return batch_list
    
    def check_all_jobs_completed_epoch(self):
        all_jobs_completed_epoch = True
        for job in self.jobs.values():
            if len(job.future_batches) > 0:
                all_jobs_completed_epoch = False
                break
        return all_jobs_completed_epoch
    
    
    
    def update_job_progess(self, 
                       previous_step_batch_id,
                       previous_step_is_cache_hit,
                       previous_batch_cached_on_miss):
     with self.lock:

        if previous_step_batch_id in self.epoch_batches[self.epoch_idx]:
            batch = self.epoch_batches[self.epoch_idx][previous_step_batch_id]
        else:
            batch = self.epoch_batches[self.epoch_idx -1][previous_step_batch_id]
            
        if previous_batch_cached_on_miss or previous_step_is_cache_hit:
            batch.set_cache_status(True)
        else:
            batch.set_cache_status(False)
        batch.access_count += 1
        if batch.access_count >= len(self.jobs):
            self.cache_client.delete(previous_step_batch_id)
    
    def get_next_batch(self, job_id: str) -> Optional[CoorDLBatch]:
        with self.lock:    
            if job_id not in self.jobs:
                logger.info(f"Registering job '{job_id}'")
                job = self.jobs.setdefault(job_id, CoorDLJob(job_id))
                job.future_batches.update(self.epoch_batches[self.epoch_idx])
            else:
                job = self.jobs[job_id]

            if self.check_all_jobs_completed_epoch():
                self.epoch_idx += 1
                self.epoch_batches[self.epoch_idx] = self.genereate_bacthes_for_epoch()
                job.epochs_completed_count += 1
                for job in self.jobs.values():
                    job.epochs_completed_count += 1
                    job.future_batches.update(self.epoch_batches[self.epoch_idx])               

            if len(job.future_batches) == 0 and not self.check_all_jobs_completed_epoch():
                return None
            else:
                next_batch:CoorDLBatch = job.next_training_step_batch()    
                #check if all jobs accessed this batch
                # if next_batch.access_count == len(self.jobs):
                #     self.cache_client.delete(next_batch.batch_id)
                #     pass

                if not next_batch.is_cached and not next_batch.caching_in_progress:
                    next_batch.set_caching_in_progress(True)
                return next_batch
            
        
    def job_ended(self, job_id):
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                logger.info(f"Job '{job_id}' ended." )
                self.jobs.pop(job_id)
    
if __name__ == "__main__":
    # Constants
    MISS_WAIT_FOR_DATA_TIME = 1.4
    HIT_WAIT_FOR_DATA_TIME = 0.001
    NUM_JOBS = 1 # Number of parallel jobs to simulate
    DELAY_BETWEEN_JOBS = 0.1  # Delay in seconds between the start of each job
    BATCHES_PER_JOB = 392  # Number of batches each job will process
    GPU_TIME = 0.01
 
    coordl_args:CoorDLArgs = CoorDLArgs(
            batch_size = 128,
            lookahead_steps = 1000,
            cache_address = '127.0.0.1:6379',
            shuffle = False,
            drop_last = False,
            workload_kind = 'vision')
    
    dataset = CoorDLDataset(data_dir='s3://sdl-cifar10/train/', 
                             batch_size=coordl_args.batch_size, 
                             drop_last=coordl_args.drop_last)
    
    batch_manager = CoorDLBatchManager(dataset=dataset, args=coordl_args)
    cache_client:redis.StrictRedis = redis.StrictRedis(host='127.0.0.1', port=6379)
    job_id = '1'
    BATCHES_PER_JOB = 394  # Number of batches each job will process
    end = time.perf_counter()
    for i in range(BATCHES_PER_JOB):
        batch:CoorDLBatch = batch_manager.get_next_batch(job_id=job_id)
        logger.info(f'Setp {i+1}, Job {job_id}, {batch.batch_id}')
        cache_client.set(batch.batch_id, 'data')
        batch_manager.update_progess(batch.batch_id, False, True, batch.epoch_id)
    batch_manager.job_ended(job_id)


    time.sleep(5)