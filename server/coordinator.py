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
from concurrent.futures import ThreadPoolExecutor, as_completed
from batch import Batch
from epoch import Epoch
from central_batch_manager import CentralBatchManager

class SUPERCoordinator:
    def __init__(self, args: SUPERArgs):
        self.args: SUPERArgs = args
        self.datasets: Dict[Dataset] = {}
        self.jobs: Dict[MLTrainingJob] = {}

        # self.epoch_partitions: Dict[int, List[Epoch]] = {}
        # self.active_epoch_partition: Epoch = None
        # self.jobs: Dict[int, MLTrainingJob] = {}
        # self.lambda_client: AWSLambdaClient = AWSLambdaClient()
        # self.prefetch_batches_stop_event = threading.Event()
        # self.token_bucket: TokenBucket = TokenBucket(capacity=args.max_lookahead_batches, refill_rate=0)
        # self.executor = ThreadPoolExecutor(max_workers=args.max_prefetch_workers)
        # self.workload_kind = self.args.workload_kind
        # self.lambda_invocation_count  = 0
        # self.lambda_invocation_lock = threading.Lock()

    def create_dataset(self, data_dir):
        if data_dir in self.datasets:
            dataset =  self.datasets[data_dir]
            message = f"Dataset '{data_dir}' registered with SUPER. Total Files: {len(dataset)}, Total Batches: {dataset.num_batches},Total Partitions: {len(dataset.partitions)}"
            success = True
        else:
            dataset = Dataset(data_dir, self.args.batch_size, self.args.drop_last, self.args.num_dataset_partitions, self.args.workload_kind)
            self.datasets[data_dir] = CentralBatchManager(dataset)
            message = f"Dataset '{data_dir}' registered with SUPER. Total Files: {len(dataset)}, Total Batches:{dataset.num_batches}, Partitions:{len(dataset.partitions)}"
            success = True
        return success, message
    
    def fetch_next_bacth_for_job(self, job_id, data_dir):

        

        job = self.jobs[job_id]
        try:
            batch = job.fetch_batch()
            return batch
        except EndOfEpochException:
            return None

    
    def create_new_job(self, job_id, data_dir):
        if job_id in self.jobs:
            message = f"Job with id '{job_id}' already registered. Skipping."
            success = False
        elif remove_trailing_slash(data_dir).casefold() != remove_trailing_slash(self.dataset.data_dir).casefold():
            success = False
            message = f"Failed to register job with id '{job_id}' because data dir '{data_dir}' was not found in SUPER."
        else:
            self.jobs[job_id] = MLTrainingJob(job_id)
            message = f"New job with id '{job_id}' successfully registered."
            success = True
        return success, message
    
    # def fetch_bacth_for_job(self, job_id, data_dir):
    #     job = self.jobs[job_id]
    #     try:
    #         batch = job.fetch_batch()
    #         return batch
    #     except EndOfEpochException:
    #         return None




if __name__ == "__main__":
    super_args: SUPERArgs = SUPERArgs()
    dataset = Dataset('s3://sdl-cifar10/train/')
    coordinator = SUPERCoordinator(super_args, dataset)
    coordinator.start_prefetcher_service()
    coordinator.start_keep_batches_alive_service()
    time.sleep(2)
    try:
        job1 = 1
        coordinator.create_new_job(job1, 's3://sdl-cifar10/train/')
        for i in range(10):
            batch = coordinator.next_batch_for_job(job1)
            for b in batch:
                logger.info(f'Job {job1}, Batch_index {i+1}, Batch_id {b.batch_id}')
            time.sleep(1)
        while True:
            pass
    finally:
        coordinator.stop_workers()
