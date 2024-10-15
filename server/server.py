import grpc
from concurrent import futures
import proto.minibatch_service_pb2 as minibatch_service_pb2
import proto.minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
import google.protobuf.empty_pb2
from logger_config import logger
import hydra
from omegaconf import DictConfig
from batch import Batch
from typing import Dict, List
from dataset import Dataset
from central_batch_manager import CentralBatchManager, DLTJob, PrefetchService
from args import SUPERArgs


class CacheAwareMiniBatchService(minibatch_service_pb2_grpc.MiniBatchServiceServicer):
    def __init__(self, args: SUPERArgs):
        self.args:SUPERArgs = args
        self.datasets: Dict[str,CentralBatchManager] = {}
        self.jobs: Dict[DLTJob] = {}

    def Ping(self, request, context):
        return minibatch_service_pb2.PingResponse(message = 'pong')
    
    def RegisterDataset(self, request, context):
        if request.data_dir in self.datasets:
            dataset =  self.datasets[request.data_dir].dataset
            if request.dataset_kind == 'vision':
                message = f"Dataset '{request.data_dir}' registered with SUPER. Total Files: {len(dataset)}, Total Batches: {dataset.num_batches},Total Partitions: {len(dataset.partitions)}"
            else:
                message = f"Dataset '{request.data_dir}' registered with SUPER. Total Files: {len(dataset)}"
            success = True
        else:
            dataset = Dataset(request.data_dir, self.args.batch_size, False, self.args.partitions_per_dataset, request.dataset_kind)
            self.datasets[request.data_dir] = CentralBatchManager(dataset=dataset, 
                                                                  args=self.args,)
            if request.dataset_kind == 'vision':
                message = f"Dataset '{request.data_dir}' registered with SUPER. Total Files: {len(dataset)}, Total Batches:{dataset.num_batches}, Partitions:{len(dataset.partitions)}"
            else:
                message = f"Dataset '{request.data_dir}' registered with SUPER. Total Files: {len(dataset)}"
            success = True
            logger.info(message)
        return minibatch_service_pb2.RegisterDatasetResponse(dataset_is_registered=success, 
                                                             total_batches=dataset.num_batches,
                                                             message=message)
    def JobUpdate(self, request, context):
        job_id = request.job_id
        data_dir = request.data_dir
        previous_step_batch_id = request.previous_step_batch_id
        previous_step_wait_for_data_time = request.previous_step_wait_for_data_time
        previous_step_is_cache_hit = request.previous_step_is_cache_hit
        previous_step_gpu_time = request.previous_step_gpu_time
        cached_previous_batch = request.cached_previous_batch
        
        self.datasets[data_dir].update_job_progess(job_id,
                                                   previous_step_batch_id,
                                                   previous_step_wait_for_data_time,
                                                   previous_step_is_cache_hit,
                                                   previous_step_gpu_time,
                                                   cached_previous_batch)
        return google.protobuf.empty_pb2.Empty()


    def GetNextBatchForJob(self, request, context):
        job_id = request.job_id
        data_dir = request.data_dir
       
        if data_dir not in self.datasets:
            message = f"Failed to register job with id '{job_id}' because data dir '{data_dir}' was not found in SUPER."
            logger.info(message)
        
        next_batch:Batch = self.datasets[data_dir].get_next_batch(job_id)
        # Create and return the response
        response = minibatch_service_pb2.GetNextBatchForJobResponse(
            job_id=request.job_id,
            batch=minibatch_service_pb2.Batch(batch_id=next_batch.batch_id, 
                                              indicies=next_batch.indicies, is_cached=next_batch.is_cached)
            )
        return response
    
    def JobEnded(self, request, context):
        job_id = request.job_id
        data_dir = request.data_dir
       
        self.datasets[data_dir].job_ended(job_id=job_id)
        return google.protobuf.empty_pb2.Empty()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def serve(config: DictConfig):
    try:
          
        logger.info("Starting SUPER Datloading Service")
        args:SUPERArgs = SUPERArgs(
            batch_size = config.workload.batch_size,
            partitions_per_dataset = config.partitions_per_dataset,
            lookahead_steps = config.lookahead_steps,
            serverless_cache_address = config.serverless_cache_address,
            use_prefetching = config.use_prefetching,
            use_keep_alive = config.use_keep_alive,
            prefetch_lambda_name = config.workload.prefetch_lambda_name,
            prefetch_cost_cap_per_hour=config.prefetch_cost_cap_per_hour,
            cache_evition_ttl_threshold = config.cache_evition_ttl_threshold,
            prefetch_simulation_time = config.prefetch_simulation_time,
            evict_from_cache_simulation_time = config.evict_from_cache_simulation_time,
            shuffle = config.workload.shuffle,
            drop_last = config.workload.drop_last,
            workload_kind = config.workload.kind)
        
        cache_service = CacheAwareMiniBatchService(args) 
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

        minibatch_service_pb2_grpc.add_MiniBatchServiceServicer_to_server(cache_service, server)
        server.add_insecure_port('[::]:50051')
        server.start()
        logger.info("Server started. Listening on port 50051...")

        # Keep the server running until interrupted
        server.wait_for_termination()

    except KeyboardInterrupt:
        logger.info("Server stopped due to keyboard interrupt")
        server.stop(0)
    except Exception as e:
            logger.exception(f"{e}. Shutting Down.")
            if  'server' in locals():
                server.stop(0)
    finally:
        pass
        # if 'coordinator' in locals():
        #     logger.info(f"Total Lambda Invocations:{coordinator.lambda_invocation_count}")
        #     coordinator.stop_workers()

if __name__ == '__main__':
    serve()
