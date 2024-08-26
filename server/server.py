import grpc
from concurrent import futures
import proto.minibatch_service_pb2 as minibatch_service_pb2
import proto.minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
import google.protobuf.empty_pb2
from logger_config import logger
import hydra
from omegaconf import DictConfig
from args import SUPERArgs
from batch import Batch
from typing import Dict, List
from dataset import Dataset
from central_batch_manager import CentralBatchManager, DLTJob

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
            message = f"Dataset '{request.data_dir}' registered with SUPER. Total Files: {len(dataset)}, Total Batches: {dataset.num_batches},Total Partitions: {len(dataset.partitions)}"
            success = True
        else:
            dataset = Dataset(request.data_dir, self.args.batch_size, self.args.drop_last, self.args.num_dataset_partitions, self.args.workload_kind)
            self.datasets[request.data_dir] = CentralBatchManager(dataset)
            message = f"Dataset '{request.data_dir}' registered with SUPER. Total Files: {len(dataset)}, Total Batches:{dataset.num_batches}, Partitions:{len(dataset.partitions)}"
            success = True
            logger.info(message)
        return minibatch_service_pb2.RegisterDatasetResponse(dataset_is_registered=success, 
                                                             total_batches=dataset.num_batches,
                                                             message=message)
    
    def GetNextBatchForJob(self, request, context):
        job_id = request.job_id
        data_dir = request.data_dir

        if data_dir not in self.datasets:
            message = f"Failed to register job with id '{job_id}' because data dir '{data_dir}' was not found in SUPER."
            logger.info(message)
        
        next_batch = self.datasets[data_dir].get_next_batch(job_id)
        # Create and return the response
        response = minibatch_service_pb2.GetNextBatchForJobResponse(
            job_id=request.job_id,
            batch=minibatch_service_pb2.Batch(batch_id=next_batch.batch_id, indicies=next_batch.indicies, is_cached=next_batch.is_cached)
            )
        return response
    

    # def GetDatasetInfo(self, request, context):
    #     num_files, num_chunks,chunk_size =  self.coordinator.get_dataset_info(request.data_dir)
    #     return minibatch_service_pb2.DatasetInfoResponse(num_files=num_files, num_chunks=num_chunks,chunk_size=chunk_size)

    
    # def JobEnded(self, request, context):
    #     self.coordinator.handle_job_ended(job_id=request.job_id)
    #     logger.info(f"Job'{request.job_id}' has ended.")
    #     return google.protobuf.empty_pb2.Empty()

       
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def serve(config: DictConfig):
    try:
        logger.info("Starting SUPER Datloading Service")
        super_args:SUPERArgs = SUPERArgs(
            workload_kind =config.workload.kind,
            s3_data_dir = config.workload.s3_data_dir,
            create_batch_lambda = config.create_batch_lambda,
            batch_size = config.workload.batch_size,
            drop_last=config.workload.drop_last,
            simulate_mode = config.simulate_mode,
            keep_alive_ping_iterval = config.workload.keep_alive_ping_iterval,
            max_lookahead_batches = config.workload.max_lookahead_batches,
            max_prefetch_workers = config.workload.max_prefetch_workers,
            cache_address = config.cache_address,
            shuffle=config.workload.shuffle,
            num_dataset_partitions =config.workload.num_dataset_partitions)
        
        cache_service = CacheAwareMiniBatchService(super_args) 
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
