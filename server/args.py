from dataclasses import dataclass

@dataclass
class SUPERArgs:
    workload_kind:str
    s3_data_dir: str
    create_batch_lambda: str
    batch_size: int
    drop_last:bool
    simulate_mode:bool
    keep_alive_ping_iterval:int
    max_lookahead_batches:int
    max_prefetch_workers:int
    cache_address:str
    shuffle:str
    num_dataset_partitions:int