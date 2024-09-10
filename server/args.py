from dataclasses import dataclass

@dataclass
class SUPERArgs:
    partitions_per_dataset:int
    batch_size:int
    lookahead_steps:int
    serverless_cache_address:str
    cache_evition_ttl_threshold:float
    use_prefetching:bool
    use_keep_alive:bool
    prefetch_lambda_name:str
    prefetch_cost_cap_per_hour:float
    prefetch_simulation_time:float
    evict_from_cache_simulation_time:float
    shuffle:bool
    drop_last:bool