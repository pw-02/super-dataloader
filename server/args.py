from dataclasses import dataclass

@dataclass
class SUPERArgs:
    partitions_per_dataset:int
    lookahead_steps:int
    serverless_cache_address:str
    cache_evition_ttl_threshold:float
    use_prefetching:bool
    prefetch_lambda_name:str
    prefetch_cost_cap_per_hour:float
    prefetch_simulation_time:float