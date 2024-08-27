import math

#the optmial prefetch conccurency can be calculated by the following formula: Tprefetch / Thit. This 
#ensures that time spent on a cycle of cache hits matches the prefetch time, 
# maximizing the chances that the next set of batches are ready in the cache when needed.

def find_optimal_prefetch_conncurrency(T_prefetch=5, T_hit=0.25,):
    optimal_prefetch_concurrency = math.ceil(T_prefetch / T_hit)
    return optimal_prefetch_concurrency

def estimate_prefetch_cost(cost_per_prefetch, number_of_prefetches_per_hour, cost_cap_per_hour):
    total_cost_per_hour = cost_per_prefetch * number_of_prefetches_per_hour
    max_allowed_prefetches = cost_cap_per_hour / cost_per_prefetch
    pass

def calculate_metrics(T_prefetch=5, N_prefetch=15, T_hit=0.25, T_miss=5.25, B_total=1000):
    """
    Calculate total cache hits and misses during the prefetching process.
    
    Parameters:
        T_prefetch (float): Time allocated for prefetching in seconds.
        N_prefetch (int): Number of batches that can be prefetched within the T_prefetch time.
        T_hit (float): Time taken for a cache hit in seconds.
        T_miss (float): Time taken for a cache miss in seconds.
        B_total (int): Total number of batches to be processed.
        
    Returns:
        total_misses (int): Total number of cache misses.
        total_hits (int): Total number of cache hits.
    """
    # Calculate the number of misses that occur during the initial prefetching phase of N_prefetch batches
    misses_during_first_prefetch = math.ceil(T_prefetch / T_miss)
  
    # Number of hits before we start getting misses after the prefetch
    hits_per_cycle = N_prefetch

    # Time gap after processing all the prefetched hits
    time_gap = max(T_prefetch - (hits_per_cycle * T_hit),0)
    
    # Number of misses that can occur during this time gap
    misses_per_cycle = math.ceil(time_gap / T_miss)
    
    # Number of hits and misses in each cycle
    cycle_size = hits_per_cycle + misses_per_cycle

    total_cycles = (B_total-1) / cycle_size #we minius 1 for the first prefetching phase not included in the cycle

    # Calculate total misses and hits
    total_misses = math.floor(total_cycles * misses_per_cycle + misses_during_first_prefetch)
    total_hits = B_total - total_misses
    total_time = int(total_hits) * T_hit + int(total_misses) * T_miss
    return int(total_misses), int(total_hits), total_time


T_prefetch=5
T_hit=0.25
N_prefetch=15
T_miss=5
B_total=79

opt_N_prefetch = find_optimal_prefetch_conncurrency(T_prefetch=5, T_hit=0.25)

misses, hits, total_time = calculate_metrics(T_prefetch, N_prefetch, T_hit, T_miss, B_total)
print(f"Total cache misses: {misses}")
print(f"Total cache hits: {hits}")
print(f"Total Time: {total_time}")
