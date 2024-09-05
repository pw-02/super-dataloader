import hashlib
import time
import math
from queue import Queue
import threading
import threading
from queue import Empty
from typing import Any, Dict
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        fmtstr = "{name}:{val" + self.fmt +"}"
        return fmtstr.format(**self.__dict__)


class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.refill_count  =0
        self.last_refill_time = time.time()
        self.prefetched_batches = set()  # Keep track of downloaded/prefecthed bacthes
        self.lock = threading.Lock()  # Lock for accessing shared resources
    
    def refill(self, tokens_to_add =1):
        with self.lock:
            now = time.time()
            delta_time = now - self.last_refill_time
            # tokens_to_add = delta_time * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill_time = now
            self.refill_count +=1

    def consume(self, tokens):
        with self.lock:
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            else:
                return False

    # def batch_prefeteched(self, batch_id):
    #     with self.lock:
    #         self.prefetched_batches.add(batch_id)
    
    # def batch_accessed(self, batchs):
    #     with self.lock:
    #         if batch_id in self.prefetched_batches:
    #             self.refill()
    #             #remove so that a token only gets added to the bucket the first time the batch is accessed
    #             #without this the lookahead_rate config setting would be inaccurate
    #             self.prefetched_batches.remove(batch_id)

    def wait_for_tokens(self):
        while not self.consume(1):
            time.sleep(0.01)  # Wait if there are not enough tokens available
            # token_bucket.refill()  # Refill tokens during the wait
    


class CustomQueue(Queue):
    def __iter__(self):
        return iter(self.queue)
    
    def remove_item(self, item, block=True, timeout=None):

        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            if item in self.queue:
                self.queue.remove(item)     
            # item = self._get()
            self.not_full.notify()
            return item
    
def gen_partition_epoch_key(epoch_id:int, partition_id:int):
        return f"{epoch_id}_{partition_id}"

def create_unique_id(int_list, length = 32):
    # Convert integers to strings and concatenate them
    id_string = ''.join(str(x) for x in int_list)
    
    # Hash the concatenated string to generate a unique ID
    unique_id = hashlib.md5(id_string.encode()).hexdigest()
    #Truncate the hash to the desired length
    return unique_id[:length]

def format_timestamp(current_timestamp, use_utc=True):
    if current_timestamp == math.inf:
        return current_timestamp
    if use_utc:
        time_struct = time.gmtime(current_timestamp)
    else:
        time_struct = time.localtime(current_timestamp)

    formatted_current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
    return formatted_current_time

def remove_trailing_slash(path:str):
    """
    Removes trailing slashes from a directory path.
    """
    path = path.strip()
    if path.endswith('/'):
        return path[:-1]  # Remove the last character (trailing slash)
    if path.endswith('\\'):
        return path[:-1]  # Remove the last character (trailing slash)
    return path

def partition_dict(original_dict: Dict[Any, Any], num_partitions, batch_size):
   # Initialize a list to hold the partitions
    partitions = [{} for _ in range(num_partitions)]
    
    # Iterate through each key and its associated list of values in the original dictionary
    for key, values in original_dict.items():
        # Calculate the total number of values
        total_values = len(values)
        
        # Calculate the base partition size such that it is divisible by batch_size
        base_partition_size = (total_values // num_partitions) // batch_size * batch_size
        
        # Calculate the remaining values that should be evenly distributed across partitions
        remaining_values = total_values - base_partition_size * num_partitions
        
        # Initialize the starting index for the partitions
        start_index = 0
        
        # Distribute the list of values across partitions
        for i in range(num_partitions):
            # Determine the size of the current partition
            if i < remaining_values:
                # If there are remaining values, add one additional base size (batch_size) to this partition
                partition_size = base_partition_size + batch_size
            else:
                # Otherwise, use the base partition size
                partition_size = base_partition_size
            
            # Calculate the end index for the current partition
            end_index = min(start_index + partition_size, total_values)
            
            # Get the subset of values for the current partition
            subset_values = values[start_index:end_index]
            
            # Add the key and subset of values to the current partition dictionary
            partitions[i][key] = subset_values
            
            # Update the start index for the next partition
            start_index = end_index
    
    return partitions


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
