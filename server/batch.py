
from typing import List
import threading
# from utils.utils import create_unique_id
import time

class Batch:
    def __init__(self, batch_indicies, batch_id, epoch_idx, partition_idx):
        self.indicies: List[int] = batch_indicies
        self.epoch_idx:int = epoch_idx
        self.partition_id:int = partition_idx
        self.batch_id:str = batch_id
        # self.epoch_seed:int = epoch_seed
        self.is_cached:bool = False
        self.caching_in_progress:bool = False
        self.next_access_time:float = None
        self.last_accessed_time:float = 0 #None #float('inf')
        self.has_been_accessed_before = False
        self.lock = threading.Lock()  # Lock for accessing shared resources

    def time_since_last_access(self):
        with self.lock:
            return time.time() - self.last_accessed_time
        
    def set_last_accessed_time(self):
        with self.lock:
            self.last_accessed_time = time.time()
    
    def is_first_access(self):
        with self.lock:
            if self.has_been_acessed_before:
                return False
            else:
                self.has_been_acessed_before = True
                return True

    def set_cache_status(self, is_cached:bool, simulate_ttl_eviction:float = None):
        with self.lock:
            self.is_cached = is_cached
        
    
    def set_caching_in_progress(self, in_progress:bool):
        with self.lock:
            self.caching_in_progress = in_progress
            
    def set_has_been_accessed_before(self, has_been_accessed_before:bool):
        with self.lock:
            self.has_been_accessed_before = has_been_accessed_before

