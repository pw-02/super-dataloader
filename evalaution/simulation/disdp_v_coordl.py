import heapq
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import csv
import os

asws_redis_instnces = {
    "cache.r7g.8xlarge	": {'memory': 209, 'price_per_hour': 2.7928},
    "cache.r5.24xlarge": {'memory': 635, 'price_per_hour': 8.2944},
    "cache.r7g.4xlarge": {'memory': 105, 'price_per_hour': 1.396}
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler("simulation.log", mode="w")  # Save to file
    ]
)

# cached_batches = {}
# cache_size_over_time = []  # Track cache size at each step
# cache_budget_per_hour = 25 #dollars
# current_cache_hourly_cost = 0
# current_hour = 1



class ServerlessCache:
    def __init__(self, max_hourly_cost, total_jobs, prefetching_enabled = True, cache_cacpity_gb=None):
        self.cache = {}
        # self.cache_size_gb = 0
        # self.max_cache_cost_per_hour = max_cache_cost_per_hour
        self.cache_size_over_time = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_jobs = total_jobs
        self.size_of_cache_object_gb = 0.039
        self.cost_per_request = 0.00003125        
        self.cost_per_prefetch = 0.000149217
        self.current_time = 0
        self.max_hourly_cost = max_hourly_cost
        self.current_hour = 0
        self.cache_cacpity_gb = cache_cacpity_gb
        self.prefetching_enabled = prefetching_enabled
        self.total_requests = 0
        if self.max_hourly_cost is not None:
            if self.prefetching_enabled:
                self.requests_allowed_per_hour = self.max_hourly_cost / (self.cost_per_request + self.cost_per_prefetch)
            else:
                self.requests_allowed_per_hour = self.max_hourly_cost / self.cost_per_request
        else:
            self.requests_allowed_per_hour = float('inf')

        self.hourly_request_buffer = 0
        self.prefetch_requests_counter = 0
        self.reset_request_buffer()  # Track cache size at each step
        self.prefetch_next_batch()
          # Track cache size at each step
    def reset_request_buffer(self):
        if self.max_hourly_cost is not None:
            if self.prefetching_enabled:
                self.hourly_request_buffer = self.max_hourly_cost / (self.cost_per_request + self.cost_per_prefetch)
            else:
                self.hourly_request_buffer = self.max_hourly_cost / self.cost_per_request
        else:
            self.hourly_request_buffer = float('inf')
    
    def get_or_insert_batch(self, requested_batch, current_time):

        #check if we have moved into a new hour
        if (current_time // 3600) +1 > self.current_hour:
            self.reset_request_buffer()
            self.current_hour += 1
        if self.hourly_request_buffer <= 0:
            return False #dont record a cache miss because in practice no request will be made to the cache
        else:
            self.hourly_request_buffer -= 1
            self.total_requests += 1
            cache_hit = False
            if requested_batch not in self.cache:
                self.cache_misses += 1
                if  self.cache_cacpity_gb  is not None and self.get_cache_size_gb() >= self.cache_cacpity_gb :
                    # Find the least recently used batch
                    lru_batch = min(self.cache, key=self.cache.get)
                    del self.cache[lru_batch]
                self.cache[requested_batch] = 1
            else:
                self.cache_hits += 1  
                self.cache[requested_batch] += 1

                #preftch the next batch the first time the batch is accessed
                if self.prefetching_enabled and self.cache[requested_batch] == 1:
                    self.prefetch_next_batch()

                if self.cache[requested_batch] >= self.total_jobs:  # Remove if all jobs accessed
                    del self.cache[requested_batch]
                cache_hit = True
        self.cache_size_over_time.append(len(self.cache))
        return cache_hit
    
    def prefetch_next_batch(self):
        #get the max item in the cache and then prefetch the next batch and decrmeet the requests allowed per hour
        if self.prefetching_enabled:
            if self.hourly_request_buffer <= 0:
                return False
            else:
                self.hourly_request_buffer -= 1
                self.total_requests += 1
                self.prefetch_requests_counter += 1
                if len(self.cache) > 0:
                    lru_batch = max(list(self.cache.keys()))
                    next_batch = lru_batch + 1
                    if next_batch not in self.cache:
                        self.cache[next_batch] = 0
                else:
                    next_batch = 1
                    self.cache[next_batch] = 0
                return True

    
    def get_cache_len(self):
        return len(self.cache)
    
    def get_cache_size_gb(self):
        return len(self.cache) *  self.size_of_cache_object_gb
    
    def get_max_num_of_cache_items(self):
        return max(self.cache_size_over_time)
    
    def get_avg_num_of_cache_items(self):
        return sum(self.cache_size_over_time) / len(self.cache_size_over_time)
    
    def get_max_cache_size_gb(self):
        return  max(self.cache_size_over_time) *  self.size_of_cache_object_gb
    
    def get_avg_cache_size_gb(self):
        
        return sum(self.cache_size_over_time) / len(self.cache_size_over_time) *  self.size_of_cache_object_gb
    
    def compute_caching_cost(self):
        total_requests = self.cache_hits + self.cache_misses
        total_cost = total_requests * self.cost_per_request
        return total_cost
    
    def compute_preetching_cost(self):
        return self.prefetch_requests_counter * self.cost_per_prefetch


class CoorDLCache:
    def __init__(self, cache_cacpity_gb, total_jobs):
        self.cache_cacpity_gb = cache_cacpity_gb
        self.cache = {}
        # self.cache_size_gb = 0
        # self.max_cache_cost_per_hour = max_cache_cost_per_hour
        self.current_cache_hourly_cost = 0
        self.cache_size_over_time = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_jobs = total_jobs
        self.size_of_cache_object_gb = 0.039
        self.current_time = 0
          # Track cache size at each step
    
    def get_or_insert_batch(self, requested_batch, current_time):
        
        cache_hit = False
        if requested_batch not in self.cache:
            self.cache_misses += 1
            if self.cache_cacpity_gb is not None and self.get_cache_size_gb() >= self.cache_cacpity_gb:
                # Find the least recently used batch
                lru_batch = min(self.cache, key=self.cache.get)
                del self.cache[lru_batch]
            self.cache[requested_batch] = 1
    
        else:
            self.cache_hits += 1  
            self.cache[requested_batch] += 1
            if self.cache[requested_batch] >= self.total_jobs:  # Remove if all jobs accessed
                del self.cache[requested_batch]
            cache_hit = True

        self.cache_size_over_time.append(len(self.cache))
        return cache_hit
    
    def get_cache_len(self):
        return len(self.cache)
    
    def get_cache_size_gb(self):
        return len(self.cache) *  self.size_of_cache_object_gb
    
    def get_max_num_of_cache_items(self):
        return max(self.cache_size_over_time)
    
    def get_avg_num_of_cache_items(self):
        return sum(self.cache_size_over_time) / len(self.cache_size_over_time)
    
    def get_max_cache_size_gb(self):
        return  max(self.cache_size_over_time) *  self.size_of_cache_object_gb
    
    def get_avg_cache_size_gb(self):
        
        return sum(self.cache_size_over_time) / len(self.cache_size_over_time) *  self.size_of_cache_object_gb
    
    def compute_cost(self, cache_instnace, total_durtion_seconds, throughput_per_s):
        if cache_instnace in asws_redis_instnces:
            hourly_cost = asws_redis_instnces[cache_instnace]['price_per_hour']
            total_cost = hourly_cost * total_durtion_seconds / 3600
            return total_cost
        else: #severless
            # if self.get_cache_len() == 0:
            #     return 0
            # Duration is in seconds
            # Memory size is in GB
            # Cost is in USD
            hours_in_a_month = 730
            seconds_in_a_month = 2628000
            # average_cache_size_gb = self.get_max_cache_size_gb()
            average_cache_size_gb = self.get_avg_cache_size_gb()

            # round_duartion_tonearest_hour = total_durtion_seconds / 3600
            # rounded_duarion = round_duartion_tonearest_hour * 3600
            data_storage_cost_monthly = average_cache_size_gb * hours_in_a_month * 0.125

            requests = throughput_per_s * seconds_in_a_month * (self.size_of_cache_object_gb * 1024 * 1024) #gb to kb
            ecpu_monthly_cost = requests * 0.0000000034

            total_monhtly_cost = data_storage_cost_monthly + ecpu_monthly_cost
            total_monhtly_cost  = data_storage_cost_monthly
            exp_cost = total_monhtly_cost/seconds_in_a_month * total_durtion_seconds
            
            return exp_cost

    


class Job:
    def __init__(self, job_id, time_per_batch, batches_per_epoch, total_epochs, total_batches_to_process, cache:CoorDLCache):
        self.job_id = job_id
        self.time_per_batch = time_per_batch  
        self.batches_to_process = list(range(1, total_batches_to_process + 1))
        self.batches_processed = 0
        self.epochs_processed = 0
        self.next_available_time = 0 
        self.wait_time = 0  
        self.epoch_completion_time = 0  
        self.reached_end_of_epoch = False
        self.epoch_delays: Dict[int, float] = {}
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.cahe_hits = 0
        self.cache_misses = 0
        self.cache = cache
        self.batch_size = 128
        self.cache_miss_penalty = 0.5 # Cache miss penalty
        self.current_hour = 1
    def compute_final_metrics(self):
        line = {}
        line['job_id'] = self.job_id
        line['epochs_processed'] = self.epochs_processed
        line['batches_processed'] = self.batches_processed
        line['samples_processed'] = self.batches_processed * self.batch_size
        line['time_per_batch'] = self.time_per_batch
        line['potential_time(sec)'] = self.batches_processed * self.time_per_batch
        line['potential_time(hour)'] = self.batches_processed * self.time_per_batch / 3600 # 1 hour = 3600 seconds
        line['actual_time(sec)'] = self.next_available_time
        line['actual_time(hour)'] = self.next_available_time / 3600 # 1 hour = 3600 seconds
        line['potential_throughput(batches/sec)'] = self.batches_processed / (self.batches_processed * self.time_per_batch)
        line['actual_throughput(batches/sec)'] = self.batches_processed / self.next_available_time
        line['potential_throughput(samples/sec)'] = self.batches_processed / (self.batches_processed * self.time_per_batch) *  self.batch_size
        line['actual_throughput(samples/sec)'] = self.batches_processed / self.next_available_time * self.batch_size
        line['total_delay'] = sum(delay for delay in self.epoch_delays.values())
        return line

    def __lt__(self, other):
        return self.next_available_time < other.next_available_time  

    def process_next_batch(self):
        if not self.batches_to_process:
            return False  # No more batches to process, prevent popping an empty list
        
        
        if (self.next_available_time // 3600) +1 > self.current_hour:
            self.current_hour += 1
            logging.info(f"cachesize after {self.current_hour} hours: {self.cache.get_cache_size_gb()}")


        next_batch = self.batches_to_process.pop(0)
        self.batches_processed += 1
        is_cache_hit = self.cache.get_or_insert_batch(next_batch, self.next_available_time)
        if is_cache_hit:
            self.cahe_hits += 1
            self.next_available_time += self.time_per_batch  
        else:
            self.cache_misses += 1
            # if self.cache.coordl_mode:
            #     cache_miss_penalty = 0.5 # Cache miss penalty
            # else:
            #     cache_miss_penalty = 0
            self.next_available_time += self.time_per_batch + self.cache_miss_penalty  #+ 0.1  # Add cache miss penalty

        # logging.debug(f"[Time {self.next_available_time:.2f}] Job {self.job_id} processed batch {next_batch}")

        # Epoch tracking
        if self.batches_processed % self.batches_per_epoch == 0:
            self.reached_end_of_epoch = True
            self.epoch_completion_time = self.next_available_time
            self.epochs_processed += 1
            # logging.info(f"Job {self.job_id} finished epoch {self.batches_processed // batches_per_epoch} at time {self.epoch_completion_time}. Throughput: {self.batches_processed / self.epoch_completion_time:.2f} batches per second")
            # logging.info(f"cachesize after epoch {self.epochs_processed}: {self.cache.get_cache_len()}")



def save_dict_list_to_csv(dict_list, output_file):
    if not dict_list:
        print("No data to save.")
        return
    headers = dict_list[0].keys()
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
        if not file_exists:
            writer.writeheader()
        for data in dict_list:
            writer.writerow(data)

# def compute_serverless_redis_costs(total_durtion_seconds, cache_size_gb, throughput_per_s, avg_size_per_request_kb):
#     # Duration is in seconds
#     # Memory size is in GB
#     # Cost is in USD
#     hours_in_a_month = 730
#     seconds_in_a_month = 2628000
#     # round_duartion_tonearest_hour = total_durtion_seconds / 3600
#     # rounded_duarion = round_duartion_tonearest_hour * 3600
#     data_storage_cost_monthly = cache_size_gb * hours_in_a_month * 0.125

#     requests = throughput_per_s * seconds_in_a_month * avg_size_per_request_kb
#     ecpu_monthly_cost = requests * 0.0000000034

#     total_monhtly_cost = data_storage_cost_monthly + ecpu_monthly_cost

#     exp_cost = total_monhtly_cost/seconds_in_a_month * total_durtion_seconds
    
#     return exp_cost



def run(config):
    coordl_mode = config['coordl_mode']
    batches_per_epoch = config['batches_per_epoch']
    total_epochs = config['total_epochs']
    total_batches = config['total_batches']
    num_jobs = len(config['job_speeds'])
    max_cache_size_gb = config['max_cache_size_gb']
    max_cache_cost_per_hour = config['max_cache_cost_per_hour']
    prefetching_enabled = config['prefetching_enabled']
    if coordl_mode:
        cache = CoorDLCache(max_cache_size_gb, num_jobs)
    else:
        cache = ServerlessCache(max_cache_cost_per_hour, num_jobs, prefetching_enabled)

    jobs = [Job(job_id,
                time_per_batch, 
                batches_per_epoch, 
                total_epochs,
                total_batches,cache) for job_id, time_per_batch in enumerate(config['job_speeds'])] 
    

    job_queue: List[Job] = []
    for job in jobs:
        heapq.heappush(job_queue, job)

    training_finished = False
    while not training_finished:
        training_finished = True  # Assume training is done unless proven otherwise

        if job_queue:
            training_finished = False  # There are jobs still running
            job = heapq.heappop(job_queue)  # Get the job with the earliest available time

            if len(job.batches_to_process) == 0:
                logging.info(f"Job {job.job_id} has no more batches to process.")
                continue

            job.process_next_batch()

            if job.reached_end_of_epoch and coordl_mode:
                all_jobs_finished_epoch = all(j.reached_end_of_epoch for j in jobs)

                if all_jobs_finished_epoch:
                    last_job_to_finish_epoch_time = max(j.epoch_completion_time for j in jobs)
                    #find the job that was the last to finish the epoch
                    logging.debug(f"All jobs finished epoch {job.batches_processed // batches_per_epoch}")

                    for other_job in jobs:
                        other_job.reached_end_of_epoch = False
                        other_job.next_available_time = last_job_to_finish_epoch_time + other_job.time_per_batch
                        other_job.epoch_delays[other_job.epochs_processed] = last_job_to_finish_epoch_time - other_job.epoch_completion_time
                        other_job.epoch_completion_time = 0
                        logging.debug(f"Job {other_job.job_id} delayed by {other_job.epoch_delays[other_job.epochs_processed]:.2f} seconds in epoch {other_job.epochs_processed}")

                    for job in jobs:
                        if len(job.batches_to_process) > 0:
                            heapq.heappush(job_queue, job)
            else:
                heapq.heappush(job_queue, job)

    for job in jobs:
        total_delay = sum(delay for delay in job.epoch_delays.values())
        logging.info(f"Total delay for Job {job.job_id} due to coordl policy: {total_delay:.2f} seconds")
    
    for job in jobs:
        throughput = job.batches_processed / job.next_available_time
        logging.info(f"Throughput for Job {job.job_id}: {throughput:.2f} batches per second")

    logging.info(f"max_cache_size_gb: {cache.get_max_num_of_cache_items()}")
    logging.info(f"max_cache_size_gb: {cache.get_max_cache_size_gb()}")

    # line['job_id'] = self.job_id
    # line['epochs_processed'] = self.epochs_processed
    # line['batches_processed'] = self.batches_processed
    # line['samples_processed'] = self.batches_processed * self.batch_size
    # line['time_per_batch'] = self.time_per_batch
    # line['potential_time(sec)'] = self.batches_processed * self.time_per_batch
    # line['potential_time(hour)'] = self.batches_processed * self.time_per_batch / 3600
    # line['actual_time(sec)'] = self.next_available_time
    # line['actual_time(hour)'] = self.next_available_time / 3600
    # line['potential_throughput(batches/sec)'] = self.batches_processed / (self.batches_processed * self.time_per_batch)
    # line['actual_throughput batches/sec)'] = self.batches_processed / self.next_available_time
    # line['potential_throughput(samples/sec)'] = self.batches_processed / (self.batches_processed * self.time_per_batch) *  self.batch_size
    # line['actual_throughput(samples/sec)'] = self.batches_processed / self.next_available_time * self.batch_size
    # line['total_delay'] = sum(delay for delay in self.epoch_delays.values())
    # Compute final metrics
    final_metrics = [job.compute_final_metrics() for job in jobs]
    save_dict_list_to_csv(final_metrics, "sim_final_metrics.csv")
    summary = {}
    summary['dataset_size(num_batches)'] = total_batches
    summary['total_jobs'] = len(jobs)
    summary['epochs_per_job'] = total_epochs
    summary['total_epochs'] = sum(job['epochs_processed'] for job in final_metrics)
    summary['total_batches_processed'] = sum(job['batches_processed'] for job in final_metrics)
    summary['total_samples_processed'] = sum(job['samples_processed'] for job in final_metrics)
    summary['toal_time(sec)'] = sum(job['actual_time(sec)'] for job in final_metrics) / len(final_metrics)
    summary['total_time(hours)'] =sum(job['actual_time(hour)'] for job in final_metrics) / len(final_metrics)
    summary['total_throughput(samples/sec)'] = sum(job['actual_throughput(samples/sec)'] for job in final_metrics)
    summary['total_throughput(bacthes/sec)'] = sum(job['actual_throughput(batches/sec)'] for job in final_metrics)
    summary['potential_throughput(samples/sec)'] = sum(job['potential_throughput(samples/sec)'] for job in final_metrics)
    summary['potential_throughput(bacthes/sec)'] = sum(job['potential_throughput(batches/sec)'] for job in final_metrics) 

    # summary['total_throughput(samples/sec)'] = summary['total_samples_processed']/ summary['toal_time(sec)']
    # summary['total_throughput(bacthes/sec)'] = summary['total_batches_processed']/ summary['toal_time(sec)']
    summary['compute_cost'] = summary['total_time(hours)'] * 12.24  # $0.1 per hour
    summary['max_number_of_cached_bacthes'] = cache.get_max_num_of_cache_items()
    summary['max_cache_size_gb'] = cache.get_max_cache_size_gb()
    summary['avg_number_of_cached_bacthes'] = cache.get_avg_num_of_cache_items()
    summary['avg_cache_size_gb'] = cache.get_avg_cache_size_gb()
    summary['cache_hits'] = cache.cache_hits
    summary['cache_misses'] = cache.cache_misses

    if coordl_mode:
        summary['cache_cost']= cache.compute_cost(
            config['redis_instance'],
            summary['toal_time(sec)'], 
            summary['total_throughput(bacthes/sec)'])
        summary['prefetching_cost'] = 0
    else:
        summary['cache_cost'] = cache.compute_caching_cost()
        summary['prefetching_cost'] = cache.compute_preetching_cost()
    summary['cache+prefetch_cost'] = summary['cache_cost'] + summary['prefetching_cost']
    # summary['total_requests'] = cache.total_requests
    summary['total_cost'] = summary['compute_cost'] + summary['cache_cost'] + summary['prefetching_cost']
    save_dict_list_to_csv([summary], "sim_final_summary_metrics.csv")


def run_increasing_job_sizes_sim(config):
    # bacthes_per_epoch = [20000,40000,60000,80000,100000]
    bacthes_per_epoch = [500000]
    job_speeds = [0.107508104, 0.339788711, 0.089724519, 0.507133094]  # ResNet-18, ResNet-50, imagenet_shufflenet_v2_x1_0, imagenet_vgg16
    job_speeds = [0.107508104, 0.507133094]  # ResNet-18, ResNet-50, imagenet_shufflenet_v2_x1_0, imagenet_vgg16

    # job_speeds = [0.107508104]  # ResNet-18, ResNet-50, imagenet_shufflenet_v2_x1_0, imagenet_vgg16

    for bach_per_epoch in bacthes_per_epoch:
        #compute_total_data_size
        config['batches_per_epoch'] = bach_per_epoch
        config['total_batches'] = config['batches_per_epoch'] * config['total_epochs']
        config['job_speeds'] = job_speeds
        run(config)


if __name__ == "__main__":
    np.random.seed(42)
    cache_instnace = 'cache.r5.24xlarge' #cache.r7g.8xlarge, cache.r5.24xlarge, cache.r7g.4xlarge
    config :Dict = {
        'coordl_mode': False,
        'batches_per_epoch': 10000,
        'total_epochs': 1,
        'job_speeds': np.random.uniform(0.1, 1.0, 10).tolist(),
        'redis_instance': None,
        'max_cache_size_gb': None, #asws_redis_instnces[cache_instnace]['memory'], #80 = $10 per hour, 400 = $50 per hour, 800 = $100 per hour
        'max_cache_cost_per_hour': None, #asws_redis_instnces[cache_instnace]['price_per_hour'],
        'prefetching_enabled': True
    }
    config['total_batches'] = config['batches_per_epoch'] * config['total_epochs']

    if  os.path.isfile("sim_final_metrics.csv"):
        os.remove("sim_final_metrics.csv")
    if  os.path.isfile("sim_final_summary_metrics.csv"):
        os.remove("sim_final_summary_metrics.csv")

    
    run_increasing_job_sizes_sim(config)
    
    # job_speeds = np.random.uniform(0.1, 1.0, 10).tolist()
    # for i in range(len(job_speeds)):
    #     #process the ith many jobs 
    #     jobs = job_speeds[:i+1]
    #     config['job_speeds'] = jobs
    #     config['max_cache_size_gb'] = max_cache_size_gb
    #     logging.info(f"Running simulation with {i+1} jobs")
    #     run(config)
    
    #bacthes_per_epoch = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000] 

    # for bach_per_epoch in bacthes_per_epoch:
    #     #compute_total_data_size
    #     toal_data_size_gb = bach_per_epoch * 0.039 #gb
    #     #give cache % of the data size 30% = 0.3, 60% = 0.6, 20% = 0.2
    #     max_cache_size_gb = toal_data_size_gb * 1.0
    #     print(f"total_data_size_gb: {toal_data_size_gb}, max_cache_size_gb: {max_cache_size_gb}")
    #     config['max_cache_size_gb'] = max_cache_size_gb
    #     config['batches_per_epoch'] = bach_per_epoch
    #     config['total_batches'] = config['batches_per_epoch'] * config['total_epochs']
    #     config['job_speeds'] = job_speeds
    #     run(config)

    # job_speeds = np.random.uniform(0.1, 1.0, 20).tolist()
    # for i in range(20):
    #     #process the ith many jobs 
    #     jobs = job_speeds[:i+1]
    #     config['job_speeds'] = jobs
    #     logging.info(f"Running simulation with {i+1} jobs")
    #     run(config)
    # job_speeds = [0.107508104, 0.339788711, 0.089724519, 0.507133094]  # ResNet-18, ResNet-50, imagenet_shufflenet_v2_x1_0, imagenet_vgg16
    # config['job_speeds'] = job_speeds
    # run(config)
