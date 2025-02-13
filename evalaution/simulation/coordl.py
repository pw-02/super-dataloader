import heapq
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import csv
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler("simulation.log", mode="w")  # Save to file
    ]
)

cached_batches = {}
cache_size_over_time = []  # Track cache size at each step
time_steps = []  # Track corresponding time values
cache_budget_per_hour = 25 #dollars
current_cache_hourly_cost = 0
current_hour = 1
class Job:
    def __init__(self, job_id, time_per_batch, batches_per_epoch, total_epochs, total_batches):
        self.job_id = job_id
        self.time_per_batch = time_per_batch  
        self.batches_to_process = list(range(1, total_batches + 1))
        self.batches_processed = 0
        self.next_available_time = 0 
        self.wait_time = 0  
        self.epoch_completion_time = 0  
        self.reached_end_of_epoch = False
        self.epoch_delays: Dict[int, float] = {}
        self.current_epoch = 0
        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.cahe_hits = 0
        self.cache_misses = 0
    
    
    def compute_final_metrics(self):
        line = {}
        line['job_id'] = self.job_id
        line['total_batches'] = self.total_batches
        line['total_epochs'] = self.total_epochs
        line['time_per_batch'] = self.time_per_batch
        line['batches_processed'] = self.batches_processed
        line['potential_throughput (batches/sec)'] = self.total_batches / (self.total_batches * self.time_per_batch)
        line['actual_throughput (batches/sec)'] = self.batches_processed / self.next_available_time
        line['potential_throughput (samples/sec)'] = self.total_batches / (self.total_batches * self.time_per_batch) * 128
        line['actual_throughput (samples/sec)'] = self.batches_processed / self.next_available_time * 128
        line['potential_time (s)'] = self.total_batches * self.time_per_batch
        line['actual_time (s)'] = self.next_available_time
        line['total_delay'] = sum(delay for delay in self.epoch_delays.values())
        return line

        # total_delay = sum(delay for delay in self.epoch_delays.values())
        # throughput = self.batches_processed / self.next_available_time
        # logging.info(f"Total delay for Job {self.job_id}: {total_delay:.2f} seconds")
        # logging.info(f"Throughput for Job {self.job_id}: {throughput:.2f} batches per second")

    def __lt__(self, other):
        return self.next_available_time < other.next_available_time  

    def process_next_batch(self, num_jobs, coordl_mode=False, max_cache_size_gb=None):
        if not self.batches_to_process:
            return False  # No more batches to process, prevent popping an empty list

        next_batch = self.batches_to_process.pop(0)
        self.batches_processed += 1
        # logging.debug(f"[Time {self.next_available_time:.2f}] Job {self.job_id} processed batch {next_batch}")

        if next_batch in cached_batches:
            self.cahe_hits += 1
            cached_batches[next_batch] += 1
            if cached_batches[next_batch] >= num_jobs:  # Remove if all jobs accessed
                del cached_batches[next_batch]
            self.next_available_time += self.time_per_batch  
        else:
            self.cache_misses += 1
            if max_cache_size_gb is not None and len(cached_batches) *  0.039 >= max_cache_size_gb:
                # Find the least recently used batch
                lru_batch = min(cached_batches, key=cached_batches.get)
                del cached_batches[lru_batch]

            cached_batches[next_batch] = 1
            if coordl_mode:
                cache_miss_penalty = 0.5 # Cache miss penalty
            else:
                cache_miss_penalty = 0
            self.next_available_time += self.time_per_batch + cache_miss_penalty  #+ 0.1  # Add cache miss penalty


        # Epoch tracking
        if self.batches_processed % self.batches_per_epoch == 0:
            self.reached_end_of_epoch = True
            self.epoch_completion_time = self.next_available_time
            self.current_epoch += 1
            # logging.info(f"Job {self.job_id} finished epoch {self.batches_processed // batches_per_epoch} at time {self.epoch_completion_time}. Throughput: {self.batches_processed / self.epoch_completion_time:.2f} batches per second")
            logging.info(f"cachesize after epoch {self.current_epoch}: {len(cached_batches)}")
        # Track cache size over time
        time_steps.append(self.next_available_time)
        cache_size_over_time.append(len(cached_batches))

    def log_epoch_delay(self, last_epoch_time):
        delay = last_epoch_time - self.epoch_completion_time
        self.epoch_delays[self.current_epoch] = delay
        self.next_available_time = last_epoch_time + self.time_per_batch
        # logging.info(f"Job {self.job_id} delayed by {delay:.2f} seconds in epoch {self.current_epoch}")

def create_jobs(job_speeds: List[float],total_batches ) -> List[Job]:
    return [Job(job_id, time_per_batch, total_batches) for job_id, time_per_batch in enumerate(job_speeds)]

import os
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

def compute_serverless_redis_costs(total_durtion_seconds, cache_size_gb, throughput_per_s, avg_size_per_request_kb):
    # Duration is in seconds
    # Memory size is in GB
    # Cost is in USD
    hours_in_a_month = 730
    seconds_in_a_month = 2628000
    # round_duartion_tonearest_hour = total_durtion_seconds / 3600
    # rounded_duarion = round_duartion_tonearest_hour * 3600
    data_storage_cost_monthly = cache_size_gb * hours_in_a_month * 0.125

    requests = throughput_per_s * seconds_in_a_month * avg_size_per_request_kb
    ecpu_monthly_cost = requests * 0.0000000034

    total_monhtly_cost = data_storage_cost_monthly + ecpu_monthly_cost

    exp_cost = total_monhtly_cost/seconds_in_a_month * total_durtion_seconds
    
    return exp_cost



def run(config):
    cached_batches.clear()
    cache_size_over_time.clear()
    time_steps.clear()
    coordl_mode = config['coordl_mode']
    total_epochs = config['total_epochs']
    batches_per_epoch = config['batches_per_epoch']
    total_batches = config['total_batches']
    jobs = [Job(job_id,  time_per_batch, batches_per_epoch, total_epochs, total_batches,) for job_id, time_per_batch in enumerate(config['job_speeds'])] 
    max_cache_size_gb = config['max_cache_size_gb']

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

            job.process_next_batch(num_jobs=len(jobs), coordl_mode=coordl_mode, max_cache_size_gb=max_cache_size_gb)

            if job.reached_end_of_epoch and coordl_mode:
                all_jobs_finished_epoch = all(j.reached_end_of_epoch for j in jobs)

                if all_jobs_finished_epoch:
                    last_job_to_finish_epoch_time = max(j.epoch_completion_time for j in jobs)
                    #find the job that was the last to finish the epoch
                    logging.debug(f"All jobs finished epoch {job.batches_processed // batches_per_epoch}")

                    for other_job in jobs:
                        other_job.reached_end_of_epoch = False
                        other_job.next_available_time = last_job_to_finish_epoch_time + other_job.time_per_batch
                        other_job.epoch_delays[other_job.current_epoch] = last_job_to_finish_epoch_time - other_job.epoch_completion_time
                        other_job.epoch_completion_time = 0
                        logging.debug(f"Job {other_job.job_id} delayed by {other_job.epoch_delays[other_job.current_epoch]:.2f} seconds in epoch {other_job.current_epoch}")

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

    logging.info(f"Max cache size: {max(cache_size_over_time)}")

    # Compute final metrics
    final_metrics = [job.compute_final_metrics() for job in jobs]
    save_dict_list_to_csv(final_metrics, "sim_final_metrics.csv")
    summary = {}
    summary['dataset_size(num_bacthes)'] = total_batches
    summary['total_jobs'] = len(jobs)
    summary['epochs_per_job'] = total_epochs
    summary['total_epochs'] = total_epochs * len(final_metrics)
    summary['total_batches_processed'] = sum(job['total_batches'] for job in final_metrics)
    summary['total_samples_processed'] = sum(job['total_batches'] * 256 for job in final_metrics)
    summary['toal_time(sec)'] = sum(job['actual_time (s)'] for job in final_metrics) / len(final_metrics)
    summary['total_time(hours)'] = summary['toal_time(sec)'] / 3600
    summary['total_throughput(samples/sec)'] = sum(job['actual_throughput (samples/sec)'] for job in final_metrics)
    summary['total_throughput(bacthes/sec)'] = sum(job['actual_throughput (batches/sec)'] for job in final_metrics) / 128
    summary['potential_throughput(samples/sec)'] = sum(job['potential_throughput (samples/sec)'] for job in final_metrics)
    summary['potential_throughput(bacthes/sec)'] = sum(job['potential_throughput (batches/sec)'] for job in final_metrics) / 128

    # summary['total_throughput(samples/sec)'] = summary['total_samples_processed']/ summary['toal_time(sec)']
    # summary['total_throughput(bacthes/sec)'] = summary['total_batches_processed']/ summary['toal_time(sec)']
    summary['compute_cost'] = summary['total_time(hours)'] * 12.24  # $0.1 per hour
    summary['max_number_of_cached_bacthes'] = max(cache_size_over_time)
    summary['max_cache_size_gb'] = summary['max_number_of_cached_bacthes'] * 0.039 #gb
    summary['avg_number_of_cached_bacthes'] = sum(cache_size_over_time) / len(cache_size_over_time)
    summary['avg_cache_size_gb'] = summary['avg_number_of_cached_bacthes'] * 0.039 #gb
    summary['cache_hits'] = sum(job.cahe_hits for job in jobs)
    summary['cache_misses'] = sum(job.cache_misses for job in jobs)
    
    if coordl_mode:
        summary['cache_cost']= compute_serverless_redis_costs(
            summary['toal_time(sec)'], 
            summary['avg_cache_size_gb'],
            summary['total_throughput(bacthes/sec)'], 
            40894.464)
    else:
        summary['cache_cost'] = summary['total_batches_processed'] * 0.0003125
    summary['total_cost'] = summary['compute_cost'] + summary['cache_cost']
    save_dict_list_to_csv([summary], "sim_final_summary_metrics.csv")



def run_increasing_job_sizes_sim(config):
    
    bacthes_per_epoch = [20000,40000,60000,80000,100000]
    job_speeds = [0.107508104, 0.339788711, 0.089724519, 0.507133094]  # ResNet-18, ResNet-50, imagenet_shufflenet_v2_x1_0, imagenet_vgg16
    for bach_per_epoch in bacthes_per_epoch:
        #compute_total_data_size
        toal_data_size_gb = bach_per_epoch * 0.039 #gb
        #give cache % of the data size 30% = 0.3, 60% = 0.6, 20% = 0.2
        max_cache_size_gb = toal_data_size_gb * 1.0
        print(f"total_data_size_gb: {toal_data_size_gb}, max_cache_size_gb: {max_cache_size_gb}")
        config['max_cache_size_gb'] = max_cache_size_gb
        config['batches_per_epoch'] = bach_per_epoch
        config['total_batches'] = config['batches_per_epoch'] * config['total_epochs']
        config['job_speeds'] = job_speeds
        run(config)


if __name__ == "__main__":
    np.random.seed(42)

    configs = {'config1': {'durtion': None, 'coordl_mode': True, 'batches_per_epoch': 9000, 'total_epochs': 1}}
    config = configs['config1']
    config['total_batches'] = config['batches_per_epoch'] * config['total_epochs']
    if  os.path.isfile("sim_final_metrics.csv"):
        os.remove("sim_final_metrics.csv")
    if  os.path.isfile("sim_final_summary_metrics.csv"):
        os.remove("sim_final_summary_metrics.csv")
    max_cache_size_gb = None
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
