import heapq
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import csv
import math
import os
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler("simulation.log", mode="w")  # Save to file
    ]
)

worklaods ={
    "cifar10": {"num_files": 50000, "batch_size": 128},
    "imagenet1k": {"num_files": 1281167, "batch_size": 256}
}
cost_per_serverless_cache_request = 0.0003125
# job_speeds = [0.0014, 0.002, 0.0033, 0.005]  # ResNet-18, ResNet-50, ResNet-101, ResNet-152
job_speeds = [0.26, 0.43, 0.83, 0.95]  # ResNet-18, ResNet-50, ResNet-101, ResNet-152

cached_batches = {}
cache_size_over_time = []  # Track cache size at each step
time_steps = []  # Track corresponding time values
cached_bacth_size_gb =  0.039 #GB
# Constants
dataset = "imagenet1k"
sim_duarion_hours = None # if none then sim will run for the allocated number of epochs
batches_per_epoch = math.ceil(worklaods[dataset]["num_files"] / worklaods[dataset]["batch_size"])
if sim_duarion_hours:
    fastest_job = min(job_speeds)
    total_epochs =  math.ceil(sim_duarion_hours * 3600 / (batches_per_epoch * fastest_job))
else:
    total_epochs = 100
total_batches = batches_per_epoch * total_epochs
pass
# job_speeds = [0.125,0.15,0.3,1.6,2.5]  # Time per batch (in seconds)


class Batch:
    def __init__(self, batch_id):
        self.batch_id = batch_id
        self.request_count = 0
        self.last_accessed_time = 0

class Cache:
    def __init__(self):
        self.cached_batches:Dict[int, Batch] = {}
        self.request_count = 0
        self.cost_per_request = 0.0003125
    
    def add_batch(self, batch_id,current_time):
        if batch_id in self.cached_batches:
            self.cached_batches[batch_id].request_count += 1
            self.cached_batches[batch_id].last_accessed_time = current_time
        else:
            self.cached_batches[batch_id] = 1
        self.request_count += 1



class Job:
    def __init__(self, job_id, time_per_batch, total_batches):
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
    
    def compute_final_metrics(self):
        line = {}
        line['job_id'] = self.job_id
        line['total_batches'] = total_batches
        line['total_epochs'] = total_epochs
        line['time_per_batch'] = self.time_per_batch
        line['batches_processed'] = self.batches_processed
        line['potential_throughput'] = total_batches / (total_batches * self.time_per_batch)
        line['actual_throughput'] = self.batches_processed / self.next_available_time
        line['potential_time'] = total_batches * self.time_per_batch
        line['actual_time'] = self.next_available_time
        line['total_delay'] = sum(delay for delay in self.epoch_delays.values())
        return line

        # total_delay = sum(delay for delay in self.epoch_delays.values())
        # throughput = self.batches_processed / self.next_available_time
        # logging.info(f"Total delay for Job {self.job_id}: {total_delay:.2f} seconds")
        # logging.info(f"Throughput for Job {self.job_id}: {throughput:.2f} batches per second")

    def __lt__(self, other):
        return self.next_available_time < other.next_available_time  

    def process_next_batch(self):
        if not self.batches_to_process:
            return False  # No more batches to process, prevent popping an empty list

        next_batch = self.batches_to_process.pop(0)
        self.batches_processed += 1
        self.next_available_time += self.time_per_batch  
        logging.debug(f"[Time {self.next_available_time:.2f}] Job {self.job_id} processed batch {next_batch}")
        # Cache management
        if next_batch in cached_batches:
            cached_batches[next_batch] += 1
            if cached_batches[next_batch] >= 2:  # Remove if both jobs accessed
                del cached_batches[next_batch]
        else:
            cached_batches[next_batch] = 1

        # Epoch tracking
        if self.batches_processed % batches_per_epoch == 0:
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

def create_jobs(job_speeds: List[float]) -> List[Job]:
    return [Job(job_id, time_per_batch, total_batches) for job_id, time_per_batch in enumerate(job_speeds)]


def save_dict_list_to_csv(dict_list, output_file):
    if not dict_list:
        print("No data to save.")
        return
    headers = dict_list[0].keys()
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter='\t')
        writer.writeheader()
        for data in dict_list:

            writer.writerow(data)
def write_log_line_csv(line):
    headers = line.keys()
    #check if file exists, if so append to it else create a new file
    file_exists = os.path.isfile("sim_report.csv")

    with open("sim_report.csv", 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
        if not file_exists:
            writer.writeheader()
        writer.writerow(line)

def run(coordl_mode=False):
    minute = 1
    jobs = create_jobs(job_speeds)
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

            #print every min to track progress
            if job.next_available_time/60 >= minute:
                write_log_line_csv({
                    "elasped_time(seconds)": job.next_available_time,
                    "elasped_time(minutes)": job.next_available_time / 60,
                    "elasped_time(hours)": job.next_available_time / 3600, 
                    "total_cached_batches": len(cached_batches),
                    "toal_cached_data_db": len(cached_batches) * cached_bacth_size_gb,
                    "total_cache_requets": sum(j.batches_processed for j in jobs),
                    "severless_cache_cost" : (cost_per_serverless_cache_request) * sum(j.batches_processed for j in jobs)})
                minute += 1
            if sim_duarion_hours and job.next_available_time > sim_duarion_hours * 3600:
                #break fron the loop if the sim duration is reached
                break

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



if __name__ == "__main__":
    coordl_mode = False
    run(coordl_mode)
