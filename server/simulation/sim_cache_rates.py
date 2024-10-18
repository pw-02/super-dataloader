import math

cache_intances = {
    'cache.m7g.8xlarge': 103.68,
    'cache.m7g.12xlarge': 157.12,
}

def compute_ec2_costs(instance_type: str, time_seconds: float):
    
    instance_prices = {
        't2.medium':  0.0464,
        'p3.8xlarge':  12.24,
        'c5n.xlarge': 0.4,
        'cache.t3.medium':  0.068,
        'cache.m7g.4xlarge': 1.257,
        'cache.m7g.xlarge': 0.315,
        'cache.m5.2xlarge': 0.4984,
        'cache.m7g.8xlarge': 2.0112,
        'cache.m7g.12xlarge': 3.016,
    }
    hours = time_seconds / 3600
    hourly_rate = instance_prices[instance_type]
    instance_cost = hourly_rate * hours
    return instance_cost
COCO = {
    'redis_cache': 'cache.m7g.8xlarge',
    'shade_batch_time_on_miss': 0.3,
    'shade_batch_time_on_hit': 0.00513509399752365,
    'shade_first_epoch_time': 986, #19612,
    # 'shade_wss': 1.0,
    'shade_repitation_factor': 2,
    'super_batch_time_on_miss': 0.2,
    'super_batch_time_on_hit': 0.1,
    'coordl_batch_time_on_miss': 0.2,
    'coordl_batch_time_on_hit': 0.001,
    # 'cordl_first_epoch_time': 986,
    'batch_size': 64,
    'batches_per_epoch': 1293,  # Corrected key name here
    'total_epochs': 2,
    'total_files': 82752,
    'total_dataset_size': 12.6,
    'gpu_time': 0.58,
    'num_jobs': 4,
    'cache_size_as_percentage_of_dataset': 1.0
}

IMAGE_NET = {
    'redis_cache': 'cache.m7g.8xlarge',
    'shade_batch_time_on_miss': 2.1,
    'shade_batch_time_on_hit': 0.0513509399752365,
    'shade_first_epoch_time': 7589, #19612,
    'shade_wss': 0.50,
    'shade_repitation_factor': 1.5,
    'super_batch_time_on_miss': 0.3,
    'super_batch_time_on_hit': 0.048,
    'coordl_batch_time_on_miss': 0.65,
    'coordl_batch_time_on_hit': 0.04812693,
    # 'cordl_first_epoch_time': 7589,
    'batch_size': 128,
    'batches_per_epoch': 8565,  # Corrected key name here
    'total_epochs': 80,
    'total_files': 1096302,
    'total_dataset_size': 155.3,
    'gpu_time': 0.341211255,
    'num_jobs': 20,
    'cache_size_as_percentage_of_dataset': 0.50
}



IMAGE_NET['shade_wss'] = IMAGE_NET['cache_size_as_percentage_of_dataset']

def compute_total_batches_cached(cache_hit_rate, workload):
    total_files_that_can_be_cached = workload['total_files'] * cache_hit_rate
    total_batches_cached = total_files_that_can_be_cached // workload['batch_size']

    # total_batches_cached = total_batches_cached // workload['batch_size']
    total_batches_missed = workload['batches_per_epoch'] - total_batches_cached
   
    return total_batches_cached, total_batches_missed


def simulate_training_job(workload=IMAGE_NET):
    # First epoch assume cold cache so total training time is the sum of the time to fetch each batch from the dataset on a cache miss
    super_total_time = 0
    shade_total_time = 0
    coordl_total_time = 0
    total_epochs = workload['total_epochs'] * workload['num_jobs']
    workload['shade_wss'] = workload['cache_size_as_percentage_of_dataset']

    # # Calculate initial times based on cache misses
    # shade_total_time += workload['shade_batch_time_on_miss'] * workload['batches_per_epoch'] + workload['gpu_time'] * workload['batches_per_epoch']
    # coordl_total_time += workload['coordl_batch_time_on_miss'] * workload['batches_per_epoch'] + workload['gpu_time'] * workload['batches_per_epoch']

    # Calculate cache hit rate and batches
    if workload['cache_size_as_percentage_of_dataset']:
        cache_size = workload['total_dataset_size'] * workload['cache_size_as_percentage_of_dataset']
    else:
        cache_size =  cache_intances[workload['redis_cache']]
    

    cache_hit_rate_shade =min(1,workload['shade_wss'] * workload['shade_repitation_factor'])
    cache_hit_rate_coordl =  min(1,cache_size / workload['total_dataset_size'])

    total_batches_cached_shade, total_batches_missed_shade = compute_total_batches_cached(cache_hit_rate_shade, workload)
    total_batches_cached_coordl, total_batches_missed_coordl = compute_total_batches_cached(cache_hit_rate_coordl, workload)

    # coordl_total_time += workload['cordl_first_epoch_time']
    # shade_total_time += workload['shade_first_epoch_time']


    # Loop over the epochs
    for i in range(total_epochs):
        shade_total_time += ((workload['shade_batch_time_on_hit'] * total_batches_cached_shade + 
                             workload['shade_batch_time_on_miss'] * total_batches_missed_shade +
                             workload['gpu_time'] * workload['batches_per_epoch']) / workload['num_jobs'])
        coordl_total_time += ((workload['coordl_batch_time_on_hit'] * total_batches_cached_coordl + 
                              workload['coordl_batch_time_on_miss'] * total_batches_missed_coordl +
                              workload['gpu_time'] * workload['batches_per_epoch']) / workload['num_jobs'])
    
    coordl_cost = compute_ec2_costs('p3.8xlarge', coordl_total_time) + compute_ec2_costs(workload['redis_cache'], coordl_total_time)
    shade_cost = compute_ec2_costs('p3.8xlarge', shade_total_time) + compute_ec2_costs(workload['redis_cache'], shade_total_time)
    # Output results
    # print(f"Super total time: {shade_total_time}, Throughput: {workload['total_files'] / shade_total_time}")
    print(f"Coordl time: {coordl_total_time:.4f}, Throughput (samples): {(total_epochs * workload['total_files']) / coordl_total_time:.4f}, Cache hit rate: {cache_hit_rate_coordl:.2f}, Cost: {coordl_cost:.4f}")
    print(f"Shade total time: {shade_total_time:.4f}, Throughput(samples): {(total_epochs * workload['total_files']) / shade_total_time:.4f}, Cache hit rate: {cache_hit_rate_shade:.4f}, Cost: {shade_cost:.4f}")
if __name__ == "__main__":
    simulate_training_job(workload=IMAGE_NET)  # Call the function without an argument
