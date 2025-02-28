import numpy as np

# Constants for cost models
serverless_cost_per_request = 0.20 / 1e6  # $0.20 per million requests
redis_cost_per_GB_per_hour = 0.025  # Hypothetical cost for Redis per GB per hour

# ImageNet dataset characteristics
image_size_bytes = 3 * 224 * 224 * 4  # Assuming 3 channels, 224x224 images, 4 bytes per float32 pixel
num_images = 1281167  # Total number of images in ImageNet dataset

# Batch sizes (you can adjust as needed)
batch_sizes = [1]  # different batch sizes

# Hypothetical number of epochs (you can modify based on your experiments)
epochs = 60  # Number of epochs for training

# Calculating the total number of requests for each batch size
def calculate_num_requests(batch_size, epochs):
    return (num_images // batch_size) * epochs  # total requests to fetch mini-batches

# Function to calculate cost for serverless cache
def serverless_cache_cost(num_requests):
    return num_requests * serverless_cost_per_request

# Function to calculate cost for Redis
def redis_cache_cost(cache_size_gb, hours=1):
    return cache_size_gb * redis_cost_per_GB_per_hour * hours

# Function to estimate Redis cache size for storing the entire dataset
def estimate_redis_cache_size():
    total_dataset_size_bytes = num_images * image_size_bytes  # Total dataset size in bytes
    return total_dataset_size_bytes / (1024 ** 3)  # Convert to GB

# Example of running the comparison for ImageNet
def compare_cache_costs():
    results = []
    redis_cache_size_gb = estimate_redis_cache_size()  # Estimate total Redis cache size
    
    for batch_size in batch_sizes:
        num_requests = calculate_num_requests(batch_size, epochs)
        
        # Serverless Cache cost calculation
        serverless_cost = serverless_cache_cost(num_requests)
        
        # Redis Cache cost calculation (using the same cache size for all batch sizes)
        redis_cost = redis_cache_cost(redis_cache_size_gb)
        
        # Collecting the results
        results.append({
            'batch_size': batch_size,
            'num_requests': num_requests,
            'serverless_cost': serverless_cost,
            'redis_cost': redis_cost
        })
    
    return results

# Running the comparison for ImageNet dataset
results = compare_cache_costs()

# Displaying the results
for result in results:
    print(f"Batch Size: {result['batch_size']}")
    print(f"Number of Requests: {result['num_requests']}")
    print(f"Serverless Cache Cost: ${result['serverless_cost']:.4f}")
    print(f"Redis Cache Cost (for entire dataset): ${result['redis_cost']:.4f}")
    print("-" * 40)
