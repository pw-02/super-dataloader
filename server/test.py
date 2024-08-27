import math

# Define constants
N_prefetch = 15  # Number of batches prefetched at a time
T_prefetch = 5  # Time taken to prefetch N_prefetch batches (seconds)
T_miss = 0.25  # Processing time for a cache miss (seconds)
T_hit = 0.25  # Processing time for a cache hit (seconds)
B_total = 1000  # Total number of batches to process

# Initialize counters for hits and misses
miss_count = 0
hit_count = 0

# Simulate processing of each batch
current_time = 0

for i in range(B_total):
    if i == 0 or (i % (N_prefetch + 1)) == 0:
        # Cache miss condition
        miss_count += 1
        current_time += T_miss
    else:
        # Cache hit condition
        hit_count += 1
        current_time += T_hit

    # Check if prefetching is done before the next request
    if i > 0 and (i + 1) % N_prefetch == 0:
        prefetch_time = (i + 1) // N_prefetch * T_prefetch
        if current_time < prefetch_time:
            current_time = prefetch_time

# Output the results
print(f"Total Cache Misses: {miss_count}")
print(f"Total Cache Hits: {hit_count}")
