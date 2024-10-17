import redis
import concurrent.futures
import time

now = time.time()
host_ip = "34.213.163.60"
cache_client = redis.StrictRedis(host=host_ip, port=6378)

def put_in_cache(batch_id):
    try:
     # Initialize Redis client
      #   cache_client = redis.StrictRedis(host=host_ip, port=6378)
        cache_client.set(batch_id, 'hello')
      #   cache_client.close()
      #   if i % 500 == 0:
      #    print(f"Successfully cached batch_id: {batch_id}")
       
    except Exception as e:
        print(f"Error putting into cache for batch_id {batch_id}: {str(e)}. eLapsed time: {time.time() - now}")


def test_concurrent_put_fetch(num_requests):
    # Function to put and fetch from cache concurrently
    batch_ids = [f"batch_{i}" for i in range(num_requests)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit concurrent put and fetch tasks
        futures = []
        for batch_id in batch_ids:
            futures.append(executor.submit(put_in_cache, batch_id))
            # futures.append(executor.submit(fetch_from_cache, batch_id))
        
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    
    print(f"Tested {num_requests} concurrent cache operations.")

if __name__ == "__main__":
    # Number of concurrent cache operations you want to test
    num_requests = 10000  # Test with 100 concurrent requests
    
    start_time = time.perf_counter()
    test_concurrent_put_fetch(num_requests)
    end_time = time.perf_counter()
    
    print(f"Concurrent test completed in {end_time - start_time:.2f} seconds.")
