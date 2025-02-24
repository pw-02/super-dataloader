import redis
import os

# Connect to Redis (modify host/port if needed)
r = redis.Redis(host='54.149.164.46', port=6378)

def creeat_50mb_file():
    # Create the 50MB file with random data (if not already created)
    file_name = "50MB_file.bin"
    if not os.path.exists(file_name):
        with open(file_name, "wb") as f:
            f.write(os.urandom(50 * 1024 * 1024))  # Write 50MB of random bytes

def cache_50mb_file_x_times(x, file_name = "50MB_file.bin"):
    # Cache the file x times with different keys
    for i in range(x):
        key = f"large_file_{i}"  # Create a unique key for each cache
        with open(file_name, "rb") as f:
            file_data = f.read()
            r.set(key, file_data)
        print(f"File cached in Redis with key: {key}")

    print("All files successfully cached in Redis!")
def get_50b_file_x_times(x):
    for i in range(x):
        key = f"large_file_{i}"
        file_data = r.get(key)
        with open(f"large_file_{i}_copy.bin", "wb") as f:
            f.write(file_data)
        print(f"File retrieved from Redis with key: {key}")

def put_in_hello_cache(batch_id):
     try:
        #gen random KEY using PRNG
        r.set(batch_id, 'hellohello')
     except Exception as e:
        print(f"Error fetching from cache: {e}")
        return None

def fetch_from_cache(batch_id):
     try:
        return r.get(batch_id)
     except Exception as e:
        print(f"Error fetching from cache: {e}")
        return None
        

if __name__ == "__main__":
   batch_id = 'batch_id_{}'.format(7)
   #cache_50mb_file_x_times(100)
   put_in_hello_cache(batch_id)
   print(fetch_from_cache(batch_id))
#   get_50b_file_x_times(10)
#    print(fetch_from_cache('large_file_0'))

