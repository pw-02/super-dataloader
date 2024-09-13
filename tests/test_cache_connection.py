import redis
import lz4.frame
from io import BytesIO
import torch
# grpc_server_address: 44.243.2.122:50051
# cache_address: 44.243.2.122:6378 #null


cache_client =redis.StrictRedis = redis.StrictRedis(host="34.221.12.143", port=6378)

def put_in_cache(batch_id):
     try:
        cache_client.set(batch_id, 'hello')
     except Exception as e:
        print(f"Error fetching from cache: {e}")
        return None

def fetch_from_cache(batch_id):
     try:
        return cache_client.get(batch_id)
     except Exception as e:
        print(f"Error fetching from cache: {e}")
        return None
        

if __name__ == "__main__":
   respsonse = fetch_from_cache("1_1_8_8f14e45fceea167a")
   compressed_batch = lz4.frame.decompress(respsonse)
   with BytesIO(compressed_batch) as buffer:
      samples = torch.load(buffer)
   print(samples)