
from io import BytesIO
import boto3
import redis
import torch
from transformers import AutoTokenizer
from collections import Counter
import os
import botocore.config
import numpy as np

# Create the S3 client with the custom config
s3_client = boto3.client('s3', config=botocore.config.Config(
    max_pool_connections=51
))
redis_client = None

# Set the TRANSFORMERS_CACHE environment variable
os.environ['HF_HOME'] = '/tmp'
tokenizer=AutoTokenizer.from_pretrained('pythia-14m-tokenizer')

def save_tokenized_data(tokenized_data):
    # Create a BytesIO buffer to hold the bytes
    buffer = BytesIO()
    # Save the tokenized data (list of tensors) to the buffer
    torch.save(tokenized_data, buffer)
    # Get the byte data from the buffer
    byte_data = buffer.getvalue()

    return byte_data

def prepare_data_chunk(bucket_name, data_path, s3_client):

    global tokenizer
    response = s3_client.get_object(Bucket=bucket_name, Key=data_path)
    content = response['Body'].read()
    binary_stream = BytesIO(content)
    tokenized_docs = []
    index = 0
    while True:
        offset = (1 + (index - 0) if index >= 0 else index + 1) * 4
        # Read the entire content of the binary file
        binary_stream.seek(offset)
        pair = binary_stream.read(8)
        begin, end = np.frombuffer(pair, np.uint32)
        if begin.item() == len(binary_stream.getvalue()):
            break
        binary_stream.seek(begin)
        raw_item_data = binary_stream.read(end - begin)

        shift_idx = 4
        sizes = np.frombuffer(raw_item_data[:shift_idx], np.uint32)
        data = ""
        for size, data_format in zip(sizes, 'str'):
            # size = size.item()
            data_bytes = raw_item_data[shift_idx : shift_idx + size]
            data += data_bytes.decode('utf-8')
            shift_idx += size
        index += 1
        tokenized_docs.append(tokenizer(data, return_tensors='pt'))
    
    return save_tokenized_data(tokenized_docs)


def lambda_handler(event, context):
    global s3_client
    global redis_client
    try:
        task = event.get('task')
        if task == 'warmup':
            return {'success': True, 'message': 'function warmed'}
        
        bucket_name = event.get('bucket_name')
        batch_samples = event.get('batch_samples')
        batch_id = event.get('batch_id')
        cache_address = event.get('cache_address', None)
        
        if not all([bucket_name, batch_samples, batch_id, cache_address]):
            return {'success': False, 'message': 'Missing parameters'}
        
        cache_host, cache_port = cache_address.split(":")
        if redis_client is None:
            redis_client = redis.StrictRedis(host=cache_host, port=int(cache_port))

        for data_path in batch_samples:
            tokenized_docs = prepare_data_chunk(bucket_name, data_path, s3_client)
            redis_client.set(batch_id, tokenized_docs)

        return {
            'success': True,
            'is_cached': True,
            'message': f"Successfully cached '{batch_id}'"
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }