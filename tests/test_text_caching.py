# Initialize Redis client
import boto3
import redis
import numpy as np
from io import BytesIO
from transformers import AutoTokenizer
import torch
from transformers import default_data_collator
import lz4.frame

model_name = "EleutherAI/pythia-14m"
cache_client = redis.StrictRedis(host="localhost", port=6379)
tokenizer = AutoTokenizer.from_pretrained('awslambda\language\create_language_batch\pythia-14m-tokenizer')
batch_size = 5
max_length = 512


def main():
    bucket_name = 'owt-5mb-text-chunks'
    data_path = 'train/chunk-0-5.bin'
    s3 = boto3.client('s3')

    # Fetch the binary file from S3
    response = s3.get_object(Bucket=bucket_name, Key=data_path)
    content = response['Body'].read()
    # Store the content in Redis
    cache_client.set(1, content)
    # Retrieve the content from Redis
    binary_data = cache_client.get(1)
    # Create a BytesIO stream from the binary data
    binary_stream = BytesIO(binary_data)
    index = 0

    docs = []

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
        docs.append(data)
        # print(f"Chunk {index}: {data}")
    tokenized_data = []
    print(f"Total number of chunks: {index}")
    for doc in docs:
        tokenized_data.append(tokenizer(doc, return_tensors='pt'))
    # print(f"Tokenized data: {tokenized_data}")
    byte_data = save_tokenized_data(tokenized_data)
    compressed_byte_data =  lz4.frame.compress(byte_data)
    cache_client.set(2, compressed_byte_data)

    # Retrieve the tokenized data from Redis
    byte_data = cache_client.get(2)
    tokenized_data = load_tokenized_data(byte_data)

    batches = create_batches(tokenized_data, batch_size, max_length)
    pass


def create_batches(tokenized_data, batch_size, max_length):
    # Create a list to hold batches
    batches = []

    # Create a data collator to handle padding and batch creation
    # collator = default_data_collator()

    # Loop through tokenized data in chunks of batch_size
    for i in range(0, len(tokenized_data), batch_size):
        # Get the current batch of tokenized data
        batch = tokenized_data[i:i + batch_size]
        
        # Apply context length adjustments
        adjusted_batch = []
        for item in batch:
            # Adjust input_ids and attention_mask
            input_ids = truncate_or_pad_sequence(item['input_ids'], max_length)
            attention_mask = truncate_or_pad_sequence(item['attention_mask'], max_length)
            adjusted_batch.append({'input_ids': input_ids, 'attention_mask': attention_mask})
        
        # # Use the collator to handle additional padding and batching
        # batch = collator(adjusted_batch)
        
        # Append the batch to the list of batches
        batches.append(batch)

    return batches

def truncate_or_pad_sequence(sequence, max_length):
    # Truncate sequence if longer than max_length
    if sequence.size(1) > max_length:
        return sequence[:, :max_length]
    # Pad sequence if shorter than max_length
    elif sequence.size(1) < max_length:
        pad_length = max_length - sequence.size(1)
        return torch.cat([sequence, torch.zeros(sequence.size(0), pad_length, dtype=sequence.dtype)], dim=1)
    return sequence



def load_tokenized_data(byte_data):
    dempressed_data = lz4.frame.decompress(byte_data)
    # Create a BytesIO buffer from the byte data
    buffer = BytesIO(dempressed_data)
    
    # Load the tokenized data (list of tensors) from the buffer
    tokenized_data = torch.load(buffer)
    
    return tokenized_data


def save_tokenized_data(tokenized_data):
    # Create a BytesIO buffer to hold the bytes
    buffer = BytesIO()
    
    # Save the tokenized data (list of tensors) to the buffer
    torch.save(tokenized_data, buffer)
    
    # Get the byte data from the buffer
    byte_data = buffer.getvalue()
    
    return byte_data

if __name__ == '__main__':
    main()