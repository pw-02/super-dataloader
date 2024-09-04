import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import boto3
import redis
from PIL import Image
import pickle
from botocore.config import Config
import torch
import torchvision.transforms as transforms
import base64
# Externalize configuration parameters
# s3_client = boto3.client('s3')

#  # Create a Config object with a larger pool size
# config = Config(
#         max_pool_connections=50  # Adjust this according to your needs
#     )
    
    # Create the S3 client with the custom config
s3_client = boto3.client('s3')
    

# redis_client = None

# def dict_to_torchvision_transform(transform_dict):
#     """
#     Converts a dictionary of transformations to a PyTorch transform object.
#     """
#     transform_list = []
#     for transform_name, params in transform_dict.items():
#         if transform_name == 'Resize':
#             transform_list.append(transforms.Resize(params))
#         elif transform_name == 'Normalize':
#             transform_list.append(transforms.Normalize(mean=params['mean'], std=params['std']))
#         elif params is None:
#             transform_list.append(getattr(transforms, transform_name)())
#         else:
#             raise ValueError(f"Unsupported transform: {transform_name}")
#     return transforms.Compose(transform_list)

def is_image_file(path: str) -> bool:
    """
    Checks if the file is an image based on its extension.
    """
    return any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp'])

def get_transform(bucket_name: str):
    # Load image
    if 'imagenet1k-sdl' in bucket_name:
       normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
        )
       return transforms.Compose([
            transforms.Resize(256),                    # Resize the image to 256x256 pixels
            transforms.RandomResizedCrop(224),   # Randomly crop a 224x224 patch
            transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            normalize,
        ])

    elif 'sdl-cifar10' in bucket_name:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding
            transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
            transforms.ToTensor(),                 # Convert the image to a PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5),  # Normalize the image
                                (0.5, 0.5, 0.5))  # Normalize the image
             ])
    else:
        return None
 

    # # Apply transformations
    # transformed = transform(image=img)
    # img_transformed = transformed['image']

    # return img_transformed

# def get_transform(bucket_name: str) -> transforms.Compose:
#     """
#     Returns a torchvision transform based on the bucket name.
#     """
#     if 'imagenet1k-sdl' in bucket_name:
#         return transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#     elif 'sdl-cifar10' in bucket_name:
#         return transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])

def bytes_to_mb(byte_data):
    size_in_bytes = len(byte_data)  # Get size in bytes
    size_in_mb = size_in_bytes / (1024 * 1024)  # Convert to megabytes
    return size_in_mb

def get_data_sample(bucket_name: str, data_sample: tuple, transform, s3_client) -> tuple:
    """
    Retrieves and transforms a sample from S3.
    """
    sample_path, sample_label = data_sample
    obj = s3_client.get_object(Bucket=bucket_name, Key=sample_path)
    data = Image.open(BytesIO(obj['Body'].read())).convert("RGB")
    # Apply transformations
    if transform:
        data = transform(data)
    return data, sample_label

def create_minibatch(bucket_name: str, samples: list, transform, s3_client) -> str:
    """
    Creates a minibatch from the samples, compresses it, and encodes it in base64.
    """
    batch_data, batch_labels = [], []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_data_sample, bucket_name, sample, transform, s3_client): sample for sample in samples}
        for future in as_completed(futures):
            data_sample, label = future.result()
            batch_data.append(data_sample)
            batch_labels.append(label)
            
    minibatch = torch.stack(batch_data), torch.tensor(batch_labels)
    with BytesIO() as buffer:
        torch.save(minibatch, buffer)
        compressed_minibatch = zlib.compress(buffer.getvalue())
        # Encode the serialized tensor with base64
        compressed_minibatch = base64.b64encode(compressed_minibatch).decode('utf-8')

    # minibatch_np = np.stack(batch_data)
    # labels_np = np.array(batch_labels)
    # Ensure correct shape for torch by permuting dimensions
    # minibatch_np = minibatch_np.transpose(0, 3, 1, 2)  # Convert to [batch_size, channels, height, width]

    # with BytesIO() as buffer:
    #     pickle.dump((minibatch_np, labels_np), buffer)
    #     compressed_minibatch = zlib.compress(buffer.getvalue())

    # minibatch = torch.stack(bacth_data), torch.tensor(batch_labels)

    # # Serialize and compress the PyTorch tensor
    # with BytesIO() as buffer:
    #     torch.save(minibatch, buffer)
    #     compressed_minibatch = zlib.compress(buffer.getvalue())

    # Encode the serialized tensor with base64
    return compressed_minibatch

def lambda_handler(event, context):
    """
    AWS Lambda handler function that processes a batch of images from an S3 bucket and caches the results in Redis.
    """
    # global s3_client, redis_client
    global s3_client

    try:
        task = event.get('task')
        if task == 'warmup':
            return {'success': True, 'message': 'function warmed'}

        bucket_name = event.get('bucket_name')
        batch_samples = event.get('batch_samples')
        batch_id = event.get('batch_id')
        
        cache_address = event.get('cache_address')
        cache_host, cache_port = cache_address.split(":")

        # # Initialize cache client if not already done
        # if redis_client is None:
        redis_client = redis.StrictRedis(host=cache_host, port=cache_port)

        transformformation = get_transform(bucket_name)


        minibatch = create_minibatch(bucket_name, batch_samples, transformformation, s3_client)
        
        minibatch_size_mb = bytes_to_mb(minibatch)

        # Cache minibatch in Redis using batch_id as the key
        redis_client.set(batch_id, minibatch)
        
        return {
            'success': True,
            'batch': {'batch_id':batch_id, 'num_files': len(batch_samples), 'size_mb': minibatch_size_mb}, 
            'is_cached': True,
            # 'message': f"Successfully cached '{batch_id}'"
        }
    except Exception as e:
        return {
            'success': False,
            'batch_id': batch_id,
            'is_cached': False,
            'message': str(e)
        }


if __name__ == '__main__':
    # Define the data dictionary with detailed formatting

    data = {
        "bucket_name": "imagenet1k-sdl",
        "batch_id": 2,
        "batch_samples": [
            ["train/n01440764/n01440764_10026.JPEG", 0],
            ["train/n01440764/n01440764_10026.JPEG", 0],
            ["train/n01440764/n01440764_10026.JPEG", 0],
        ],
        "cache_address": "54.184.21.219:6378",
        "task": "vision"
    }

    # Call the lambda_handler function with the defined data
    lambda_handler(data, None)
