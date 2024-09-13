from typing import List, Tuple, Dict
import functools
from epoch import Epoch
from typing import List, Tuple
import numpy as np
from utils import partition_dict
from aws_utils import S3Url
import json
from urllib.parse import urlparse
import boto3
from boto3.exceptions import botocore

# The base class `BaseDataset` contains common properties and methods shared by both `Dataset` and `DatasetPartition`.
# Define the base class with common properties and methods
class BaseDataset:
    def __init__(self, samples: Dict[str, List[str]], batch_size: int, drop_last: bool):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.samples = samples

        # Calculate the number of batches
        if self.drop_last:
            self.num_batches = len(self) // self.batch_size
        else:
            self.num_batches = (len(self) + self.batch_size - 1) // self.batch_size

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
                for class_index, blob_class in enumerate(self.samples)
                for blob in self.samples[blob_class]]
    
    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())
    
    def get_samples(self, indices: List[int]):
        samples = []
        for i in indices:
            samples.append(self._classed_items[i])
        return samples
    

# Dataset class inherits from BaseDataset
class Dataset(BaseDataset):
    def __init__(self, data_dir: str, batch_size: int, drop_last: bool, num_partitions: int = 10, kind = 'vision'):
        # Load samples from data directory
        self.data_dir = data_dir
        self.batch_size = batch_size
        if kind == 'vision':
            samples = self.load_paired_s3_object_keys(
            data_dir, True, True
            )
        else:
            samples = self.load_paired_s3_object_keys(
                data_dir, False, True
            )
            
        self.bucket_name =  S3Url(data_dir).bucket
        super().__init__(samples, batch_size, drop_last)
        # Call the base class initializer
        super().__init__(self.samples, batch_size, drop_last)
        self.partitions = self._create_partitions(num_partitions)
    
    def is_image_file(self, path: str):
        return any(path.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'])

    def load_paired_s3_object_keys(self, s3_uri:str, images_only:bool, use_index_file:bool = True):
        paired_samples = {}
        s3url = S3Url(s3_uri)
        s3_client = boto3.client('s3')
        index_file_key = s3url.key + '_paired_index.json'
        if use_index_file:
            try:
                index_object = s3_client.get_object(Bucket=s3url.bucket, Key=index_file_key)
                file_content = index_object['Body'].read().decode('utf-8')
                paired_samples = json.loads(file_content)
                return paired_samples
            except botocore.exceptions.ClientError as e:
                print(f"Error reading index file '{index_file_key}': {str(e)}")

        # If use_index_file is False or encounter errors with index file, build paired_samples from S3 objects
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3url.bucket, Prefix=s3url.key)
        for page in pages:
            for blob in page.get('Contents', []):
                blob_path = blob.get('Key')
                if blob_path.endswith("/"):
                    continue  # Ignore folders
                
                stripped_path = self.remove_prefix(blob_path, s3url.key).lstrip("/")
                if stripped_path == blob_path:
                    continue  # No matching prefix, skip
                
                if images_only and not self.is_image_file(blob_path):
                    continue  # Skip non-image files
                
                if 'index.json' in blob_path:
                    continue

                blob_class = stripped_path.split("/")[0]
                blobs_with_class = paired_samples.get(blob_class, [])
                blobs_with_class.append(blob_path)
                paired_samples[blob_class] = blobs_with_class
        
        if use_index_file and len(paired_samples) > 0:
            index_object = s3_client.put_object(Bucket=s3url.bucket, Key=index_file_key, 
                                                Body=json.dumps(paired_samples, indent=4).encode('UTF-8'))    
        return paired_samples
    
    def remove_prefix(self, s: str, prefix: str) -> str:
        if not s.startswith(prefix):
            return s
        return s[len(prefix) :]



    def _create_partitions(self, num_partitions: int) -> Dict[int, 'DatasetPartition']:
        partitions = partition_dict(self.samples, num_partitions, self.batch_size)

        for partition in partitions:
            partition_size = sum(len(samples) for samples in partition.values())
            if partition_size == 0:
                partitions.remove(partition)
        
        return {idx + 1: DatasetPartition(idx + 1, partition, self.batch_size, self.drop_last)
                for idx, partition in enumerate(partitions)}
    
    def summarize(self):
        print(f"Data Directory: {self.data_dir}\nTotal Files: {len(self)}\nTotal Batches: {self.num_batches}")


# DatasetPartition class inherits from BaseDataset
class DatasetPartition(BaseDataset):
    def __init__(self, partition_id: int, samples: Dict[str, List[str]], batch_size: int, drop_last: bool):
        # Call the base class initializer
        super().__init__(samples, batch_size, drop_last)
        # Initialize partition-specific properties
        self.partition_id = partition_id
        self.epochs: Dict[int, Epoch] = {}
    
    def __len__(self):
        return super().__len__()
