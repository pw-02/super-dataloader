
import boto3
from urllib.parse import urlparse
import json
import time
from boto3.exceptions import botocore
from datetime import datetime, timedelta, timezone

class S3Url(object):
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()
    
def get_boto_s3_client():
    return boto3.client('s3')

def check_s3_object_exists(file_path:str, s3_client = None):
    try:
        if s3_client is None:
            s3_client = get_boto_s3_client()
            s3_client.head_object(Bucket=S3Url(file_path).bucket, Key=S3Url(file_path).key)
            return True
    except Exception:
        return False

def get_total_lambda_invocations(function_name, start_time, end_time):
    client = boto3.client('cloudwatch')
    response = client.get_metric_statistics(
        Namespace='AWS/Lambda',
        MetricName='Invocations',
        Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600 ,  # 1-hour period #
        Statistics=['Sum']
    )
    datapoints = response['Datapoints']
    total_invocations = sum(dp['Sum'] for dp in datapoints)
    return total_invocations


def get_average_lambda_duration(function_name, start_time=datetime.now(timezone.utc)  - timedelta(minutes=30), end_time = datetime.now(timezone.utc)):
        client = boto3.client('cloudwatch')
        response = client.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Duration',
            Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,  # 5-minute intervals
            Statistics=['Average']
        )
        
        datapoints = response['Datapoints']
        if datapoints:
            average_duration = sum(dp['Average'] for dp in datapoints) / len(datapoints)
            return average_duration / 1000  # Convert from ms to seconds
        else:
            return None

    # # Function to get current AWS costs using Cost Explorer
    # def get_current_costs():
    #     ce_client = boto3.client('ce')  # Cost Explorer client
    #     response = ce_client.get_cost_and_usage(
    #         TimePeriod={'Start': '2024-08-01', 'End': '2024-08-28'},  # Example dates
    #         Granularity='DAILY',
    #         Metrics=['UnblendedCost']
    #     )
    #     total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    #     return total_cost

def get_memory_allocation_of_lambda(function_name):
        client = boto3.client('lambda')
        response = client.get_function_configuration(FunctionName=function_name)
        return response['MemorySize']
    
# Function to compute Lambda cost
def compute_lambda_cost(requests, duration_sec, memory_mb):    
    if requests < 1 or duration_sec == 0:
        return 0
    # AWS Lambda pricing details (replace these with the latest rates from AWS)
    cost_per_million_requests = 0.20 / 1_000_000  # Example cost per request in dollars
    cost_per_gb_second = 0.00001667  # Example cost per GB-second in dollars

    # Convert memory in MB to GB
    memory_gb = memory_mb / 1024

    # Calculate total duration in GB-seconds
    duration_gb_seconds = (duration_sec / 3600) * memory_gb
    
    # Compute request cost
    request_cost = requests * cost_per_million_requests

    # Compute duration cost
    duration_cost = duration_gb_seconds * cost_per_gb_second

    # Total cost
    total_cost = request_cost + duration_cost
    return total_cost



