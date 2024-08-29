
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




class AWSLambdaClient():

    def __init__(self):
        self.lambda_client = None
    
    def invoke_function(self, function_name:str, payload, simulate = False):
        start_time = time.perf_counter()
        if simulate:
            time.sleep(0.01)
            response ={}
            response['duration'] = time.perf_counter() - start_time
            response['message'] = ""
            response['success'] = True
        else:
            if  self.lambda_client is None:
                self.lambda_client = boto3.client('lambda') 

            response = self.lambda_client.invoke(FunctionName=function_name,InvocationType='RequestResponse',Payload=payload)
            response_data = json.loads(response['Payload'].read().decode('utf-8'))
            if 'errorMessage' in response_data:
                response['duration'] = time.perf_counter() - start_time
                response['success'] = False
                response['message'] = response_data['errorMessage']
            else:
                response['duration'] = time.perf_counter() - start_time
                response['success'] = response_data['success']
                response['message'] = response_data['message']
        return response
    
    def warm_up_lambda(self, function_name):
        event_data = {'task': 'warmup'}
        return self.invoke_function(function_name, json.dumps(event_data))  # Pass the required payload or input parameters


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
    




