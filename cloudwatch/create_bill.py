import boto3
from datetime import datetime, timedelta
import time
import os
import gzip
import glob
import csv
import re
# Initialize the CloudWatch Logs client
# Example timestamps (replace these with your experiment start and end times)
experiment_start_time = '2024-08-27T10:00:00Z'
experiment_end_time = '2024-08-29T12:00:00Z'
# Convert ISO 8601 format to datetime object
start_time = datetime.fromisoformat(experiment_start_time.replace('Z', '+00:00'))
end_time = datetime.fromisoformat(experiment_end_time.replace('Z', '+00:00'))

# Convert datetime to Unix timestamp (milliseconds)
start_timestamp = int(start_time.timestamp() * 1000)
end_timestamp = int(end_time.timestamp() * 1000)

def empty_log_group(log_group_name = 'CreateVisionTrainingBatch'):
    logs_client = boto3.client('logs')
    paginator = logs_client.get_paginator('describe_log_streams')
    for page in paginator.paginate(logGroupName=log_group_name):
        for stream in page['logStreams']:
            logs_client.delete_log_stream(logGroupName=log_group_name, logStreamName=stream['logStreamName'])
    logs_client.delete_log_group(logGroupName=log_group_name)

def export_logs_to_s3(log_group_name = '/aws/lambda/CreateVisionTrainingBatch', s3_bucket = 'supercloudwtachexports', s3_prefix = 'logs'):
    logs_client = boto3.client('logs')
    response = logs_client.create_export_task(
        logGroupName=log_group_name,
        fromTime=start_timestamp,
        to=end_timestamp,
        destination=s3_bucket,
        destinationPrefix=s3_prefix
    )
    task_id = response['taskId']
    # Monitor the status of the export task
    while True:
        request = logs_client.describe_export_tasks(taskId=task_id)
        status = request['exportTasks'][0]['status']['code']
        print(f'Task ID {task_id} status: {status}')

        if status in ['COMPLETED', 'FAILED']:
            break
    
        # Wait for a while before checking the status again
        time.sleep(5)
    return response['taskId']

def download_exported_logs(s3_bucket = 'supercloudwtachexports', export_prefix = 'exports/1', destination_folder = 'cloudwatch'):
    base_dir = os.path.join(destination_folder, export_prefix)
    # Create the base directory if it does not exist
    os.makedirs(base_dir, exist_ok=True)

    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=export_prefix):
        for obj in page.get('Contents', []):
            object_key = obj['Key']
            local_path = os.path.join(destination_folder, object_key)
            local_dir = os.path.dirname(local_path)
             # Create local directory if it does not exist
            os.makedirs(local_dir, exist_ok=True)
            print(f'Downloading object: s3://{s3_bucket}/{object_key}')
            s3_client.download_file(s3_bucket, object_key, local_path)
    else:
        print('Export task failed or is still in progress.')


# Function to get current AWS costs using Cost Explorer
def get_current_aws_costs():
    ce_client = boto3.client('ce')  # Cost Explorer client
    response = ce_client.get_cost_and_usage(
        TimePeriod={'Start': '2024-08-01', 'End': '2024-08-28'},  # Example dates
        Granularity='DAILY',
        Metrics=['UnblendedCost']
    )
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost

def prarse_exported_logs(destination_folder = 'cloudwatch', export_prefix = 'exports', skip_unzip = False):
    base_dir = os.path.join(destination_folder, export_prefix)
    output_file = os.path.join(base_dir, "bill.csv")
    decompressed_file_paths = []
    base_dir = os.path.join(destination_folder, export_prefix)

    if not skip_unzip:
         # Traverse the directory hierarchy
        for dirpath, _, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.endswith('.gz'):
                    gz_file_path = os.path.join(dirpath, filename)
                    decompressed_file_path = gz_file_path[:-3]  # Remove '.gz' suffix
                    
                    # Decompress the file
                    with gzip.open(gz_file_path, 'rb') as f_in:
                        with open(decompressed_file_path, 'wb') as f_out:
                            f_out.write(f_in.read())
                    
                    # Save the decompressed file path
                    decompressed_file_paths.append(decompressed_file_path)

    log_data = []
    start_pattern = re.compile(r'START RequestId: ([\w-]+)')
    report_pattern = re.compile(r'REPORT RequestId: ([\w-]+)\s*Duration: ([\d.]+) ms\s*Billed Duration: ([\d.]+) ms\s*Memory Size: (\d+) MB\s*Max Memory Used: (\d+) MB(?:\s*Init Duration: ([\d.]+) ms)?')
    timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)')

    for log_file in decompressed_file_paths:
        with open(log_file, 'r') as log_file:
            content = log_file.read()
            # Extracting logs using regex
            current_request = {}
            for line in content.splitlines():
                start_match = start_pattern.search(line)
                report_match = report_pattern.search(line)
                
                if start_match:
                    request_id = start_match.group(1)
                    current_request = {
                        'Timestamp': '',
                        'RequestId': request_id,
                        'Duration': '0',
                        'Billed Duration': '0',
                        'Memory Size': '0',
                        'Max Memory Used': '0',
                        'Init Duration': '0'
                    }
                
                if report_match:
                    request_id, duration, billed_duration, memory_size, max_memory_used, init_duration = report_match.groups()
                    if request_id == current_request.get('RequestId'):
                        match = timestamp_pattern.search(line)
                        timestamp = match.group(1)
                        current_request.update({
                           'Timestamp': timestamp,
                            'RequestId': request_id,
                            'Duration': duration,
                            'Billed Duration': billed_duration,
                            'Memory Size': memory_size,
                            'Max Memory Used': max_memory_used,
                            'Init Duration': init_duration if init_duration else '0'
                        })
                        # Append data when report is fully parsed
                        log_data.append(current_request)
                        current_request = {}

    
    # Write parsed data to CSV
    with open(output_file, 'w', newline='') as csv_file:
        fieldnames = ['Timestamp', 'RequestId', 'Duration', 'Billed Duration', 'Memory Size', 'Max Memory Used', 'Init Duration']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_data)
    
if __name__ == '__main__':
    # Get the current AWS costs
    log_group_name = '/aws/lambda/CreateVisionTrainingBatch'
    s3_bucket = 'supercloudwtachexports'
    export_prefix = 'exports'
    destination_folder = 'cloudwatch'
    # empty_log_group(log_group_name)
    # export_logs_to_s3(log_group_name, s3_bucket, export_prefix)
    download_exported_logs(s3_bucket, export_prefix, 'cloudwatch')
    prarse_exported_logs(destination_folder,export_prefix, False)
