import boto3
import time
import os

# Initialize AWS clients


def export_logs_to_s3(log_group_name, s3_bucket_name, s3_prefix):
    logs_client = boto3.client('logs')

    """Exports CloudWatch logs to S3."""
    # Create a unique export task name
    task_name = f"export-{log_group_name}-{int(time.time())}"

    # Define the export task
    response = logs_client.create_export_task(
        taskName=task_name,
        logGroupName=log_group_name,
        fromTime=0,  # Export logs from the past day
        to=int(time.time() * 1000),  # Export logs up to now
        destination=s3_bucket_name,
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
    print(f"Export task created: {response['taskId']} for log group {log_group_name}")



def download_logs_from_s3(s3_bucket_name, s3_prefix, download_path):
    s3_client = boto3.client('s3')

    """Downloads exported logs from S3."""
    response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_prefix)
    if 'Contents' not in response:
        print(f"No logs found in S3 bucket {s3_bucket_name} with prefix {s3_prefix}.")
        return
    
    # Iterate over each file in the S3 prefix
    for obj in response['Contents']:
        # Get the S3 object key
        s3_key = obj['Key']
        
        # Create corresponding local path
        local_path = os.path.join(download_path, *s3_key.split('/'))
        local_path = os.path.normpath(local_path)  # Normalize path for Windows

        # Ensure the local directory exists
        local_dirname = os.path.dirname(local_path)
        os.makedirs(local_dirname, exist_ok=True)
        
        # Download the S3 object to the local path
        print(f"Downloading {s3_key} to {local_path}")
        s3_client.download_file(s3_bucket_name, s3_key, local_path)

    # for obj in response['Contents']:
    #     file_name = obj['Key'].split('/')[-1]

    #     s3_client.download_file(s3_bucket_name, obj['Key'], os.path.join(download_path, file_name))
    #     print(f"Downloaded: {file_name}")

def delete_log_group(log_group_name):
    logs_client = boto3.client('logs')
    """Deletes a CloudWatch log group."""
    response = logs_client.delete_log_group(logGroupName=log_group_name)
    print(f"Deleted log group: {log_group_name}")

def delete_all_groups():
    logs_client = boto3.client('logs')
    
    log_groups = logs_client.describe_log_groups(logGroupNamePrefix='/')['logGroups']

    for log_group in log_groups:
        log_group_name = log_group['logGroupName']
        delete_log_group(log_group_name)
        # print(f"Deleted log group: {log_group_name}")

def empty_s3_bucket(bucket_name):
    s3_client = boto3.client('s3')
    
    # List objects in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    
    if 'Contents' not in response:
        print(f"No objects found in bucket '{bucket_name}'.")
        return
    
    # Delete all objects
    objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
    
    # Delete the objects
    if objects_to_delete:
        print(f"Deleting {len(objects_to_delete)} objects from bucket '{bucket_name}'.")
        s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={
                'Objects': objects_to_delete,
                'Quiet': True
            }
        )
    
    # Check if there are more objects to delete (pagination)
    while response.get('IsTruncated'):  # Continue if there are more pages
        continuation_token = response.get('NextContinuationToken')
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            ContinuationToken=continuation_token
        )
        
        objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
        
        if objects_to_delete:
            print(f"Deleting {len(objects_to_delete)} more objects from bucket '{bucket_name}'.")
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={
                    'Objects': objects_to_delete,
                    'Quiet': True
                }
            )
    
    print(f"Bucket '{bucket_name}' has been emptied.")


def main():

    logs_client = boto3.client('logs')

    # Configuration
    s3_bucket_name = 'supercloudwtachexports'  # Replace with your S3 bucket name
    log_group_prefix = '/'  # Replace with the desired log group prefix to filter
    s3_download_path = 'cloudwatch\downloaded_logs'  # Directory where logs will be downloaded

    # Ensure the download path exists
    os.makedirs(s3_download_path, exist_ok=True)

    # List all CloudWatch log groups
    log_groups = logs_client.describe_log_groups(logGroupNamePrefix=log_group_prefix)['logGroups']

    for log_group in log_groups:
        log_group_name = log_group['logGroupName']
        print(f"Processing log group: {log_group_name}")

        # Export logs to S3
        s3_prefix = f'exported-logs/{log_group_name.replace("/", "_")}'
        export_logs_to_s3(log_group_name, s3_bucket_name, s3_prefix)

        # Wait for the export to complete (this can be replaced with more sophisticated checking)
        print("Waiting for export to complete...")

        # Download logs from S3
        download_logs_from_s3(s3_bucket_name, s3_prefix, s3_download_path)

        # # Delete logs from CloudWatch
        # delete_log_group(log_group_name)

if __name__ == "__main__":
    empty_s3_bucket('supercloudwtachexports')
    delete_all_groups()

