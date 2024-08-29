import boto3

def get_total_invocations(function_name, start_time, end_time):
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

def get_hourly_invocations(function_name, start_time, end_time):
    client = boto3.client('cloudwatch')
    response = client.get_metric_statistics(
        Namespace='AWS/Lambda',
        MetricName='Invocations',
        Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
        StartTime=start_time,
        EndTime=end_time,
        Period=3600,  # 1-hour period
        Statistics=['Sum']
    )
    
    datapoints = response['Datapoints']
    hourly_invocations = {dp['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'): dp['Sum'] for dp in datapoints}
    return hourly_invocations


def get_memory_allocation(function_name):
    client = boto3.client('lambda')
    response = client.get_function_configuration(FunctionName=function_name)
    return response['MemorySize']

def get_average_duration(function_name, start_time, end_time):
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

# Function to compute Lambda cost
def compute_lambda_cost(requests, duration_sec, memory_mb):    
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


if __name__ == '__main__':
    from datetime import datetime, timedelta
    # Example usage
    function_name = 'CreateVisionTrainingBatch'
    requests = 1_000_000  # Number of requests
    duration_sec = 0.5  # Average duration of each invocation in seconds
    memory_mb = get_memory_allocation('CreateVisionTrainingBatch')
    # start_time = datetime.utcnow() - timedelta(minutes=45)
    start_time = datetime.utcnow()  - timedelta(minutes=30)
    end_time = datetime.utcnow()    
    average_duration = get_average_duration(function_name, start_time, end_time)
    total_invocations = get_total_invocations(function_name, start_time, end_time)
    hourly_invocation = get_hourly_invocations(function_name, start_time, end_time)

    # Compute the cost
    cost = compute_lambda_cost(requests, duration_sec, memory_mb)
    print(f"Total Lambda cost: ${cost:.2f}")
