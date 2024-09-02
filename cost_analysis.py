import boto3
import  pytz
import time
import math

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

def calculate_delay_between_requests(requests_per_hour, cost_per_request, cost_threshold):
    # Calculate the maximum allowable requests per hour within the cost threshold
    max_requests_per_hour = cost_threshold / cost_per_request
    
    # If the current request rate is within the threshold, no delay is needed
    if requests_per_hour <= max_requests_per_hour:
        return 0  # No delay needed
    
    # Calculate the required delay in hours between each request to stay within the cost threshold
    delay_per_request_hours = (1 / max_requests_per_hour) - (1 / requests_per_hour)
    
    # Convert delay from hours to seconds for practical use
    delay_per_request_seconds = delay_per_request_hours * 3600
    
    return delay_per_request_seconds

def calculate_delay(requests_per_hour, cost_per_request, cost_threshold):
    # Calculate the total cost per hour
    total_cost_per_hour = requests_per_hour * cost_per_request
    
    # Calculate the maximum allowable requests per hour within the cost threshold
    max_requests_per_hour = cost_threshold / cost_per_request
    
    # Determine if the current rate exceeds the threshold
    if total_cost_per_hour <= cost_threshold:
        return 0  # No delay needed as we are within the threshold
    
    # Calculate the excess number of requests
    excess_requests_per_hour = requests_per_hour - max_requests_per_hour
    
    # Calculate the delay required to spread out the excess requests
    delay_hours = excess_requests_per_hour / requests_per_hour
    
    return delay_hours

def simulate_cost_with_delay(requests_per_hour, cost_per_request, safe_delay_seconds):
    # Convert delay from seconds to hours
    safe_delay_hours = safe_delay_seconds / 3600
    
    # Calculate the effective request rate with the delay included
    requests_per_second = requests_per_hour / 3600
    effective_requests_per_second = 1 / (1 / requests_per_second + safe_delay_seconds)
    effective_requests_per_hour = effective_requests_per_second * 3600
    
    # Calculate the total cost for the adjusted requests
    total_cost = effective_requests_per_hour * cost_per_request
    
    return effective_requests_per_hour, total_cost


if __name__ == '__main__':
    from datetime import datetime, timedelta
    # Example usage
    function_name = 'CreateVisionTrainingBatch'
    requests = 1_000_000  # Number of requests
    duration_sec = 0.5  # Average duration of each invocation in seconds
    # start_time = datetime.utcnow() - timedelta(minutes=45)
    start_time = datetime.utcnow()  - timedelta(minutes=120)
    end_time = datetime.utcnow()
    
    memory_mb = get_memory_allocation('CreateVisionTrainingBatch')
    average_duration = get_average_duration(function_name, start_time, end_time)
    total_invocations = get_total_invocations(function_name, start_time, end_time)
    # Compute the cost
    current_cost  = compute_lambda_cost(total_invocations, average_duration, memory_mb)
    print(f"Total {function_name} cost: ${current_cost:.16f}")


    # Example usage
    cost_per_request = 0.0000003164791245 #current_cost/total_invocations  # Cost per request in dollars
    cost_threshold_per_hour = 10   # Maximum allowable cost per hour
    requests_per_hour = 60000000      # Number of requests per hour

    delay = calculate_delay_between_requests(
        requests_per_hour, 
        cost_per_request, 
        cost_threshold_per_hour,
    )

    # Simulate cost and effective request rate with the calculated delay
    effective_requests_per_hour, total_cost = simulate_cost_with_delay(
        requests_per_hour, 
        cost_per_request, 
        delay
        )
    
    print(f"Delay to stay within cost threshold: {delay:.8f} seconds")
    print(f"Effective requests per hour with delay: {math.ceil(effective_requests_per_hour)} requests/hour")
    print(f"Total cost with delay: ${total_cost:.2f}")





