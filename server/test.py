import math
import boto3

ce_client = boto3.client('ce')  # Cost Explorer client

# Function to get current AWS costs using Cost Explorer
def get_aws_current_costs():
    ce_client = boto3.client('ce')  # Cost Explorer client
    response = ce_client.get_cost_and_usage(
        TimePeriod={'Start': '2024-08-01', 'End': '2024-08-28'},  # Example dates
        Granularity='DAILY',
        Metrics=['UnblendedCost']
    )
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost

def get_lambda_costs(start_date, end_date):
    response = ce_client.get_cost_and_usage(
        TimePeriod={
            'Start': start_date,  # Start date in 'YYYY-MM-DD' format
            'End': end_date       # End date in 'YYYY-MM-DD' format
        },
        Granularity='DAILY',  # Can be 'DAILY', 'MONTHLY', or 'HOURLY'
        Metrics=['UnblendedCost'],  # Use 'UnblendedCost' to get the actual cost
        Filter={
            'Dimensions': {
                'Key': 'SERVICE',
                'Values': ['AWS Lambda']  # Filter for AWS Lambda only
            }
        }
    )

    # Sum up costs for the entire period
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost

# Function to get costs for a single AWS Lambda function using a tag
def get_lambda_cost_by_tag(start_date, end_date, tag_key, tag_value):
    response = ce_client.get_cost_and_usage(
        TimePeriod={
            'Start': start_date,  # Start date in 'YYYY-MM-DD' format
            'End': end_date       # End date in 'YYYY-MM-DD' format
        },
        Granularity='DAILY',  # Can be 'DAILY', 'MONTHLY', or 'HOURLY'
        Metrics=['UnblendedCost'],  # Use 'UnblendedCost' to get the actual cost
        Filter={
            'And': [
                {
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['AWS Lambda']  # Filter for AWS Lambda service only
                    }
                },
                {
                    'Tags': {
                        'Key': tag_key,  # Specify the tag key
                        'Values': [tag_value]  # Specify the tag value
                    }
                }
            ]
        }
    )

    # Sum up costs for the entire period
    total_cost = sum(float(day['Total']['UnblendedCost']['Amount']) for day in response['ResultsByTime'])
    return total_cost


print(get_lambda_costs('2024-08-01', '2024-08-28'))  # Example dates
print(get_lambda_cost_by_tag('2024-08-01', '2024-08-28', 'aws:cloudformation:logical-id', '	CreateVisionBatchFunction'))  # Example tag
# Define constants









def find_optimal_prefetch_conncurrency(T_prefetch=5, T_hit=0.25,):
    optimal_prefetch_concurrency = math.ceil(T_prefetch / T_hit)
    return optimal_prefetch_concurrency

def estimate_prefetch_cost(cost_per_prefetch, number_of_prefetches_per_hour, cost_cap_per_hour):
    total_cost_per_hour = cost_per_prefetch * number_of_prefetches_per_hour
    max_allowed_prefetches = cost_cap_per_hour / cost_per_prefetch
    pass

def calculate_metrics(T_prefetch=5, N_prefetch=15, T_hit=0.25, T_miss=5.25, B_total=1000):
    """
    Calculate total cache hits and misses during the prefetching process.
    
    Parameters:
        T_prefetch (float): Time allocated for prefetching in seconds.
        N_prefetch (int): Number of batches that can be prefetched within the T_prefetch time.
        T_hit (float): Time taken for a cache hit in seconds.
        T_miss (float): Time taken for a cache miss in seconds.
        B_total (int): Total number of batches to be processed.
        
    Returns:
        total_misses (int): Total number of cache misses.
        total_hits (int): Total number of cache hits.
    """
    # Calculate the number of misses that occur during the initial prefetching phase of N_prefetch batches
    misses_during_first_prefetch = math.ceil(T_prefetch / T_miss)
  
    # Number of hits before we start getting misses after the prefetch
    hits_per_cycle = N_prefetch

    # Time gap after processing all the prefetched hits
    time_gap = max(T_prefetch - (hits_per_cycle * T_hit),0)
    
    # Number of misses that can occur during this time gap
    misses_per_cycle = math.ceil(time_gap / T_miss)
    
    # Number of hits and misses in each cycle
    cycle_size = hits_per_cycle + misses_per_cycle

    total_cycles = (B_total-1) / cycle_size #we minius 1 for the first prefetching phase not included in the cycle

    # Calculate total misses and hits
    total_misses = math.floor(total_cycles * misses_per_cycle + misses_during_first_prefetch)
    total_hits = B_total - total_misses
    total_time = int(total_hits) * T_hit + int(total_misses) * T_miss
    return int(total_misses), int(total_hits), total_time

