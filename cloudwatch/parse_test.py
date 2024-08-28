import re
import csv

# Define the input log file and output CSV file
log_file = 'cloudwatch/exports/6d29fbaa-bb9f-4c40-90df-afad2cdcce25/2024-08-28-[$LATEST]4eeec71d18b1461580e35b89cb060777/000000'
csv_file = 'lambda_logs.csv'

# Initialize a list to hold the parsed log data
log_data = []

# Define the regex patterns for matching log entries
start_pattern = re.compile(r'START RequestId: ([\w-]+)')
report_pattern = re.compile(r'REPORT RequestId: ([\w-]+)\s*Duration: ([\d.]+) ms\s*Billed Duration: ([\d.]+) ms\s*Memory Size: (\d+) MB\s*Max Memory Used: (\d+) MB(?:\s*Init Duration: ([\d.]+) ms)?')
timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)')


# Temporary storage for log entries
current_request = {}

# Read the log file and parse the entries
with open(log_file, 'r') as file:
    content = file.read()
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
                    'Duration': duration,
                    'Billed Duration': billed_duration,
                    'Memory Size': memory_size,
                    'Max Memory Used': max_memory_used,
                    'Init Duration': init_duration if init_duration else '0'
                })
                # Append data when report is fully parsed
                log_data.append(current_request)
                current_request = {}


    

    # Write the parsed data to a CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Timestamp','RequestId', 'Duration', 'Billed Duration', 'Memory Size', 'Max Memory Used', 'Init Duration'])
        writer.writeheader()
        writer.writerows(log_data)

    print(f"CSV file '{csv_file}' has been created with the parsed log data.")
