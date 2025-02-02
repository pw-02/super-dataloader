import numpy as np
import matplotlib.pyplot as plt

#number_of_requests_per_epoch
number_of_requests_per_epoch = 1281167
number_of_epochs = 90
number_of_jobs = 1
# total_number_of_requests = number_of_requests_per_epoch * number_of_epochs * number_of_jobs
# total_cost_of_request_to_cache = 0.20 / 1e6 * total_number_of_requests

for i in range(1, 21):
    number_of_jobs = i
    total_number_of_requests = number_of_requests_per_epoch * number_of_epochs * number_of_jobs
    total_cost_of_request_to_cache = 0.20 / 1e6 * total_number_of_requests / 1
    print(f"Total cost of request to cache for {i} jobs: {total_cost_of_request_to_cache}")

# # Constants
# requests_per_second_per_job = 1300  # Each job makes 1000 requests per second
# seconds_per_hour = 3600  # 1 hour in seconds
# serverless_cost_per_request = 0.20 / 1e6  # $0.20 per million requests

# # Redis cost assumption (adjust based on actual Redis pricing)
# imagenet_size_gb = 150  # ImageNet dataset size
# redis_price_per_gb_per_hour = 0.25  # Example: $0.25 per GB per hour (modify as needed)
# redis_hourly_cost = 5.184 #imagenet_size_gb * redis_price_per_gb_per_hour  # Fixed Redis cost per hour

# # Define job counts to test
# job_counts = np.array([1, 4, 8, 12, 16,20])

# # Calculate total requests per hour for each job count
# requests_per_hour = job_counts * requests_per_second_per_job * seconds_per_hour

# # Compute serverless cost per hour
# serverless_costs_per_hour = requests_per_hour * serverless_cost_per_request

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(job_counts, serverless_costs_per_hour, marker='o', linestyle='-', color='blue', label="Serverless Cache Cost")
# plt.axhline(y=redis_hourly_cost, color='red', linestyle='--', label="Redis Fixed Cost")

# plt.xlabel("Number of Jobs")
# plt.ylabel("Hourly Cost ($)")
# # plt.title("Serverless Cache Cost vs. Redis Cost (Hourly)")
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.legend()
# plt.show()
