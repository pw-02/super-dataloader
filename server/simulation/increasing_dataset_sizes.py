
def compute_serverless_redis_costs(total_durtion_seconds, cache_size_gb, throughput_per_s, avg_size_per_request_kb):
    hours_in_a_month = 730
    seconds_in_a_month = 2628000
    data_storage_cost_monthly = cache_size_gb * hours_in_a_month * 0.125

    requests = throughput_per_s * seconds_in_a_month * avg_size_per_request_kb
    ecpu_monthly_cost = requests * 0.0000000034

    total_monhtly_cost = data_storage_cost_monthly + ecpu_monthly_cost

    exp_cost = total_monhtly_cost/seconds_in_a_month * total_durtion_seconds
    return exp_cost


def run():
    dataset_sizes = {30: 268741, 60: 544913, 90: 835949, 120: 1096302, 150: 1365043}
    dataloaer_throuhgputs = {'coordl':1398, 'super':1435, 'shade':1398}
    super_cache_costs = {30: 0.769, 60: 2.44, 90: 3.66, 120: 4.88, 150: 6.11}
    num_jobs = 4
    p38xlarge_cost = 12.24/60/60
    for size in dataset_sizes:
        for loader in dataloaer_throuhgputs:
            total_durationn = dataset_sizes[size] * num_jobs / dataloaer_throuhgputs[loader]
            if loader != 'super':
                cache_cost = compute_serverless_redis_costs(total_durationn, size,dataloaer_throuhgputs[loader], 200)
            else:
                cache_cost = super_cache_costs[size]
            compute_cost =p38xlarge_cost * total_durationn
            total_cost = cache_cost + compute_cost
            print(f"size: {size} GB, DataLoader: {loader}, Duration:{total_durationn}, Throughput: {dataloaer_throuhgputs[loader]} samples/sec, Cost: {total_cost} USD")
            


if __name__ == '__main__':
    run()



