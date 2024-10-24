import glob
import pandas as pd
import os
from collections import OrderedDict
import csv

def convert_csv_to_dict(csv_file, first_epoch_only = False):
    df = pd.read_csv(csv_file)
    if 'bill.csv' in csv_file:
        return df.to_dict(orient='list')
    # Filter the rows where 'Epoch Index' is equal to 1
    if first_epoch_only:
        filtered_df = df[df['Epoch Index'] == 1]
    else:
        if df[df['Epoch Index'] > 1].empty:
            filtered_df = df[df['Epoch Index'] == 1]

        else:
            filtered_df = df[df['Epoch Index'] > 1]

    return filtered_df.to_dict(orient='list')

def save_dict_list_to_csv(dict_list, output_file):
    if not dict_list:
        print("No data to save.")
        return
    headers = dict_list[0].keys()
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for data in dict_list:
            writer.writerow(data)


def get_subfolder_names(folder_path, include_children = False):
    subfolders = glob.glob(os.path.join(folder_path, '*'))
    basenames = []
    for subfolder in subfolders:
        if include_children:
            subfolder_names = glob.glob(os.path.join(subfolder, '*'))
            for subfolder_name in subfolder_names:
                if os.path.isdir(subfolder_name):
                    basenames.append(os.path.basename(os.path.normpath(subfolder_name)))
        else:
            if os.path.isdir(subfolder):
                basenames.append(os.path.basename(os.path.normpath(subfolder)))
    return basenames

def get_first_epoch_only_from_file(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '**', 'metrics.csv'), recursive=True)
    first_epoch_data = []
    for csv_file in csv_files:
        csv_data = convert_csv_to_dict(csv_file)
        
    return first_epoch_data

def compute_serverless_redis_costs(total_durtion_seconds, cache_size_gb, throughput_per_s, avg_size_per_request_kb):
    # Duration is in seconds
    # Memory size is in GB
    # Cost is in USD
    hours_in_a_month = 730
    seconds_in_a_month = 2628000
    # round_duartion_tonearest_hour = total_durtion_seconds / 3600
    # rounded_duarion = round_duartion_tonearest_hour * 3600
    data_storage_cost_monthly = cache_size_gb * hours_in_a_month * 0.125

    requests = throughput_per_s * seconds_in_a_month * avg_size_per_request_kb
    ecpu_monthly_cost = requests * 0.0000000034

    total_monhtly_cost = data_storage_cost_monthly + ecpu_monthly_cost

    exp_cost = total_monhtly_cost/seconds_in_a_month * total_durtion_seconds
    return exp_cost


def scale_data_for_epoch(data, folder_path):
    scaled_for_one_epoch_metrics = {}
    if 'imagenet' in folder_path:
        if 'super' not in exp_summary['name']:

            total_files = 1096302 * data['num_jobs']
            total_batches = 8565 * data['num_jobs']
            dataset_Size_gb = 120

            # num_files_80_percent = 1096302 * 0.8 * data['num_jobs']
            # num_files_60_percent = 1096302 * 0.6 * data['num_jobs']
            # num_files_40_percent = 1096302 * 0.4 * data['num_jobs']
            # num_files_20_percent = 1096302 * 0.2 * data['num_jobs']

            cache_hits_throughpput = 1398.02888496756
            cache_hits_wait_on_data_percent = 0.01
            cache_hits_wait_on_transformation_percent = 0.06
            cache_hits_gpu_processing_percent = 0.93

            cache_miss_throughpput = 170.124
            cache_miss_wait_on_data_percent = 0.84
            cache_miss_wait_on_transformation_percent = 0.04
            cache_miss_gpu_processing_percent = 0.11

            for size in [1, 0.8, 0.6, 0.4, 0.2]:
                cahce_hits = total_files * size
                cache_misses = total_files - cahce_hits
                time_for_cache_hit = cahce_hits / cache_hits_throughpput
                time_for_cache_miss = cache_misses / cache_miss_throughpput
                total_time = time_for_cache_hit + time_for_cache_miss
                total_throughput = total_files / total_time
                compute_cost = compute_ec2_costs('p3.8xlarge', total_time)
                cache_size = dataset_Size_gb * size
                cache_cost = compute_serverless_redis_costs(total_time, cache_size, total_throughput, 200)

                time_spent_on_data = time_for_cache_hit * cache_hits_wait_on_data_percent + time_for_cache_miss * cache_miss_wait_on_data_percent
                time_spent_on_transformation = time_for_cache_hit * cache_hits_wait_on_transformation_percent + time_for_cache_miss * cache_miss_wait_on_transformation_percent
                time_spent_on_gpu_processing = time_for_cache_hit * cache_hits_gpu_processing_percent + time_for_cache_miss * cache_miss_gpu_processing_percent

                percent_spent_on_data = time_spent_on_data / total_time
                percent_spent_on_transformation = time_spent_on_transformation / total_time
                percent_spent_on_gpu_processing = time_spent_on_gpu_processing / total_time

                scaled_for_one_epoch_metrics[f'{size*100}%_cost'] = cache_cost + compute_cost
                # scaled_for_one_epoch_metrics[f'{size*100}%_duration'] = total_time
                scaled_for_one_epoch_metrics[f'{size*100}%_throughput'] = total_throughput
                scaled_for_one_epoch_metrics[f'{size*100}%_IO%'] =  percent_spent_on_data
                scaled_for_one_epoch_metrics[f'{size*100}%_Transformation%'] = percent_spent_on_transformation
                scaled_for_one_epoch_metrics[f'{size*100}%_GPU%'] = percent_spent_on_gpu_processing
                # scaled_for_one_epoch_metrics[f'{size*100}%_time_breakdown'] = {'IO': percent_spent_on_data, 'Transformation:': percent_spent_on_transformation, 'GPU': percent_spent_on_gpu_processing}


    return scaled_for_one_epoch_metrics

  
def get_training_summary(folder_path, kind):
    first_epoch_only = True if kind == 'first_epoch' else False
    metrics = OrderedDict({
         "num_jobs": 0,
         "total_batches": 0,
         "total_samples": 0,
         "total_tokens": 0,
         "total_time(s)": 0,
         "wait_on_data_time(s)": 0,
         "gpu_processing_time(s)": 0,
         "data_fetch_time(s)": 0,
         "transformation_time(s)": 0,
         "cache_hits": 0,
        #  "avg_gpu_time(s)": 0,
        #  "avg_data_fetch_time(s)": 0,
        # "avg_data_transformation_time_on_hit(s)": 0,
        # "avg_data_transformation_time_on_miss(s)": 0,
        # "avg_data_fetch_time_on_hit(s)": 0,
        # "avg_data_fetch_time_on_miss(s)": 0,
        # "avg_transformation_time(s)": 0,
        # "avg_wait_on_data_time(s)": 0,
    })
    search_pattern = os.path.join(folder_path, '**', 'metrics.csv')
    for metrics_csv in glob.iglob(search_pattern, recursive=True):
        csv_data = convert_csv_to_dict(metrics_csv, first_epoch_only)
        metrics["num_jobs"] += 1
        metrics["total_batches"] += len(csv_data["Batch Index"])
        if "Batch Size" in csv_data:
            metrics["total_samples"] += sum(csv_data["Batch Size"])
        else:
            metrics["total_samples"] += (len(csv_data["Batch Index"]) * 32) #batch size was 32

            metrics["total_tokens"] += sum(csv_data["Batch Size (Tokens)"])
 
        # metrics["total_samples"] += sum(csv_data["Batch Size"])
        metrics["total_time(s)"] += sum(csv_data["Iteration Time (s)"])
        metrics["wait_on_data_time(s)"] += sum(csv_data["Iteration Time (s)"]) - sum(csv_data["GPU Processing Time (s)"])
        metrics["gpu_processing_time(s)"] += sum(csv_data["GPU Processing Time (s)"])
        metrics["data_fetch_time(s)"] += sum(csv_data["Data Load Time (s)"])
        metrics["transformation_time(s)"] += sum(csv_data["Transformation Time (s)"])
        metrics["cache_hits"] += sum(csv_data["Cache_Hits (Samples)"])
    
    # metrics["avg_gpu_time(s)"] = metrics["gpu_processing_time(s)"] / metrics["total_batches"]
    # metrics["avg_data_fetch_time(s)"] = metrics["data_fetch_time(s)"] / metrics["total_batches"]
    # metrics["avg_transformation_time(s)"] = metrics["transformation_time(s)"] / metrics["total_batches"]
    # metrics["avg_wait_on_data_time(s)"] = metrics["wait_on_data_time(s)"] / metrics["total_batches"]

    if metrics['num_jobs'] > 0:
        for key in ['total_time(s)', "wait_on_data_time(s)", "gpu_processing_time(s)", "data_fetch_time(s)", "transformation_time(s)"]:
            metrics[key] = metrics[key] / metrics['num_jobs']
        
        # metrics["throughput(batches/s)"] = metrics["total_batches"] / metrics["total_time(s)"]
        metrics["throughput(samples/s)"] = metrics["total_samples"] / metrics["total_time(s)"]
        
        metrics["cache_hit(%)"] = metrics["cache_hits"] / metrics["total_samples"]
        metrics["compute_time(%)"] = metrics["gpu_processing_time(s)"] / metrics["total_time(s)"]
        metrics["waiting_on_data_time(%)"] = metrics["wait_on_data_time(s)"] / metrics["total_time(s)"]

        transform_percent = metrics["transformation_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_fetch_time(s)"])
        data_fetch_percent = metrics["data_fetch_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_fetch_time(s)"])
        # metrics["transform_time(%)"] = metrics["transformation_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_fetch_time(s)"])
        # metrics["data_fetch_time(%)"] = metrics["data_fetch_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_fetch_time(s)"])
        metrics["transform_delay(%)"] = transform_percent *  metrics["waiting_on_data_time(%)"] 
        metrics["data_fetch_delay(%)"] = data_fetch_percent *  metrics["waiting_on_data_time(%)"] 
    
    return metrics

def compute_ec2_costs(instance_type: str, time_seconds: float):
    instance_prices = {
        't2.medium':  0.0464,
        'p3.8xlarge':  12.24,
        'c5n.xlarge': 0.4,
        'cache.t3.medium':  0.068,
        'cache.m7g.4xlarge': 1.257,
        'cache.m7g.xlarge': 0.315,
        'cache.m7g.12xlarge': 3.016,
        'cache.m5.2xlarge': 0.4984,

    }
    #  # Convert seconds to hours
    # if 'cache' in instance_type:
    #     hours = max(time_seconds / 3600, 1)
    # else:
    hours = time_seconds / 3600
    hourly_rate = instance_prices[instance_type]
    instance_cost = hourly_rate * hours
    return instance_cost





def get_cost_summary(folder_path, exp_duration, exp_thrpughput, cache_size_gb, averge_data_transfer_per_request_kb):
    metrics = OrderedDict({
         "total_lambda_cost": 0,
         "prefetch_lambda_cost": 0,
         "sion_cache_lambda_cost": 0,
         "sion_cache_proxy_cost": 0,
         "training_compute_cost": 0,
         "redis_cache_cost": 0,
         "total_cost": 0
    })
    metrics["training_compute_cost"] = compute_ec2_costs('p3.8xlarge', exp_duration)

    if 'super' in folder_path:
        search_pattern = os.path.join(folder_path, '**', 'bill.csv')
        for cost_csv in glob.iglob(search_pattern, recursive=True):
            #comute data loading costs
            csv_data = convert_csv_to_dict(cost_csv)
            systems = list(csv_data["System"])

            for idx, system in enumerate(systems):
                if 'InfiniSore' in system:
                    metrics["sion_cache_lambda_cost"] += csv_data["Total Cost"][idx]
                elif 'PREFETCH' in system:
                    metrics["prefetch_lambda_cost"] += csv_data["Total Cost"][idx]
                metrics["total_lambda_cost"] += csv_data["Total Cost"][idx]

            metrics["sion_cache_proxy_cost"] = compute_ec2_costs('t2.medium', exp_duration)
            metrics["total_cost"] = metrics["total_lambda_cost"] + metrics["training_compute_cost"] + metrics["sion_cache_proxy_cost"]
    
    # elif 'pytorch' in folder_path:
    #     if 'cifar10' in folder_path:
    #         metrics["redis_cache_cost"] = compute_serverless_redis_costs(exp_duration, 1, num_samples, 2.3)
    #     elif 'imagenet' in folder_path:
    #         metrics["redis_cache_cost"] = compute_serverless_redis_costs(exp_duration, 43, num_samples, 116)
    #     else:
    #          ValueError("Unknown dataset")
    #     metrics["total_cost"] = metrics["training_compute_cost"] + metrics["redis_cache_cost"]
    else:
        metrics["redis_cache_cost"] = compute_serverless_redis_costs(exp_duration,1,exp_thrpughput,200)
        metrics["total_cost"] = metrics["training_compute_cost"] + metrics["redis_cache_cost"]

    return metrics

if __name__ == "__main__":
    folder_path = "C:\\Users\\pw\\Desktop\\dataloading_gpu_cache_sizes_results\\cifar10_vit"
    base_name = os.path.basename(os.path.normpath(folder_path))
    exp_names = get_subfolder_names(folder_path, include_children = False)
    for kind in ['after_first_epoch']: #'first_epoch', 'after_first_epoch'
        overall_summary = []
        for exp in exp_names:
            exp_summary  = {}
            exp_summary['name'] = exp
            exp_path = os.path.join(folder_path, exp)
            train_summary = get_training_summary(exp_path, kind)
            exp_summary.update(train_summary)
            cost_summary = get_cost_summary(
                folder_path=exp_path,
                exp_duration =  train_summary["total_time(s)"],
                exp_thrpughput =  train_summary["throughput(samples/s)"],
                cache_size_gb = 30,
                averge_data_transfer_per_request_kb=200)
            exp_summary.update(cost_summary)
            exp_summary.update(scale_data_for_epoch(exp_summary, folder_path))
            save_dict_list_to_csv([exp_summary], os.path.join(exp_path, f'{exp}_{kind}__summary.csv'))
            overall_summary.append(exp_summary)

        save_dict_list_to_csv(overall_summary, os.path.join(folder_path, f'{base_name}_{kind}_overall_summary.csv'))



    # subfolders = glob.glob(os.path.join(folder_path, '*'))
    
    # overall_summary = []

    # for subfolder in subfolders:
    #     csv_data = convert_all_csv_to_dict_vision(subfolder)
    #     #csv_data = convert_all_csv_to_dict_vision(subfolder)
    #     folder_name = os.path.basename(os.path.normpath(subfolder))
    #     overall_summary.append(csv_data)  # Collect the folder summary
    #     print(csv_data)

    # # Save all folder summaries to a single CSV file
    # save_dict_list_to_csv(overall_summary, os.path.join(folder_path, 'overall_summary.csv'))
