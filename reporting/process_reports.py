import glob
import pandas as pd
import os
from collections import OrderedDict
import csv

def convert_csv_to_dict(csv_file, start_timestamp = None, end_timestamp = None):
    df = pd.read_csv(csv_file)
    if 'bill.csv' in csv_file:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Filter the DataFrame based on the timestamp range
        filtered_df = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
        return filtered_df.to_dict(orient='list')
    # Filter the rows where 'Epoch Index' is equal to 1

    # if df[df['Epoch Index'] > 1].empty:
    #         filtered_df = df[df['Epoch Index'] == 1]
    # else:
    #         filtered_df = df[df['Epoch Index'] > 1]

    return df.to_dict(orient='list')

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


def get_training_summary(folder_path):
    start_time_stamp = None
    end_time_stamp = None
    metrics = OrderedDict({
         "num_jobs": 0,
         "total_batches": 0,
         "total_samples": 0,
         "total_tokens": 0,
         "max_cached_batches": 0,
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
        csv_data = convert_csv_to_dict(metrics_csv)
        if not start_time_stamp or csv_data['Timestamp (UTC)'][0] < start_time_stamp:
            #convert to datetime
            start_time_stamp = csv_data['Timestamp (UTC)'][0]
        if not end_time_stamp or csv_data['Timestamp (UTC)'][-1] > end_time_stamp:
            end_time_stamp = csv_data['Timestamp (UTC)'][-1]
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
        if max(csv_data["Cache_Size"]) > metrics["max_cached_batches"]:
            metrics["max_cached_batches"] = max(csv_data["Cache_Size"])
    
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
    
    return metrics, start_time_stamp, end_time_stamp

if __name__ == "__main__":

    paths = ["C:\\Users\\pw\\Desktop\\image_classification\\coordl\\cifar10",
             "C:\\Users\\pw\\Desktop\\vision transformer\\coordl\\cifar10",
              "C:\\Users\\pw\\Desktop\\vision transformer\\coordl\\imagenet"
            ]
    
    for folder_path in paths:
        base_name = os.path.basename(os.path.normpath(folder_path))
        exp_names = get_subfolder_names(folder_path, include_children = False)
        overall_summary = []
        
        for exp in exp_names:
                exp_summary  = {}
                exp_summary['name'] = exp
                exp_path = os.path.join(folder_path, exp)
                train_summary, start_timestamp, end_timestamp = get_training_summary(exp_path)
                exp_summary.update(train_summary)
                save_dict_list_to_csv([exp_summary], os.path.join(exp_path, f'{exp}__summary.csv'))
                overall_summary.append(exp_summary)

        save_dict_list_to_csv(overall_summary, os.path.join(folder_path, f'{base_name}_overall_summary.csv'))