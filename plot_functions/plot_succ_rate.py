import pandas as pd
import matplotlib.pyplot as plt
import configparser
import os
import glob
import numpy as np
import sys


def read_data(filepath): 
    # Reload the new CSV file
    new_data = pd.read_csv(filepath)

    # Rename `nb_trajectories` to `length_trajectory` for consistency
    new_data = new_data.rename(columns={'nb_trajectories': 'length_trajectory'})
    return new_data

def read_data_from_directory(directory_path):
    # List all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    # Initialize an empty DataFrame
    combined_data = pd.DataFrame()

    # Read each CSV file and append to the combined DataFrame
    for file in csv_files:
        print(f"Reading file: {file}")
        data = pd.read_csv(file)
        # Rename `nb_trajectories` to `length_trajectory` for consistency
        data = data.rename(columns={'nb_trajectories': 'length_trajectory'})
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    return combined_data

def extract_data(data):
    # Extract relevant columns
    relevant_data_new = data[['method', 'length_trajectory', 'success_rate', 'failure_rate', 'run_time']]

    # Group by method and calculate the average performance for each length_trajectory
    return relevant_data_new

def plot_data(data, filename):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data_succ_rate = grouped_data.success_rate.mean().reset_index()
    data_avoid_rate = grouped_data.failure_rate.mean().reset_index()
    # Plot average performance against length_trajectory for each method
    plt.figure(figsize=(12, 8))
    for method in data_succ_rate['method'].unique():
        method_data = data_succ_rate[data_succ_rate['method'] == method]
        label = method+"_success_rate"
        plt.plot(method_data['length_trajectory'], method_data['success_rate'], label=label)
        method_data = data_avoid_rate[data_avoid_rate['method'] == method]
        label = method+"_failure_rate"
        plt.plot(method_data['length_trajectory'], method_data['failure_rate'], label=label)

    # Set plot labels and legend
    plt.xlabel('Length Trajectory')
    plt.ylabel('Probability of satisfying the specification')
    plt.title('Success probability vs. Length Trajectory')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_succ_rate(data, filename):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data_succ_rate = grouped_data.success_rate.mean().reset_index()
    # data_avoid_rate = grouped_data.failure_rate.mean().reset_index()
    # Plot average performance against length_trajectory for each method
    plt.figure(figsize=(12, 8))
    for method in data_succ_rate['method'].unique():
        method_data = data_succ_rate[data_succ_rate['method'] == method]
        label = method+"_success_rate"
        plt.plot(method_data['length_trajectory'], method_data['success_rate'], label=label)
        # method_data = data_avoid_rate[data_avoid_rate['method'] == method]
        # label = method+"_failure_rate"
        # plt.plot(method_data['length_trajectory'], method_data['failure_rate'], label=label)

    # Set plot labels and legend
    plt.xlabel('Length Trajectory')
    plt.ylabel('Probability of satisfying the specification')
    plt.title('Success probability vs. Length Trajectory')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    

def plot_failure_rate(data, filename):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    # data_succ_rate = grouped_data.success_rate.mean().reset_index()
    data_avoid_rate = grouped_data.failure_rate.mean().reset_index()
    # Plot average performance against length_trajectory for each method
    plt.figure(figsize=(12, 8))
    for method in data_avoid_rate['method'].unique():
        # method_data = data_succ_rate[data_succ_rate['method'] == method]
        # label = method+"_success_rate"
        # plt.plot(method_data['length_trajectory'], method_data['success_rate'], label=label)
        method_data = data_avoid_rate[data_avoid_rate['method'] == method]
        label = method+"_failure_rate"
        plt.plot(method_data['length_trajectory'], method_data['failure_rate'], label=label)

    # Set plot labels and legend
    plt.xlabel('Length Trajectory')
    plt.ylabel('Probability of satisfying the specification')
    plt.title('Success probability vs. Length Trajectory')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
   

# pass this path as an argument in the commandline
if len(sys.argv) > 1:
    directory_path = sys.argv[1]
else:
    directory_path = "/internship/code/results/wet_chicken/shield/wet_chicken_results/"
data = read_data_from_directory(directory_path)
data = extract_data(data)
plot_data(data, 'results_succ_fail_rate.png')
plot_succ_rate(data, 'results_succ_rate.png')
plot_failure_rate(data, 'results_fail_rate.png')