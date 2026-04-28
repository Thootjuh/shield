import pandas as pd
import matplotlib.pyplot as plt
import configparser
import os
import glob
import numpy as np
import sys


def read_data(filepath): 
    new_data = pd.read_csv(filepath)
    new_data = new_data.rename(columns={'nb_trajectories': 'length_trajectory'})
    return new_data


def read_data_from_directory(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    combined_data = pd.DataFrame()

    for file in csv_files:
        print(f"Reading file: {file}")
        data = pd.read_csv(file)
        data = data.rename(columns={'nb_trajectories': 'length_trajectory'})
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    return combined_data


def extract_data(data):
    return data[['method', 'length_trajectory', 'success_rate', 'failure_rate', 'avoid_rate', 'run_time']]

def get_color_map(methods):
    methods = sorted(methods)
    cmap = plt.cm.get_cmap('tab20', len(methods))
    return {method: cmap(i) for i, method in enumerate(methods)}


def plot_data(data, filename, color_map):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data_succ_rate = grouped_data.success_rate.mean().reset_index()
    data_fail_rate = grouped_data.failure_rate.mean().reset_index()

    plt.figure(figsize=(12, 8))

    for method in data_succ_rate['method'].unique():
        color = color_map[method]

        method_data = data_succ_rate[data_succ_rate['method'] == method]
        plt.plot(
            method_data['length_trajectory'],
            method_data['success_rate'],
            label=method + "_success",
            color=color,
            linestyle='-'
        )

        method_data = data_fail_rate[data_fail_rate['method'] == method]
        plt.plot(
            method_data['length_trajectory'],
            method_data['failure_rate'],
            label=method + "_failure",
            color=color,
            linestyle='--'
        )

    plt.xlabel('Length Trajectory')
    plt.ylabel('Probability of satisfying the specification')
    plt.title('Success & Failure probability vs. Length Trajectory')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_succ_rate(data, filename, color_map):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data_succ_rate = grouped_data.success_rate.mean().reset_index()

    plt.figure(figsize=(12, 8))
    for method in data_succ_rate['method'].unique():
        method_data = data_succ_rate[data_succ_rate['method'] == method]
        plt.plot(
            method_data['length_trajectory'],
            method_data['success_rate'],
            label=method,
            color=color_map[method]
        )

    plt.xlabel('Length Trajectory')
    plt.ylabel('Probability of satisfying the specification')
    plt.title('Success probability vs. Length Trajectory')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_failure_rate(data, filename, color_map):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data_fail_rate = grouped_data.failure_rate.mean().reset_index()

    plt.figure(figsize=(12, 8))
    for method in data_fail_rate['method'].unique():
        method_data = data_fail_rate[data_fail_rate['method'] == method]
        plt.plot(
            method_data['length_trajectory'],
            method_data['failure_rate'],
            label=method,
            color=color_map[method]
        )

    plt.xlabel('Length Trajectory')
    plt.ylabel('Probability of satisfying the specification')
    plt.title('Failure probability vs. Length Trajectory')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_avoid_rate(data, filename, color_map):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data_avoid_rate = grouped_data.avoid_rate.mean().reset_index()

    plt.figure(figsize=(12, 8))
    for method in data_avoid_rate['method'].unique():
        method_data = data_avoid_rate[data_avoid_rate['method'] == method]
        plt.plot(
            method_data['length_trajectory'],
            method_data['avoid_rate'],
            label=method,
            color=color_map[method]
        )

    plt.xlabel('Length Trajectory')
    plt.ylabel('Probability of satisfying the specification')
    plt.title('Avoid probability vs. Length Trajectory')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


# MAIN
if len(sys.argv) > 1:
    directory_path = sys.argv[1]
else:
    directory_path = "/internship/code/results/wet_chicken/shield/wet_chicken_results/"

data = read_data_from_directory(directory_path)
data = extract_data(data)
color_map = get_color_map(data['method'].unique())

# plot_data(data, 'results_succ_fail_rate.png', color_map)
plot_succ_rate(data, 'results_succ_rate.png', color_map)
plot_failure_rate(data, 'results_fail_rate.png', color_map)
plot_avoid_rate(data, 'results_avoid_rate.png', color_map)