import pandas as pd
import matplotlib.pyplot as plt
import configparser
import os
import glob
import numpy as np
import sys
import seaborn as sns

sns.set_theme(
    style="whitegrid",
    context="paper",
    font="sans-serif",
    rc={
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    },
)

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
    return data[['method', 'length_trajectory', 'method_perf', 'run_time', 'nb_states']]


def get_color_map(methods):
    methods = sorted(methods)
    cmap = plt.cm.get_cmap('tab20', len(methods))
    return {method: cmap(i) for i, method in enumerate(methods)}


def plot_data(data, filename, color_map):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data = grouped_data.method_perf.mean().reset_index()

    plt.figure(figsize=(12, 8))
    for method in data['method'].unique():
        method_data = data[data['method'] == method]
        plt.plot(
            method_data['length_trajectory'],
            method_data['method_perf'],
            label=method,
            color=color_map[method]
        )

    plt.xlabel('Number of Trajectories')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance of the SPIBB-MRL method')
    # plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    
# MAIN
directory_path = sys.argv[1]

# Optional method filter
method_filter = sys.argv[2] if len(sys.argv) > 2 else None

data = read_data_from_directory(directory_path)
data = extract_data(data)

if method_filter is not None:
    data = data[data['method'] == method_filter]

    if data.empty:
        print(f"No data found for method '{method_filter}'")
        sys.exit(1)

color_map = get_color_map(data['method'].unique())

plot_data(data, 'Results.png', color_map)