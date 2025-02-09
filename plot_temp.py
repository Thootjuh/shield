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
    relevant_data_new = data[['method', 'length_trajectory', 'hyperparam', 'method_perf', 'run_time', ]]

    # Group by method and calculate the average performance for each length_trajectory
    return relevant_data_new

def plot_data(data, filename):
    grouped_data = data.groupby(['hyperparam', 'length_trajectory'])
    data = grouped_data.method_perf.mean().reset_index()
    # Plot average performance against length_trajectory for each method
    plt.figure(figsize=(12, 8))
    for method in data['hyperparam'].unique():
        method_data = data[data['hyperparam'] == method]
        plt.plot(method_data['length_trajectory'], method_data['method_perf'], label=method)

    # Set plot labels and legend
    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory by Method (New Dataset)')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_data_interval(data, filename):
    grouped_data = data.groupby(['hyperparam', 'length_trajectory'])
    
    # Calculate mean, lower, and upper bounds of the 90% confidence interval
    summary = grouped_data['method_perf'].agg(
        mean='mean',
        lower=lambda x: np.percentile(x, 1),  # Lower 5th percentile
        upper=lambda x: np.percentile(x, 99) # Upper 95th percentile
    ).reset_index()
    
    # Plot average performance with confidence intervals
    plt.figure(figsize=(12, 8))
    for method in summary['hyperparam'].unique():
        method_data = summary[summary['hyperparam'] == method]
        plt.plot(
            method_data['length_trajectory'], 
            method_data['mean'], 
            label=method
        )
        plt.fill_between(
            method_data['length_trajectory'], 
            method_data['lower'], 
            method_data['upper'], 
            alpha=0.2
        )

    # Set plot labels and legend
    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory by Method (New Dataset)')
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
plot_data(data, 'Results.png')
plot_data_interval(data, 'Results_interval.png')


