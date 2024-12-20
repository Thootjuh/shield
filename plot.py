import pandas as pd
import matplotlib.pyplot as plt
import configparser
import os
import glob
import numpy as np

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
    relevant_data_new = data[['method', 'length_trajectory', 'method_perf', 'run_time']]

    # Group by method and calculate the average performance for each length_trajectory
    return relevant_data_new

def plot_data(data, filename):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data = grouped_data.method_perf.mean().reset_index()
    # Plot average performance against length_trajectory for each method
    plt.figure(figsize=(12, 8))
    for method in data['method'].unique():
        method_data = data[data['method'] == method]
        plt.plot(method_data['length_trajectory'], method_data['method_perf'], label=method)

    # Set plot labels and legend
    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory by Method (New Dataset)')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_positive_data(data, filename):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data = grouped_data.method_perf.mean().reset_index()
    # Plot average performance against length_trajectory for each method
    plt.figure(figsize=(12, 8))
    for method in data['method'].unique():
        method_data = data[data['method'] == method]
        plt.plot(method_data['length_trajectory'], method_data['method_perf'], label=method)

    # Set plot labels and legend
    plt.ylim(0, 1)
    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory by Method (New Dataset)')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def calculate_cvar(data, alpha=0.01):
    # Calculate CVaR for each method and trajectory
    cvar_results = []
    for method in data['method'].unique():
            method_data = data[data['method'] == method]
            for traj in method_data['length_trajectory'].unique():
                traj_data = method_data[method_data['length_trajectory'] == traj]['method_perf']
                if len(traj_data) > 0:
                    threshold = np.percentile(traj_data, alpha * 100)  # Find the 1% threshold
                    cvar = traj_data[traj_data <= threshold].mean()  # Calculate the mean of the bottom 1%
                    cvar_results.append({'method': method, 'length_trajectory': traj, 'cvar': cvar})
    return pd.DataFrame(cvar_results)

def plot_cvar(cvar_data, filename):
   # Plot CVaR against trajectory for each method
    plt.figure(figsize=(12, 8))
    for method in cvar_data['method'].unique():
        method_data = cvar_data[cvar_data['method'] == method]
        plt.plot(method_data['length_trajectory'], method_data['cvar'], label=method)

    # Set plot labels and legend
    plt.xlabel('Length Trajectory')
    plt.ylabel('1%-CVaR (Conditional Value at Risk)')
    plt.title('1%-CVaR vs. Length Trajectory by Method')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
   

#TODO: make it so that you can pass this path as an argument in the commandline
directory_path = "/internship/code/results/random_mdps/shielded/mod_baseline_only/experiment_results/"
data = read_data_from_directory(directory_path)
data = extract_data(data)
plot_data(data, 'Baseline.png')
plot_positive_data(data, 'BaselinePositive.png')
cvar = calculate_cvar(data)
plot_cvar(cvar, "BaselineCvar")

