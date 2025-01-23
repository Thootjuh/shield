import glob
import os
import pandas as pd
import numpy as np
import sys
import re 
from collections import defaultdict
import matplotlib.pyplot as plt


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


def group_by_methods(data):
    method_groups = defaultdict(list)
    
    for method in data['method'].unique():
        base_method = method.split('shield-')[-1]  # Extract base method name
        method_groups[base_method].append(method)
    
    grouped_dfs = []
    for base_method, methods in method_groups.items():
        grouped_dfs.append(data[data['method'].isin(methods)])
    
    return grouped_dfs

def plot_data_interval(data, filename, method_name):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    
    # Calculate mean, lower, and upper bounds of the 90% confidence interval
    summary = grouped_data['method_perf'].agg(
        mean='mean',
        lower=lambda x: np.percentile(x, 10),  # Lower 5th percentile
        upper=lambda x: np.percentile(x, 90) # Upper 95th percentile
    ).reset_index()
    
    # Plot average performance with confidence intervals
    plt.figure(figsize=(12, 8))
    for method in summary['method'].unique():
        method_data = summary[summary['method'] == method]
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
    plt.title(f'Average Performance vs. Length Trajectory for {method_name}')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    
def plot_data(data, filename, method_name):
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
    plt.title(f'Average Performance vs. Length Trajectory for {method_name}')
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

def plot_cvar(cvar_data, filename, method_name):
   # Plot CVaR against trajectory for each method
    plt.figure(figsize=(12, 8))
    for method in cvar_data['method'].unique():
        method_data = cvar_data[cvar_data['method'] == method]
        plt.plot(method_data['length_trajectory'], method_data['cvar'], label=method)

    # Set plot labels and legend
    plt.xlabel('Length Trajectory')
    plt.ylabel('1%-CVaR (Conditional Value at Risk)')
    plt.title(f'1%-CVaR vs. Length Trajectory for {method_name}')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    
def plot_all_methods(data, filename):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data = grouped_data.method_perf.mean().reset_index()
    
    base_methods = set(method.split('shield-')[-1] for method in data['method'].unique())
    colors = plt.cm.get_cmap('tab10', len(base_methods))
    
    plt.figure(figsize=(12, 8))
    color_map = {}
    for idx, base_method in enumerate(sorted(base_methods)):
        base_color = colors(idx)
        color_map[base_method] = base_color
    
    sorted_methods = sorted(data['method'].unique(), key=lambda x: (x.replace('shield-', ''), 'shield-' in x))
    
    for method in sorted_methods:
        base_method = method.split('shield-')[-1]
        method_data = data[data['method'] == method]
        linestyle = '--' if 'shield-' in method else '-'
        color = color_map[base_method]
        
        plt.plot(method_data['length_trajectory'], method_data['method_perf'],
                 label=method, linestyle=linestyle, color=color)
    
    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory for all methods')
    plt.legend(title='Method', loc='best')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
   
# Group by method/method-shield
directory_path = sys.argv[1]
data = read_data_from_directory(directory_path)
data = extract_data(data)
data_list = group_by_methods(data)
    
plot_all_methods(data, "results_all.png")
print(data_list[0])
for method in data_list:
    method_name = method.iloc[0]['method'] # This assumes that the first entry in the dataframe is the non-shielded variant, which is the case for the included experiments
    filename = "results_" + method_name + ".png"
    plot_data(method, filename, method_name)
    filename = "results_" + method_name + "_intervall.png"
    plot_data_interval(method, filename, method_name)
    filename = "results_" + method_name + "_cvar.png"
    cvar_data = calculate_cvar(method)
    plot_cvar(cvar_data, filename, method_name)
