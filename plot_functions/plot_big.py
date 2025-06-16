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
        print(data.columns)
        data = data.rename(columns={'baseline_perf': 'pi_b_perf'})
        data = data.rename(columns={'nb_trajectories': 'length_trajectory'})
        combined_data = pd.concat([combined_data, data], ignore_index=True)
        print(data.columns)
    return combined_data

def extract_data(data):
    # Extract relevant columns
    relevant_data_new = data[['method', 'length_trajectory', 'method_perf', 'run_time', 'pi_b_perf', 'pi_star_perf']]

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

def plot_all_methods(data, filename):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    grouped_data = grouped_data.method_perf.mean().reset_index()
    # base_methods = set(method.split('shield-')[-1] for method in grouped_data['method'].unique())
    sorted_methods = sorted(grouped_data['method'].unique(), key=lambda x: (x.replace('shield-', ''), 'shield-' in x))

    # Sort methods so they have the right color
    color_methods = [m for m in sorted_methods if m != 'shielded_baseline']
    unshielded = [m for m in color_methods if not m.startswith('shield-')]
    shielded = [m for m in color_methods if m.startswith('shield-')]
    shielded_dict = {m.replace('shield-', ''): m for m in shielded}
    color_methods = []
    for method in sorted(unshielded):
        color_methods.append(method)
        if method in shielded_dict:
            color_methods.append(shielded_dict[method])
    
    # set colors
    colors = plt.cm.get_cmap('tab20', 20)
    plt.figure(figsize=(12, 8))
    plt.xscale('log')
    color_map = {}
    for idx, base_method in enumerate(color_methods):
        base_color = colors(idx)
        color_map[base_method] = base_color
    
    
    sorted_methods = sorted(grouped_data['method'].unique(), key=lambda x: (x.replace('shield-', ''), 'shield-' in x))
    
    for method in sorted_methods:
        if method == 'shielded_baseline':
            base_method = method.split('shield-')[-1]
            method_data = grouped_data[grouped_data['method'] == method]
            linestyle = '--'
            color = 'black'
            plt.plot(method_data['length_trajectory'], method_data['method_perf'],
                 label=method, linestyle=linestyle, color=color)
        else:
            # Plot the normal performance
            base_method = method.split('shield-')[-1]
            method_data = grouped_data[grouped_data['method'] == method]
            linestyle = '-'
            color = color_map[method]
            
            plt.plot(method_data['length_trajectory'], method_data['method_perf'],
                    label=method, linestyle=linestyle, color=color)
            
            #plot the CVar performance
            method_data_raw = data[data['method'] == method]
            cvar_data = calculate_cvar(method_data_raw)
            linestyle = '--'
            plt.plot(cvar_data['length_trajectory'], cvar_data['cvar'],
                    label=method+'_cvar', linestyle=linestyle, color=color)
    
    # Plot optimal and baseline policy if present
    if 'pi_star_perf' in data.columns:
        grouped = data.groupby('length_trajectory')['pi_star_perf'].mean().reset_index()
        plt.plot(grouped['length_trajectory'], grouped['pi_star_perf'], linestyle=':', color='black', label='optimal policy')
    if 'pi_b_perf' in data.columns:
        grouped = data.groupby('length_trajectory')['pi_b_perf'].mean().reset_index()
        plt.plot(grouped['length_trajectory'], grouped['pi_b_perf'], linestyle='dashdot', color='black', label='baseline policy')
    
    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory for all methods on the '+ env_name +' environment')
    plt.legend(title='Method', loc='best')
    plt.grid(True)
    plt.savefig(filename, format='pdf')
    plt.show()
   
# Group by method/method-shield
directory_path = sys.argv[1]
# env_name = sys.argv[2]
env_name = "Random MDPs"
data = read_data_from_directory(directory_path)
data = extract_data(data)
print(data.columns)
data_list = group_by_methods(data)
    
plot_all_methods(data, "results_all_random_MDPs.pdf")
# plot_all_cvar(data, "results_all_cvar.png")
# for method in data_list:
#     method_name = method.iloc[0]['method'] # This assumes that the first entry in the dataframe is the non-shielded variant, which is the case for the included experiments
#     # filename = "results_" + method_name + ".png"
#     # plot_data(method, filename, method_name)
#     # filename = "results_" + method_name + "_intervall.png"
#     # plot_data_interval(method, filename, method_name)
#     filename = "results_" + method_name + "_cvar.png"
#     cvar_data = calculate_cvar(method)
#     plot_cvar(cvar_data, filename, method_name, method)