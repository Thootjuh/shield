import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import pandas as pd
import numpy as np

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
        # print(data.columns)
        data = data.rename(columns={'baseline_perf': 'pi_b_perf'})
        data = data.rename(columns={'nb_trajectories': 'length_trajectory'})
        combined_data = pd.concat([combined_data, data], ignore_index=True)
        # print(data.columns)
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

def plot_all_methods(data, env_name, ax):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    grouped_data = grouped_data.method_perf.mean().reset_index()
    
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
    color_map = {}
    for idx, base_method in enumerate(color_methods):
        base_color = colors(idx)
        color_map[base_method] = base_color
    
    for method in sorted_methods:
        if method == 'shielded_baseline':
            method_data = grouped_data[grouped_data['method'] == method]
            ax.plot(method_data['length_trajectory'], method_data['method_perf'],
                    label=method, linestyle='--', color='black')
        else:
            method_data = grouped_data[grouped_data['method'] == method]
            color = color_map[method]
            ax.plot(method_data['length_trajectory'], method_data['method_perf'],
                    label=method, linestyle='-', color=color)
            
            # CVar plotting
            method_data_raw = data[data['method'] == method]
            cvar_data = calculate_cvar(method_data_raw)
            ax.plot(cvar_data['length_trajectory'], cvar_data['cvar'],
                    label=method+'_cvar', linestyle='--', color=color)

    if 'pi_star_perf' in data.columns:
        grouped = data.groupby('length_trajectory')['pi_star_perf'].mean().reset_index()
        ax.plot(grouped['length_trajectory'], grouped['pi_star_perf'], linestyle=':', color='black', label='optimal policy')
    if 'pi_b_perf' in data.columns:
        grouped = data.groupby('length_trajectory')['pi_b_perf'].mean().reset_index()
        ax.plot(grouped['length_trajectory'], grouped['pi_b_perf'], linestyle='dashdot', color='black', label='baseline policy')
    
    ax.set_xscale('log')
    ax.set_xlabel('Length Trajectory')
    ax.set_ylabel('Avg Method Performance')
    ax.set_title(f'{env_name}')
    ax.grid(True)

def main(parent_directory):
    subdirs = sorted([os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))])
    
    if len(subdirs) != 4:
        print("Error: Expected exactly 4 subdirectories in the provided parent directory.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, subdir in enumerate(subdirs):
        # env_name = os.path.basename(subdir)
        data = read_data_from_directory(subdir)
        data = extract_data(data)
        data_list = group_by_methods(data)  # If needed by `calculate_cvar`
        
        plot_all_methods(data, environments[idx], axes[idx])
        
    # Legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize='small', title='Methods')

    plt.subplots_adjust(bottom=0.15)  # Make space at the bottom for the legend
    # plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leaves space at bottom and top
    plt.savefig("results_grid_plot.pdf", format='pdf')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_grid.py <parent_directory>")
        sys.exit(1)
    environments = ["Random MDPs","Wet Chicken", "Pacman", "Frozen Lake"]
    parent_dir = sys.argv[1]
    main(parent_dir)
