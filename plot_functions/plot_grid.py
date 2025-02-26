import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import glob
from collections import defaultdict

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

def plot_all_methods(ax, data, title, color):
    """Plots data on a given subplot axis."""
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data = grouped_data.method_perf.mean().reset_index()
    base_methods = set(method.split('shield-')[-1] for method in data['method'].unique())
    
    color_map = {}
    for idx, base_method in enumerate(sorted(base_methods)):
        base_color = color(idx)
        color_map[base_method] = base_color
        
    sorted_methods = sorted(data['method'].unique(), key=lambda x: (x.replace('shield-', ''), 'shield-' in x))
    
    legend_handles = []
    legend_labels = []
    for method in sorted_methods:
        base_method = method.split('shield-')[-1]
        method_data = data[data['method'] == method]
        linestyle = '--' if 'shield-' in method else '-'
        color = color_map[base_method]
        
        line, = ax.plot(method_data['length_trajectory'], method_data['method_perf'],
                 label=method, linestyle=linestyle, color=color)
        legend_handles.append(line)
        legend_labels.append(method)
    
    ax.set_xlabel('Length Trajectory')
    ax.set_ylabel('Avg. Performance')
    ax.set_title(title)
    ax.grid(True)
    
    return legend_handles, legend_labels


def get_subdirectories(parent_directory):
    """Finds exactly 5 subdirectories inside a given parent directory."""
    subdirectories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory)
                      if os.path.isdir(os.path.join(parent_directory, d))]
    
    if len(subdirectories) != 5:
        raise ValueError(f"Expected 5 subdirectories, but found {len(subdirectories)} in {parent_directory}")
    
    return subdirectories  # Sort for consistency

def plot_multiple_directories(parent_directory, output_filename):
    """Generates a 3x2 grid plot from data in 5 subdirectories."""
    directories = get_subdirectories(parent_directory)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))  # 3 rows, 2 columns
    axes = axes.flatten()

    data_list = []

    for directory in directories:
        data = read_data_from_directory(directory)
        column_names = data.columns
        data = extract_data(data)
        data_list.append(data)        
        
    grouped_data = data.groupby(['method', 'length_trajectory'])
    data = grouped_data.method_perf.mean().reset_index()
    base_methods = set(method.split('shield-')[-1] for method in data['method'].unique())
    color = plt.cm.get_cmap('tab10', len(base_methods))
    for i, (directory, data) in enumerate(zip(directories, data_list)):
        handles, labels = plot_all_methods(axes[i], data, title=environments[i], color=color)
        
    # Last subplot: Create a legend and caption
    legend_ax = axes[-1]
    legend_ax.axis('off')  # Hide axes

    # Create legend
    legend_ax.legend(handles, labels, title="Method Legend", loc="center")

    # # Caption text
    # legend_ax.text(0.5, 0.1, "Comparison of average method performance\nacross multiple datasets",
    #                fontsize=12, ha='center')

    # Save and show
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()
    
    
environments = ["Aircraft Collision Avoidance", "Wet Chicken", "Slippery Grid World", "Random MDPs", "Pacman"]
parent_directory = sys.argv[1]
plot_multiple_directories(parent_directory, "combined_results.png")
