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

    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory by Method')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_positive_data(data, filename, color_map):
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

    plt.ylim(bottom=0)
    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory by Method')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_data_interval(data, filename, color_map):
    grouped_data = data.groupby(['method', 'length_trajectory'])

    summary = grouped_data['method_perf'].agg(
        mean='mean',
        lower=lambda x: np.percentile(x, 10),
        upper=lambda x: np.percentile(x, 90)
    ).reset_index()

    plt.figure(figsize=(12, 8))
    for method in summary['method'].unique():
        method_data = summary[summary['method'] == method]
        color = color_map[method]

        plt.plot(
            method_data['length_trajectory'],
            method_data['mean'],
            label=method,
            color=color
        )
        plt.fill_between(
            method_data['length_trajectory'],
            method_data['lower'],
            method_data['upper'],
            color=color,
            alpha=0.2
        )

    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory by Method')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_positive_data_interval(data, filename, color_map):
    grouped_data = data.groupby(['method', 'length_trajectory'])

    summary = grouped_data['method_perf'].agg(
        mean='mean',
        lower=lambda x: np.percentile(x, 10),
        upper=lambda x: np.percentile(x, 90)
    ).reset_index()

    plt.figure(figsize=(12, 8))
    for method in summary['method'].unique():
        method_data = summary[summary['method'] == method]
        color = color_map[method]

        plt.plot(
            method_data['length_trajectory'],
            method_data['mean'],
            label=method,
            color=color
        )
        plt.fill_between(
            method_data['length_trajectory'],
            method_data['lower'],
            method_data['upper'],
            color=color,
            alpha=0.2
        )

    plt.ylim(bottom=0)
    plt.xlabel('Length Trajectory')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Length Trajectory by Method')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def calculate_cvar(data, alpha=0.01):
    cvar_results = []
    for method in data['method'].unique():
        method_data = data[data['method'] == method]
        for traj in method_data['length_trajectory'].unique():
            traj_data = method_data[method_data['length_trajectory'] == traj]['method_perf']
            if len(traj_data) > 0:
                threshold = np.percentile(traj_data, alpha * 100)
                cvar = traj_data[traj_data <= threshold].mean()
                cvar_results.append({
                    'method': method,
                    'length_trajectory': traj,
                    'cvar': cvar
                })
    return pd.DataFrame(cvar_results)


def plot_cvar(cvar_data, filename, color_map):
    plt.figure(figsize=(12, 8))
    for method in cvar_data['method'].unique():
        method_data = cvar_data[cvar_data['method'] == method]
        plt.plot(
            method_data['length_trajectory'],
            method_data['cvar'],
            label=method,
            color=color_map[method]
        )

    plt.xlabel('Length Trajectory')
    plt.ylabel('1%-CVaR')
    plt.title('1%-CVaR vs. Length Trajectory by Method')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_avg_perf_vs_nb_states(data, filename, color_map):
    grouped_data = data.groupby(['method', 'nb_states'])
    data_mean = grouped_data.method_perf.mean().reset_index()

    plt.figure(figsize=(12, 8))
    for method in data_mean['method'].unique():
        method_data = data_mean[data_mean['method'] == method]
        plt.plot(
            method_data['nb_states'],
            method_data['method_perf'],
            label=method,
            color=color_map[method]
        )

    plt.xlabel('Number of States')
    plt.ylabel('Average Method Performance')
    plt.title('Average Performance vs. Number of States by Method')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_avg_perf_vs_nb_states_max_traj(data, filename, color_map):
    max_traj = data['length_trajectory'].max()
    print(f"Using length_trajectory = {max_traj}")

    filtered_data = data[data['length_trajectory'] == max_traj]

    grouped_data = filtered_data.groupby(
        ['method', 'nb_states']
    )['method_perf'].mean().reset_index()

    plt.figure(figsize=(12, 8))
    for method in grouped_data['method'].unique():
        method_data = grouped_data[grouped_data['method'] == method]
        plt.plot(
            method_data['nb_states'],
            method_data['method_perf'],
            label=method,
            color=color_map[method]
        )

    plt.xlabel('Number of States')
    plt.ylabel('Average Method Performance')
    plt.title(f'Average Performance vs. Number of States (length_trajectory = {max_traj})')
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

plot_data(data, 'Results.png', color_map)
plot_positive_data(data, 'Results_positive.png', color_map)
plot_data_interval(data, 'Results_interval.png', color_map)
plot_positive_data_interval(data, 'Results_positive_Interval.png', color_map)
plot_avg_perf_vs_nb_states(data, 'Results_vs_nb_states.png', color_map)
plot_avg_perf_vs_nb_states_max_traj(data, 'Results_vs_nb_states_max_traj.png', color_map)

cvar = calculate_cvar(data)
plot_cvar(cvar, "Results_CVAR.png", color_map)