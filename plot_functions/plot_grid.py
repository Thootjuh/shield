import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns  # Added seaborn
from collections import defaultdict
import glob
import pandas as pd
import numpy as np
import math

def read_data_from_directory(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    combined_data = pd.DataFrame()

    for file in csv_files:
        print(f"Reading file: {file}")
        data = pd.read_csv(file)
        data = data.rename(columns={'baseline_perf': 'pi_b_perf'})
        data = data.rename(columns={'nb_trajectories': 'length_trajectory'})
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

def extract_data(data):
    relevant_data_new = data[['method', 'length_trajectory', 'method_perf', 'run_time', 'pi_b_perf', 'pi_star_perf']]
    return relevant_data_new

def group_by_methods(data):
    method_groups = defaultdict(list)
    
    for method in data['method'].unique():
        base_method = method.split('shield-')[-1]
        method_groups[base_method].append(method)
    
    grouped_dfs = []
    for base_method, methods in method_groups.items():
        grouped_dfs.append(data[data['method'].isin(methods)])
    
    return grouped_dfs

def calculate_cvar(data, alpha=0.01):
    cvar_results = []
    for method in data['method'].unique():
        method_data = data[data['method'] == method]
        for traj in method_data['length_trajectory'].unique():
            traj_data = method_data[method_data['length_trajectory'] == traj]['method_perf']
            if len(traj_data) > 0:
                threshold = np.percentile(traj_data, alpha * 100)
                cvar = traj_data[traj_data <= threshold].mean()
                std = traj_data[traj_data <= threshold].std()
                cvar_results.append({'method': method, 'length_trajectory': traj, 'cvar': cvar, 'std': std})
    return pd.DataFrame(cvar_results)

def plot_all_methods_avg(data, env_name, ax):
    grouped_data = data.groupby(['method', 'length_trajectory']).agg(
        method_perf_mean=('method_perf', 'mean'),
        method_perf_std=('method_perf', 'std')  # calculate std deviation
    ).reset_index()
    
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
    cmap = plt.cm.get_cmap('tab20', 20)
    colors = [cmap(1), cmap(0), cmap(3), cmap(2)]
    color_map = {}
    for idx, base_method in enumerate(color_methods):
        base_color = colors[idx]
        color_map[base_method] = base_color
    
    for method in sorted_methods:
        method_data = grouped_data[grouped_data['method'] == method]
        x = method_data['length_trajectory']
        y = method_data['method_perf_mean']
        yerr = method_data['method_perf_std']
        
        if method == 'shielded_baseline':
            method_data = grouped_data[grouped_data['method'] == method]
            ax.plot(x, y,
                    label=method, linestyle='--', color="black")
        else:  
            marker = 'x' if 'shield-' in method else 's'
            method_data = grouped_data[grouped_data['method'] == method]
            color = color_map[method]
            ax.errorbar(x, y, yerr=yerr, label=method, linestyle='-', color=color, marker=marker, capsize=4)

            # CVar plotting
            # method_data_raw = data[data['method'] == method]
            # if env_name == 'Pacman':
            #     cvar_data = calculate_cvar(method_data_raw, 0.1)
            # else:
            #     cvar_data = calculate_cvar(method_data_raw)
 
            # ax.errorbar(cvar_data['length_trajectory'], cvar_data['cvar'], yerr=cvar_data['std'],
            #         label=method+' (cvar)', linestyle='--', color=color, marker=marker, capsize=4)

    if 'pi_star_perf' in data.columns:
        grouped = data.groupby('length_trajectory')['pi_star_perf'].mean().reset_index()
        ax.plot(grouped['length_trajectory'], grouped['pi_star_perf'], linestyle=':', color='#656565', label='optimal policy')
    if 'pi_b_perf' in data.columns:
        grouped = data.groupby('length_trajectory')['pi_b_perf'].mean().reset_index()
        ax.plot(grouped['length_trajectory'], grouped['pi_b_perf'], linestyle='dashdot', color='#656565', label='baseline policy')
        
    ax.set_xscale('log')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Avg. Performance')
    ax.set_title(f'{env_name}')
    ax.grid(True)
        
def plot_all_methods(data, env_name, ax):
    grouped_data = data.groupby(['method', 'length_trajectory'])
    grouped_data = grouped_data.method_perf.mean().reset_index()
    
    sorted_methods = sorted(grouped_data['method'].unique(), key=lambda x: (x.replace('shield-', ''), 'shield-' in x))

    color_methods = [m for m in sorted_methods if m != 'shielded_baseline']
    unshielded = [m for m in color_methods if not m.startswith('shield-')]
    shielded = [m for m in color_methods if m.startswith('shield-')]
    shielded_dict = {m.replace('shield-', ''): m for m in shielded}
    color_methods = []
    for method in sorted(unshielded):
        color_methods.append(method)
        if method in shielded_dict:
            color_methods.append(shielded_dict[method])
    
    cmap = plt.cm.get_cmap('tab20', 20)
    colors = [cmap(1), cmap(0), cmap(3), cmap(2)]
    color_map = {}
    for idx, base_method in enumerate(color_methods):
        base_color = colors[idx]
        color_map[base_method] = base_color
    
    for method in sorted_methods:
        if method == 'shielded_baseline':
            method_data = grouped_data[grouped_data['method'] == method]
            ax.plot(method_data['length_trajectory'], method_data['method_perf'],
                    label=method, linestyle='--', color="black")
        else:  
            marker = 'x' if 'shield-' in method else 's'
            method_data = grouped_data[grouped_data['method'] == method]
            color = color_map[method]
            ax.plot(method_data['length_trajectory'], method_data['method_perf'],
                    label=method, linestyle='-', color=color, marker=marker, markersize=4)

            method_data_raw = data[data['method'] == method]
            cvar_data = calculate_cvar(method_data_raw)
            ax.plot(cvar_data['length_trajectory'], cvar_data['cvar'],
                    label=method+' (cvar)', linestyle='--', color=color, marker=marker, markersize=4)

    if 'pi_star_perf' in data.columns:
        grouped = data.groupby('length_trajectory')['pi_star_perf'].mean().reset_index()
        ax.plot(grouped['length_trajectory'], grouped['pi_star_perf'], linestyle=':', color='#656565', label='optimal policy')
    if 'pi_b_perf' in data.columns:
        grouped = data.groupby('length_trajectory')['pi_b_perf'].mean().reset_index()
        ax.plot(grouped['length_trajectory'], grouped['pi_b_perf'], linestyle='dashdot', color='#656565', label='baseline policy')
    
    ymin, ymax = ax.get_ylim()
    if env_name == 'Wet Chicken':
        ax.set_ylim(bottom=-60, top=ymax)
    elif env_name == 'Random MDPs':
        ax.set_ylim(bottom=-4, top=ymax)   
    elif env_name == 'Frozen Lake':
        ax.set_ylim(bottom=-5, top=ymax)
    ax.set_xscale('log')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Performance')
    ax.set_title(f'{env_name}')
    ax.grid(True)

def main(parent_directory):
    # Seaborn style settings
    sns.set_context("paper", font_scale=2.5)  # Can be adjusted (e.g., "paper", "poster")
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    plt.rcParams['font.family'] = 'serif'
    subdirs = sorted([os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))])
    
    if len(subdirs) != 4:
        print("Error: Expected exactly 4 subdirectories in the provided parent directory.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, subdir in enumerate(subdirs):
        data = read_data_from_directory(subdir)
        data = extract_data(data)
        data_list = group_by_methods(data)
        
        plot_all_methods(data, environments[idx], axes[idx])
        
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ['DUIPI', 'DUIPI (CVaR)', 'Shielded-DUIPI', 'Shielded-DUIPI (CVaR)', 
              'SPIBB', 'SPIBB (CVaR)', 'Shielded-SPIBB', 'Shielded-SPIBB (CVaR)', 
              'Shielded Baseline', 'Optimal', 'Baseline']
    # labels = ['Shielded Baseline', 'Optimal', 'Baseline',
    #             'DUIPI (CVaR)', 'Shielded-DUIPI (CVaR)', 
    #             'SPIBB (CVaR)', 'Shielded-SPIBB (CVaR)', 
    #           ]
    # labels = ['Shielded Baseline', 'Optimal', 'Baseline',
    #             'DUIPI', 'Shielded-DUIPI', 
    #             'SPIBB', 'Shielded-SPIBB', 
    #           ]
    
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize='15', title='Methods')
    fig.suptitle("Method performance plotted against dataset size")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.subplots_adjust(bottom=0.23)
    plt.savefig("results_grid_plot.png", format='png')
    plt.savefig("results_grid_plot.pdf", format='pdf')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_grid.py <parent_directory>")
        sys.exit(1)
    environments = ["Random MDPs","Wet Chicken", "Pacman", "Frozen Lake"]
    parent_dir = sys.argv[1]
    main(parent_dir)
