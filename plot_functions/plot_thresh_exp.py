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
    return data[['method', 'length_trajectory', 'method_perf', 'run_time', 'threshold']]

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

def plot_cvar(cvar_data, filename):
    plt.figure(figsize=(12, 8))
    for method in cvar_data['method'].unique():
        method_data = cvar_data[cvar_data['method'] == method]
        plt.plot(method_data['length_trajectory'], method_data['cvar'], label=method)

    plt.xlabel('Length Trajectory')
    plt.ylabel('1%-CVaR')
    plt.title('1%-CVaR vs. Length Trajectory')
    plt.legend(title='Method')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_shield_threshold_avg(data, filename):
    shield = data[data["method"] == "shield-SPIBB"]
    spibb = data[data["method"] == "SPIBB"]

    # Shield lines: one per threshold
    shield_grouped = shield.groupby(["threshold", "length_trajectory"])["method_perf"].mean().reset_index()
    
    # SPIBB line: aggregate over runs per trajectory
    spibb_grouped = spibb.groupby("length_trajectory")["method_perf"].mean().reset_index()

    plt.figure(figsize=(12, 8))

    # plot shield-SPIBB curves per threshold
    for thr in sorted(shield_grouped["threshold"].unique()):
        thr_data = shield_grouped[shield_grouped["threshold"] == thr]
        plt.plot(
            thr_data["length_trajectory"],
            thr_data["method_perf"],
            marker="o",
            label=f"shield-SPIBB (threshold={thr})"
        )

    # plot SPIBB line
    plt.plot(
        spibb_grouped["length_trajectory"],
        spibb_grouped["method_perf"],
        linestyle="--",
        linewidth=3,
        label="SPIBB"
    )

    plt.xlabel("Length Trajectory")
    plt.ylabel("Average Method Performance")
    plt.title("Performance vs Length Trajectory (Shield Thresholds + SPIBB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()




def plot_shield_threshold_interval(data, filename):
    shield = data[data["method"] == "shield-SPIBB"]
    spibb = data[data["method"] == "SPIBB"]

    # Shield stats by threshold trajectory
    grouped = shield.groupby(["threshold", "length_trajectory"])["method_perf"]
    summary = grouped.agg(
        mean="mean",
        lower=lambda x: np.percentile(x, 10),
        upper=lambda x: np.percentile(x, 90),
    ).reset_index()

    # SPIBB stats
    spibb_grouped = spibb.groupby("length_trajectory")["method_perf"].agg(
        mean="mean",
        lower=lambda x: np.percentile(x, 10),
        upper=lambda x: np.percentile(x, 90)
    ).reset_index()

    plt.figure(figsize=(12, 8))

    # shield-SPIBB curves per threshold
    for thr in sorted(summary["threshold"].unique()):
        thr_data = summary[summary["threshold"] == thr]
        plt.plot(
            thr_data["length_trajectory"],
            thr_data["mean"],
            marker="o",
            label=f"shield-SPIBB (threshold={thr})"
        )
        plt.fill_between(
            thr_data["length_trajectory"],
            thr_data["lower"],
            thr_data["upper"],
            alpha=0.2
        )

    # SPIBB line + band
    plt.plot(
        spibb_grouped["length_trajectory"],
        spibb_grouped["mean"],
        linestyle="--",
        linewidth=3,
        label="SPIBB"
    )
    plt.fill_between(
        spibb_grouped["length_trajectory"],
        spibb_grouped["lower"],
        spibb_grouped["upper"],
        alpha=0.15
    )

    plt.xlabel("Length Trajectory")
    plt.ylabel("Method Performance")
    plt.title("Performance Interval vs Length Trajectory (Shield Thresholds + SPIBB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()



def plot_shield_threshold_cvar(data, filename, alpha=0.01):
    shield = data[data["method"] == "shield-SPIBB"]
    spibb = data[data["method"] == "SPIBB"]

    # Compute CVaR for shield per threshold per trajectory
    rows = []
    for thr in sorted(shield["threshold"].unique()):
        thr_data = shield[shield["threshold"] == thr]
        for traj in sorted(thr_data["length_trajectory"].unique()):
            vals = thr_data[thr_data["length_trajectory"] == traj]["method_perf"]
            cutoff = np.percentile(vals, alpha * 100)
            cvar = vals[vals <= cutoff].mean()
            rows.append((thr, traj, cvar))
    shield_cvar = pd.DataFrame(rows, columns=["threshold", "length_trajectory", "cvar"])

    # Compute CVaR for standard SPIBB
    spibb_rows = []
    for traj in sorted(spibb["length_trajectory"].unique()):
        vals = spibb[spibb["length_trajectory"] == traj]["method_perf"]
        cutoff = np.percentile(vals, alpha * 100)
        cvar = vals[vals <= cutoff].mean()
        spibb_rows.append((traj, cvar))
    spibb_cvar = pd.DataFrame(spibb_rows, columns=["length_trajectory", "cvar"])

    plt.figure(figsize=(12, 8))

    # shield-SPIBB curves
    for thr in sorted(shield_cvar["threshold"].unique()):
        thr_data = shield_cvar[shield_cvar["threshold"] == thr]
        plt.plot(
            thr_data["length_trajectory"],
            thr_data["cvar"],
            marker="o",
            label=f"shield-SPIBB (threshold={thr})"
        )

    # SPIBB curve
    plt.plot(
        spibb_cvar["length_trajectory"],
        spibb_cvar["cvar"],
        linestyle="--",
        linewidth=3,
        label="SPIBB"
    )

    plt.xlabel("Length Trajectory")
    plt.ylabel(f"{int(alpha*100)}%-CVaR")
    plt.title(f"CVaR vs Length Trajectory (Shield Thresholds + SPIBB, α={alpha})")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_shield_threshold_avg_clipped(data, filename, ymin=None, ymax=None):
    shield = data[data["method"] == "shield-SPIBB"]
    spibb = data[data["method"] == "SPIBB"]

    # Shield lines: one per threshold
    shield_grouped = shield.groupby(["threshold", "length_trajectory"])["method_perf"].mean().reset_index()

    # SPIBB line
    spibb_grouped = spibb.groupby("length_trajectory")["method_perf"].mean().reset_index()

    plt.figure(figsize=(12, 8))

    # Shield-SPIBB curves per threshold
    for thr in sorted(shield_grouped["threshold"].unique()):
        thr_data = shield_grouped[shield_grouped["threshold"] == thr]
        plt.plot(
            thr_data["length_trajectory"],
            thr_data["method_perf"],
            marker="o",
            label=f"shield-SPIBB (threshold={thr})"
        )

    # SPIBB curve
    plt.plot(
        spibb_grouped["length_trajectory"],
        spibb_grouped["method_perf"],
        linestyle="--",
        linewidth=3,
        label="SPIBB"
    )

    # === Apply clipping if provided ===
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    elif ymin is not None:
        plt.ylim(bottom=ymin)
    elif ymax is not None:
        plt.ylim(top=ymax)

    plt.xlabel("Length Trajectory")
    plt.ylabel("Average Method Performance")
    plt.title("Shield-SPIBB vs SPIBB (Average Performance, Clipped)")
    plt.legend(title="Method / Threshold")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


# ================================================================
# === MAIN EXECUTION
# ================================================================

if len(sys.argv) > 1:
    directory_path = sys.argv[1]
else:
    directory_path = "/internship/code/results/wet_chicken/shield/wet_chicken_results/"

data = read_data_from_directory(directory_path)
data = extract_data(data)

# Original plots


cvar = calculate_cvar(data)
# plot_cvar(cvar, "Results_CVAR.png")

# New threshold-based plots
plot_shield_threshold_avg(data, "Results_threshold_avg.png")
plot_shield_threshold_interval(data, "Results_threshold_interval.png")
plot_shield_threshold_cvar(data, "Results_threshold_CVAR.png")
plot_shield_threshold_avg_clipped(data, "Results_shield_threshold_avg_clipped.png", ymin=10)
