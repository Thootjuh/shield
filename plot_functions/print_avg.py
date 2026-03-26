import pandas as pd
import glob
import os
import sys

def read_data_from_directory(directory_path):
    """
    Reads and aggregates all CSV files from a given directory.
    """
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    combined_data = pd.DataFrame()

    for file in csv_files:
        print(f"Reading file: {file}")
        data = pd.read_csv(file)
        data = data.rename(columns={'baseline_perf': 'pi_b_perf'})
        data = data.rename(columns={'nb_trajectories': 'length_trajectory'})
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    return combined_data


# --- Read command line arguments ---
if len(sys.argv) < 4:
    print("Usage: python print_avg.py <directory_path> <nb_trajectories> <column>")
    sys.exit(1)

directory_path = sys.argv[1]
nb_trajectories_value = int(sys.argv[2])
column = sys.argv[3]

# --- Load data ---
df = read_data_from_directory(directory_path)

# --- Filter for the specified number of trajectories ---
df_filtered = df[df["length_trajectory"] == nb_trajectories_value]

# --- Compute average performance per method ---
avg_perf = (
    df_filtered.groupby("method", as_index=False)[column]
    .mean()
)

print(f"\nAverage performance for nb_trajectories = {nb_trajectories_value}:\n")
print(avg_perf)