import pandas as pd
import glob
import os
import sys

def read_data_from_directory(directory_path):
    """
    Reads and aggregates all CSV files from a given directory.

    Args:
        directory_path (str): Path to the directory containing CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame with all CSV data, with renamed columns for consistency.
    """
    # List all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    # Initialize an empty DataFrame
    combined_data = pd.DataFrame()

    # Read each CSV file and append to the combined DataFrame
    for file in csv_files:
        print(f"Reading file: {file}")
        data = pd.read_csv(file)
        # Rename `nb_trajectories` to `length_trajectory` for consistency
        data = data.rename(columns={'baseline_perf': 'pi_b_perf'})
        data = data.rename(columns={'nb_trajectories': 'length_trajectory'})
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data


# Load the CSV
directory_path = sys.argv[1]
df = read_data_from_directory(directory_path)

# Compute average method performance per method
avg_perf = (
    df.groupby("method", as_index=False)["method_perf"]
      .mean()
)

print(avg_perf)