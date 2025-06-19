import pandas as pd
import sys
import os


folder_path = sys.argv[1]

# folder_path = 'path/to/your/folder'  # Replace with your folder path

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Remove '_bayesian' from 'method' column
        if 'method' in df.columns:
            df['method'] = df['method'].astype(str).str.replace('_bayesian', '', regex=False)
        
        # Save back to the same file
        df.to_csv(file_path, index=False)

print("Done updating all CSV files.")
# df = pd.read_csv(directory_path)

# # Define the method name you want to remove
# # method_to_remove = ["R_min", 'shield-R_min', 'DUIPI_bayesian', 'shield-DUIPI_bayesian', 'RaMDP', 'shield-RaMDP', 'WorstCaseRMDP', 'shield-WorstCaseRMDP']
# # method_to_remove = sys.argv[2]
# # Filter out the rows where "method" is not equal to the unwanted method
# # df = df[df["method"] != method_to_remove]
# df = df.assign(pi_star_perf='5.403600876626367')
# # Save the filtered data back to a CSV
# df.to_csv(directory_path, index=False)