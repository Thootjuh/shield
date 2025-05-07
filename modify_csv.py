import pandas as pd
import sys


directory_path = sys.argv[1]

df = pd.read_csv(directory_path)

# Define the method name you want to remove
# method_to_remove = ["R_min", 'shield-R_min', 'DUIPI_bayesian', 'shield-DUIPI_bayesian', 'RaMDP', 'shield-RaMDP', 'WorstCaseRMDP', 'shield-WorstCaseRMDP']
method_to_remove = sys.argv[2]
method_to_keep =sys.argv[2]
# Filter out the rows where "method" is not equal to the unwanted method
df = df[df["hyperparam"] == int(method_to_keep)]

# Save the filtered data back to a CSV
df.to_csv(directory_path, index=False)