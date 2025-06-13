import pandas as pd
import sys


directory_path = sys.argv[1]

df = pd.read_csv(directory_path)

# Define the method name you want to remove
# method_to_remove = ["R_min", 'shield-R_min', 'DUIPI_bayesian', 'shield-DUIPI_bayesian', 'RaMDP', 'shield-RaMDP', 'WorstCaseRMDP', 'shield-WorstCaseRMDP']
# method_to_remove = sys.argv[2]
# Filter out the rows where "method" is not equal to the unwanted method
# df = df[df["method"] != method_to_remove]
df = df.assign(pi_star_perf='5.403600876626367')
# Save the filtered data back to a CSV
df.to_csv(directory_path, index=False)