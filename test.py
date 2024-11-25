import numpy as np

# Original 1D array
arr = np.array([[1, 2, 3], [4,5,6]])

# Add a new axis
arr_col = arr[:,:, np.newaxis]  # Transform to a column vector

print("Original shape:", arr.shape)  # (3,)
print("Column vector shape:", arr_col.shape)  # (3, 1)
print(arr_col)