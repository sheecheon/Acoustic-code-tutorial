import numpy as np

# 5-dimensional array (e.g., 2x3x4x5x6 array)
arr = np.random.randint(0, 100, (2, 3, 4, 5, 6))

# Sort indices along the fourth axis (axis=3)
sorted_indices = np.argsort(arr, axis=3)

# Get indices for the smallest and second smallest values
smallest_indices = sorted_indices[..., 0]
second_smallest_indices = sorted_indices[..., 1]

# To obtain the values, you need to use broadcasting and advanced indexing
# Create a tuple of indices to fetch the corresponding values

# Dimensions for the axes not being sorted
dim0, dim1, dim2, dim4 = np.meshgrid(
    np.arange(arr.shape[0]), 
    np.arange(arr.shape[1]), 
    np.arange(arr.shape[2]), 
    np.arange(arr.shape[4]),
    indexing='ij'
)

# Expand dimensions for advanced indexing
smallest_values = arr[dim0, dim1, dim2, smallest_indices, dim4]
second_smallest_values = arr[dim0, dim1, dim2, second_smallest_indices, dim4]

print("Smallest values:\n", smallest_values)
print("Indices of smallest values:\n", smallest_indices)
print("Second smallest values:\n", second_smallest_values)
print("Indices of second smallest values:\n", second_smallest_indices)

