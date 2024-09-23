import numpy as np
from numba import cuda
from perception_Traj_init import dims,Nsize



# Placeholder for method1 and method2, updated for higher dimensions
@cuda.jit(device=True)
def method1(indices):

    return

@cuda.jit(device=True)
def method2(indices):
    return np.prod(np.array(indices))  # Simplified; replace with actual logic

# Function to convert flattened 1D index to multi-dimensional indices
def flat_index_to_multi_dim(flat_index, dims):
    indices = []
    for dim in reversed(dims):
        indices.append(flat_index % dim)
        flat_index //= dim
    return tuple(reversed(indices))

# GPU kernel to calculate the matrix
@cuda.jit
def calculate_matrix(d_matrix, dims):
    i = cuda.grid(1)  # Get 1D global thread index
    if i < d_matrix.size:
        # Convert flattened index i into multi-dimensional indices
        multi_indices = flat_index_to_multi_dim(i, dims)

        # Perform the matrix calculation using multiple methods
        d_matrix[i] = method1(multi_indices) + method2(multi_indices)

# Initialize the flattened matrix on the host (CPU)
matrix = np.zeros(Nsize, dtype=np.float32)

# Allocate memory on the device (GPU)
d_matrix = cuda.to_device(matrix)

# Define grid and block size for CUDA (adjustable based on GPU)
threads_per_block = 256
blocks_per_grid = (Nsize + threads_per_block - 1) // threads_per_block

# Launch the kernel to calculate the matrix in parallel on the GPU
calculate_matrix[blocks_per_grid, threads_per_block](d_matrix, dims)

# Copy the result back from device (GPU) to host (CPU)
matrix = d_matrix.copy_to_host()

# Reshape the flattened matrix back to 8D form
matrix_reshaped = matrix.reshape(dims)

# Print a small part of the result for verification
print(matrix_reshaped[0, 0, 0, 0, 0, 0, 0, 0])  # Adjust for debugging purposes
