import numpy as np

def convert_to_intrinsic_matrix(intrinsics):
    fx, fy, cx, cy = intrinsics
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
    return intrinsic_matrix

# Your input array A
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Apply the function to each row of the array
intrinsic_matrices = np.array([convert_to_intrinsic_matrix(row) for row in A])

print(intrinsic_matrices)