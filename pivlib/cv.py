import numpy as np


def findHomography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Find the homography matrix using Direct Linear Transform (DLT).

    Parameters:
    -
    - src_pts: Source points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    - dst_pts: Destination points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    
    Returns:
    -
    - H: Homography matrix, a 3x3 matrix representing the transformation from src_pts to dst_pts.
    """
    # Find homography matrix using Direct Linear Transform (DLT)

    # Check if the number of source and destination points is the same
    if len(src_pts) != len(dst_pts):
        raise ValueError("There must be the same number of source and destination points")
    
    # Check if there are at least 4 points to compute the homography
    if len(src_pts) < 4:
        raise ValueError("More than 4 points are required")

    # Build the matrix A for the homogeneous linear system
    A = []
    for src, dst in zip(src_pts, dst_pts):
        x, y = src
        u, v = dst
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)

    # Perform Singular Value Decomposition (SVD) on matrix A
    _, _, V = np.linalg.svd(A)
    
    # The homography matrix is the last column of V (right singular vector)
    H = V[-1, :].reshape((3, 3))

    # Normalize the homography matrix to ensure H[2, 2] is 1
    H /= H[2, 2]
    
    return H

def warpPerspective(src: np.ndarray, H: np.ndarray, dst_size) -> np.ndarray:
    """
    Apply a perspective transformation (warp) to an image using a homography matrix.

    Parameters:
    -
    - src: Source image, a 2D or 3D NumPy array.
    - H: Homography matrix, a 3x3 matrix representing the perspective transformation.
    - dst_size: Size of the destination image, specified as (width, height).

    Returns:
    -
    - dst: Warped image, a 2D or 3D NumPy array.
    """
    # Generate an empty image
    width, height = dst_size
    channel = src.shape[2] if src.ndim > 2 else 1
    dst = np.zeros((height, width, channel), dtype=src.dtype)
    
    # Iterate over the destination image
    for qy in range(height):
        for qx in range(width):
            # Calculate the inverse mapping using the homography matrix H
            p = np.dot(np.linalg.inv(H), [qx, qy, 1])
            
            px, py = int(p[0]/p[-1] + 0.5), int(p[1]/p[-1] + 0.5)
            
            # Check if the source pixel is within bounds
            if px >= 0 and py >= 0 and px < src.shape[1] and py < src.shape[0]:
                dst[qy, qx] = src[py, px]
    
    return dst

def addWeighted(src1: np.ndarray, alpha: float, src2: np.ndarray, beta: float, gamma: float = 0.0) -> np.ndarray:
    """
    Blends two images with specified weights and an optional bias.

    Parameters:
    -
    - src1: First input array (image).
    - alpha: Weight of the first image elements.
    - src2: Second input array (image).
    - beta: Weight of the second image elements.
    - gamma: Scalar added to each sum (optional, default is 0.0).

    Returns:
    -
    - dst: Resulting blended image.
    """
    # Ensure input arrays have the same shape
    assert src1.shape == src2.shape, "Input images must have the same shape"

    # Perform the weighted summation
    return np.clip(src1 * alpha + src2 * beta + gamma, 0, 255).astype(np.uint8)

def ransac(src_pts: np.ndarray, dst_pts: np.ndarray, n_iter: int, ransacReprojThreshold: float):
    """
    Find the homography matrix using RANSAC.

    Parameters:
    -
    - src_pts: Source points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    - dst_pts: Destination points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    - n_iter: Number of iterations.
    - ransacReprojThreshold: Threshold to determine if a point is an inlier.

    Returns:
    -
    - H: Homography matrix, a 3x3 matrix representing the transformation from src_pts to dst_pts.
    - mask: Mask of inliers, a 1D array of shape (N,) where 1 indicates that the point is an inlier.
    """
    
    best_num_inliers = 0
    best_H = np.array([])
    best_mask = np.array([])

    for _ in range(n_iter):
        # Randomly select 4 points from the source and destination points
        idx = np.random.choice(len(src_pts), 4)
        src_pts_4 = src_pts[idx]
        dst_pts_4 = dst_pts[idx]

        # Compute the homography matrix using the 4 points
        H = findHomography(src_pts_4, dst_pts_4)

        # Compute the error using all points
        src_pts_hom = np.hstack((src_pts, np.ones((len(src_pts), 1))))
        dst_pts_hom = np.hstack((dst_pts, np.ones((len(dst_pts), 1))))
        dst_pts_hom_hat = np.dot(H, src_pts_hom.T).T
        error = np.array(np.linalg.norm(dst_pts_hom - dst_pts_hom_hat, axis=1))

        # Compute the number of inliers
        inliers = error < ransacReprojThreshold
        num_inliers = np.sum(inliers)

        # Update the best homography if the current homography has more inliers
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_H = H
            best_mask = inliers

    # Return the best homography and the mask of inliers
    return best_H, best_mask

