import numpy as np


def pad(img: np.ndarray, left_border: int=0, right_border: int=0, top_border: int=0, bottom_border: int=0) -> np.ndarray:
    """
    Add borders to an image.
    
    Parameters:
    -
    - img: Image, a 2D or 3D NumPy array.
    - left_border: Width of the left border.
    - right_border: Width of the right border.
    - top_border: Height of the top border.
    - bottom_border: Height of the bottom border.
    
    Returns:
    -
    - img: Image with borders, a 2D or 3D NumPy array.
    """
    # Add borders to the image
    img = np.pad(img, ((top_border, bottom_border), (left_border, right_border), (0, 0)), mode="constant") # type: ignore
    
    return img

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
        raise ValueError("At least 4 points are required")

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
        if num_inliers > best_num_inliers: # type: ignore
            best_num_inliers = num_inliers
            best_H = H
            best_mask = inliers

    # Return the best homography and the mask of inliers
    return best_H, best_mask

############## UNTESTED (made by copilot) ##############
def warpAndStitch(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Warp and stitch two images using a homography matrix.
    
    Parameters:
    -
    - img1: First image, a 2D or 3D NumPy array.
    - img2: Second image, a 2D or 3D NumPy array.
    - H: Homography matrix, a 3x3 matrix representing the transformation from img1 to img2.

    Returns:
    -
    - img: Stitched image, a 2D or 3D NumPy array.
    """
    
    
    
    # Get the dimensions of the images
    width1, height1 = img1.shape[:2]
    width2, height2 = img2.shape[:2]
    
    # Get the corners of the images
    corners1 = np.array([[0, 0], [0, height1], [width1, height1], [width1, 0]])
    corners2 = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]])
    
    # Warp the corners of the images to find their positions in the stitched image
    corners1_warped = np.dot(H, np.hstack((corners1, np.ones((4, 1))))).T
    corners1_warped = corners1_warped[:, :2] / corners1_warped[:, 2:]
    
    # Find the minimum and maximum x and y coordinates of the warped corners
    x_min = int(np.min(corners1_warped[:, 0]))
    x_max = int(np.max(corners1_warped[:, 0]))
    y_min = int(np.min(corners1_warped[:, 1]))
    y_max = int(np.max(corners1_warped[:, 1]))
    
    # Compute the size of the stitched image
    width = x_max - x_min
    height = y_max - y_min
    
    # Compute the translation matrix to shift the image
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
    # Warp the first image
    img1_warped = warpPerspective(img1, np.dot(T, H), (width, height))
    
    # Warp the second image
    img2_warped = warpPerspective(img2, T, (width, height))
    
    # Blend the two images
    img = addWeighted(img1_warped, 0.5, img2_warped, 0.5)

    return img
