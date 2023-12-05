import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt


def findHomography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Find the homography matrix using Direct Linear Transform (DLT).

    Parameters:
    - src_pts: Source points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    - dst_pts: Destination points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    
    Returns:
    - H: Homography matrix, a 3x3 matrix representing the transformation from src_pts to dst_pts.
    """

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

def ransac(src_pts: np.ndarray, dst_pts: np.ndarray, n_iter: int, inlier_threshold: float):
    """
    Find the homography matrix using RANSAC.

    Parameters:
    - src_pts: Source points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    - dst_pts: Destination points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    - n_iter: Number of iterations.
    - inlier_threshold: Threshold to determine if a point is an inlier.

    Returns:
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
        inliers = error < inlier_threshold
        num_inliers = np.sum(inliers)

        # Update the best homography if the current homography has more inliers
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_H = H
            best_mask = inliers

    # Return the best homography and the mask of inliers
    return best_H, best_mask

def warpPerspective(src: np.ndarray, H: np.ndarray, dst_size) -> np.ndarray:
    """
    Apply a perspective transformation (warp) to an image using a homography matrix.

    Parameters:
    - src: Source image, a 2D or 3D NumPy array.
    - H: Homography matrix, a 3x3 matrix representing the perspective transformation.
    - dst_size: Size of the destination image, specified as (width, height).

    Returns:
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

mat_file = scipy.io.loadmat('PB\PB6\Class6_ransac\keypoint_pairs.mat')

# Print the keys (variable names) in the MATLAB file
print("Variables in the MATLAB file:")
print(mat_file.keys())

""" # Access specific variables and print their content
for variable_name in mat_file.keys():
    print(f"\nContent of variable '{variable_name}':")
    print(mat_file[variable_name]) """

u1 = np.array(mat_file['U1']).flatten()
v1 = np.array(mat_file['V1']).flatten()
u2 = np.array(mat_file['U2']).flatten()
v2 = np.array(mat_file['V2']).flatten()

# Create a list of corresponding points
points1 = np.array([u1, v1]).T
points2 = np.array([u2, v2]).T

# Display both images side by side and use the keypoint pairs to draw lines between them

# Load the images
image1 = plt.imread('PB\PB6\Class6_ransac\homogLAB_do_this\parede1.jpg')
image2 = plt.imread('PB\PB6\Class6_ransac\homogLAB_do_this\parede2.jpg')

# Create an empty image to concatenate the two images side by side
concatenated_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)

# Copy the images into the concatenated image
concatenated_image[:image1.shape[0], :image1.shape[1]] = image1
concatenated_image[:image2.shape[0], image1.shape[1]:] = image2

# Compute the homography using RANSAC
homography, mask = ransac(points2, points1, n_iter=1000, inlier_threshold=10)

src_pts = points1[mask]
dst_pts = points2[mask]

# Compute the homography matrix using the 4 points
H = findHomography(dst_pts, src_pts)

mask = np.bool_(mask.flatten())

# Draw lines between corresponding keypoints
for p1, p2 in zip(points1, points2):
    # Shift the x-coordinate for the second image since it's concatenated next to the first image
    p2[0] += image1.shape[1]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='green', linewidth=1)

# Show the concatenated image with lines
plt.imshow(concatenated_image)
plt.show()

# Draw the inliers (which are the points that were used to compute the homography)
inlier_image = concatenated_image.copy()

for p1, p2 in zip(src_pts, dst_pts):
    p2[0] += image1.shape[1]
    # Draw a circle at each inlier
    plt.scatter(*p1, color='red')
    plt.scatter(*p2, color='red')
    
    # Draw a line between the inliers
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=1)

# Show the image with inliers
plt.imshow(inlier_image)
plt.show()

print(image2.shape[:2][::-1])

dst = warpPerspective(image2, H, image2.shape[:2][::-1])
# Draw the transformed image side by side with the first image
with_transform = concatenated_image.copy()

# Copy the transformed image into the empty space
with_transform[:image1.shape[0], :image1.shape[1]] = image1
with_transform[:dst.shape[0], dst.shape[1]:] = dst

# Show the concatenated image with lines
plt.imshow(with_transform)
plt.show()

exit()