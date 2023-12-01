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

print(np.shape(u1))

# Display both images side by side and use the keypoint pairs to draw lines between them

# Load the images
image1 = plt.imread('PB\PB6\Class6_ransac\homogLAB_do_this\parede1.jpg')
image2 = plt.imread('PB\PB6\Class6_ransac\homogLAB_do_this\parede2.jpg')

# Create an empty image to concatenate the two images side by side
concatenated_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)

# Copy the images into the concatenated image
concatenated_image[:image1.shape[0], :image1.shape[1]] = image1
concatenated_image[:image2.shape[0], image1.shape[1]:] = image2

# Draw lines between corresponding keypoints
for x1, y1, x2, y2 in zip(u1, v1, u2, v2):
    # Shift the x-coordinate for the second image since it's concatenated next to the first image
    x2 += image1.shape[1]
    cv2.line(concatenated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green line

# Display the concatenated image with lines
# cv2.imshow('Concatenated Image with Lines', concatenated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Compute the homography using RANSAC
# Create a list of corresponding points
points1 = np.array([u1, v1]).T
points2 = np.array([u2, v2]).T

print(np.shape(points1))
print(points1)

# Compute the homography using RANSAC
homography, mask = ransac(points1, points2, n_iter=1000, inlier_threshold=10)

src_pts = points1[mask]
dst_pts = points2[mask]

# Compute the homography matrix using the 4 points
H = findHomography(src_pts, dst_pts)

# Compute the error using all points
src_pts_hom = np.hstack((src_pts, np.ones((len(src_pts), 1))))
dst_pts_hom = np.hstack((dst_pts, np.ones((len(dst_pts), 1))))
dst_pts_hom_hat = np.dot(H, src_pts_hom.T).T

mask = np.bool_(mask.flatten())
print(mask.shape)

# Print the mask
print("\nMask:")
print(mask)

# Draw the inliers (which are the points that were used to compute the homography)
inlier_image = concatenated_image.copy()

for p1, p2 in zip(points1[mask], points2[mask]):
    p2[0] += image1.shape[1]
    # Draw a circle at each inlier
    cv2.circle(inlier_image, tuple(map(int,p1)), 5, (0, 0, 255), -1)  # Red circle
    cv2.circle(inlier_image, tuple(map(int,p2)), 5, (0, 0, 255), -1)  # Red circle

    # Draw a line between the inliers
    cv2.line(inlier_image, tuple(map(int,p1)), tuple(map(int,p2)), (0, 0, 255), 1)  # Red line

# Display the image with the inliers
cv2.imshow('Image with Inliers', inlier_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()