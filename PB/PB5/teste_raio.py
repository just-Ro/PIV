from scipy import ndimage
import numpy as np
import cv2 as cv
from sklearn.neighbors import NearestNeighbors                                                       #the goattt

stepsize = 1

def padding(image, padding_size: int, color: int=255):

    # Create a larger canvas with padding
    canvas = np.ones((image.shape[0] + 2 * padding_size, image.shape[1] + 2 * padding_size, 3), dtype=np.uint8) * color

    # Place the image with padding on the canvas
    canvas[padding_size:padding_size + image.shape[0], padding_size:padding_size + image.shape[1]] = image
    
    return canvas

def findHomography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Find the homography matrix using Direct Linear Transform (DLT).

    Parameters:
    - src_pts: Source points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    - dst_pts: Destination points, a 2D array of shape (N, 2) representing (x, y) coordinates.
    
    Returns:
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

def addWeighted(src1: np.ndarray, alpha: float, src2: np.ndarray, beta: float, gamma: float = 0.0) -> np.ndarray:
    """
    Blends two images with specified weights and an optional bias.

    Parameters:
    - src1: First input array (image).
    - alpha: Weight of the first image elements.
    - src2: Second input array (image).
    - beta: Weight of the second image elements.
    - gamma: Scalar added to each sum (optional, default is 0.0).

    Returns:
    - dst: Resulting blended image.
    """
    # Ensure input arrays have the same shape
    assert src1.shape == src2.shape, "Input images must have the same shape"

    # Perform the weighted summation
    return np.clip(src1 * alpha + src2 * beta + gamma, 0, 255).astype(np.uint8)

def featureMatching(frame1: np.ndarray, frame2: np.ndarray):
    """
    Find the nearest neighbors between two sets of keypoints.
    Both sets of keypoints must have the same input shape \
        (num_features, num_descriptors) and the same number of descriptors.
    
    Parameters:
    -
    - frame1: Keypoints from the source image
    - frame2: Keypoints from the destination image
    
    Returns:
    -
    - keypoints1, keypoints2: A list of corresponding keypoint pairs
    """

    print(f"frame1: {frame1.shape}")
    print(f"frame2: {frame2.shape}")
    
    # Get the frame indice with the largest feature set
    frame = (frame1, frame2)
    largest = frame1.shape[0] < frame2.shape[0]

    # Create a NearestNeighbors model
    knn_model = NearestNeighbors(n_neighbors=1)
    knn_model.fit(frame[largest][:,2:])

    # Use kneighbors to find the nearest neighbor indices
    indices = knn_model.kneighbors(frame[~largest][:,2:], n_neighbors=1)[1].flatten()

    # Match largest feature space with smallest feature space indices
    if largest == 0:
        keypoints1, keypoints2 = frame[0][:, :2][indices], frame[1][:, :2]
    else:
        keypoints1, keypoints2 = frame[0][:, :2], frame[1][:, :2][indices]


    # Falta meter as imagens lado a lado para ver melhor(está a funcionar bem) 
    # Display the pairs
    """ print('Keypoint Pairs:')
    print('Image 1\tImage 2')
    for i in range(len(keypoints1)):
        print('({:.2f}, {:.2f})\t({:.2f}, {:.2f})'.format(keypoints1[i, 0], keypoints1[i, 1], keypoints2[i, 0], keypoints2[i, 1]))

    # Plotting for visualization
    plt.scatter(frame1[:, 0], frame1[:, 1], label='Keypoints in Image 1', marker='o', color='blue')
    plt.scatter(frame2[:, 0], frame2[:, 1], label='Keypoints in Image 2', marker='x', color='red')

    # Plot lines between corresponding keypoints
    for i in range(len(keypoints1)):
        plt.plot([frame1[i, 0], frame2[i, 0]], [frame1[i, 1], frame2[i, 1]], color='green')

    plt.legend()
    plt.title('Corresponding Keypoints between Images')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show() """

    return keypoints1, keypoints2

def feature_extraction(img):
    # Convert the image to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create a SIFT object
    sift = cv.SIFT.create()

    # detect keypoints and compute descriptors of the image
    keypoints, descriptors = sift.detectAndCompute(gray_img, None) # type: ignore

    # Draw keypoints on the image
    img_with_keypoints = cv.drawKeypoints(img, keypoints, img, flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)

    # Display the frame with keypoints
    cv.imshow('Frame with Keypoints', img_with_keypoints)
    cv.waitKey(stepsize)  # Adjust the wait time to control the speed of the video
    
    # Display the image with keypoints
    # plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.show()

    keypoints_coord = []
    # store the keypoints coordinates
    for point in keypoints:
        keypoints_coord.append(point.pt)

    keypoints_coord = np.array(keypoints_coord)
    
    descriptors = np.array(descriptors)

    features = np.append(keypoints_coord, descriptors, axis=1)
    features = np.transpose(features)

    return features


border = 100
######################
features1, features2 = feature_extraction(cv.imread('parede1.jpg')), feature_extraction(cv.imread('parede2.jpg'))

features1 = features1.T
features2 = features2.T
######################
c1, c2 = featureMatching(features1, features2)


im1 = padding(cv.imread('parede1.jpg'),border,0)
im2 = padding(cv.imread('parede2.jpg'),border,0)


c1 = c1 + border
c2 = c2 + border

# world = np.array([[0,0],[841,0],[0,1189],[841,1189]])*0.09 + 2*border

# H1 = findHomography(c1, world)
# H2 = findHomography(c2, world)

# Warp both images
# result_image1 = warpPerspective(im1, H1, (im1.shape[1], im1.shape[0]))
# result_image2 = warpPerspective(im2, H2, (im2.shape[1], im2.shape[0]))

# blended_result = addWeighted(result_image1, 0.5, result_image2, 0.5, 0)

for point in c1:
    cv.circle(im1, tuple(map(int, point)), 5, (0, 255, 0), -1)

for point in c2:
    cv.circle(im2, tuple(map(int, point)), 5, (0, 255, 0), -1)

cv.imshow('Image 1', im1)
cv.imshow('Image 2', im2)
# cv.imshow('Warped Image 1', result_image1)
# cv.imshow('Warped Image 2', result_image2)
# cv.imshow('Blended Result', blended_result)
cv.waitKey(0)
cv.destroyAllWindows()