import cv2 as cv
import numpy as np
from sklearn.neighbors import NearestNeighbors                                                       #the goattt
import matplotlib.pyplot as plt
from pivlib.cv import ransac
from mpl_toolkits.mplot3d import Axes3D

RANSAC_ITER = 1000
STEPSIZE = 10

def feature_extraction(img):
    stepsize = 1
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
    #cv.waitKey(stepsize)  # Adjust the wait time to control the speed of the video
    
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


def featureMatching(frame1: np.ndarray, frame2: np.ndarray, distance_threshold: float):
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

    # Get the frame indice with the largest feature set
    frame = (frame1, frame2)
    largest = frame1.shape[0] < frame2.shape[0]

    # Create a NearestNeighbors model
    knn_model = NearestNeighbors(n_neighbors=1)
    knn_model.fit(frame[largest][:,2:])

    # Use kneighbors to find the nearest neighbor indices
    distances, indices = knn_model.kneighbors(frame[~largest][:,2:], n_neighbors=1)
    indices = indices.flatten()

    print(f"number of matches: {len(indices)}")

    # Match largest feature space with smallest feature space indices
    if largest == 0:
        keypoints1, keypoints2 = frame[0][:, :2][indices], frame[1][:, :2]
    else:
        keypoints1, keypoints2 = frame[0][:, :2], frame[1][:, :2][indices]

    # Filter out matches based on distance
    filteredindices = np.where(distances.flatten() < distance_threshold)[0]

    # Ensure that indices are within bounds
    filteredindices = filteredindices[filteredindices < len(indices)]
    
    keypoints1, keypoints2 = keypoints1[filteredindices], keypoints2[filteredindices]
    
    filtereddistances = distances[filteredindices]

    std_dev = float(np.std(filtereddistances))

    return keypoints1, keypoints2, std_dev

def triangulatePoints(P0, P1, pts0, pts1):
    Xs = []
    for (p, q) in zip(pts0.T, pts1.T):
        # Solve 'AX = 0'
        A = np.vstack((p[0] * P0[2] - P0[0], p[1] * P0[2] - P0[1], q[0] * P1[2] - P1[0], q[1] * P1[2] - P1[1]))
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        Xs.append(Vt[-1])
    return np.vstack(Xs).T

def main():
    # image1 = cv.imread('PCRegistration/rgb_image1_3.png')
    # image2 = cv.imread('PCRegistration/rgb_image2_3.png')
    image1 = cv.imread('PB/PB8/cubo1.jpg')
    image2 = cv.imread('PB/PB8/cubo2.jpg')
    print(f"image1 shape: {image1}")
    print(f"image2 shape: {image2.shape}")

    keypoints1 = feature_extraction(image1)
    keypoints2 = feature_extraction(image2)

    print(f"keypoints1 shape: {keypoints1.shape}")
    print(f"keypoints2 shape: {keypoints2.shape}")

    keypoints1, keypoints2, std_dev = featureMatching(keypoints1.T, keypoints2.T, 100)

    print(f"MATCHED keypoints1 shape: {keypoints1.shape}")
    print(f"MATCHED keypoints2 shape: {keypoints2.shape}")

    # _, inliers = ransac(keypoints1, keypoints2, RANSAC_ITER, 1)

    # keypoints1, keypoints2 = keypoints1[inliers], keypoints2[inliers]
    # print(f"number of inliers: {len(keypoints1)}")

    f, cx, cy = 1000., 320., 240.
    # Estimate relative pose of two view
    F, _ = cv.findFundamentalMat(keypoints1, keypoints2, cv.FM_8POINT)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    E = K.T @ F @ K
    _, R, t, _ = cv.recoverPose(E, keypoints1, keypoints2)
    
    # Reconstruct 3D points (triangulation)
    P0 = K @ np.eye(3, 4, dtype=np.float32)
    Rt = np.hstack((R, t))
    P1 = K @ Rt
    X_4d = triangulatePoints(P0, P1, keypoints1.T, keypoints2.T)
    X = cv.convertPointsFromHomogeneous(X_4d.T).reshape(-1, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', marker='o', label='3D Points')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.legend()
    plt.show()

    print(f"X shape: {X.shape}")

    # Write the reconstructed 3D points
    np.savetxt('3d_points.txt', X)

    pass



if __name__ == "__main__":
    main()