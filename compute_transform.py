import numpy as np
import sys
from scipy.io import loadmat, savemat
from sklearn.neighbors import NearestNeighbors    #the goattt
from pivlib.cv import ransac, findHomography
from pivlib.config import Config
from pivlib.utils import Progress
from constants import *

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

    #print(f"number of matches: {len(indices)}")

    # Match largest feature space with smallest feature space indices
    if largest == 0:
        keypoints1, keypoints2 = frame[0][:, :2][indices], frame[1][:, :2]
    else:
        keypoints1, keypoints2 = frame[0][:, :2], frame[1][:, :2][indices]

    std_dev = float(np.std(distances))
    increment = 10

    # Filter out matches based on distance
    filteredindices = np.where(distances.flatten() < distance_threshold)[0]

    # Ensure that indices are within bounds
    filteredindices = filteredindices[filteredindices < len(indices)]
    
    while len(filteredindices) < 4:
        distance_threshold += increment
        
        if distance_threshold >= 1001:
            print(f"Warning: Retry limit exceeded. Number of matches: {len(filteredindices)}")
            return keypoints1, keypoints2, std_dev
        
        filteredindices = np.where(distances.flatten() < distance_threshold)[0]

        filteredindices = filteredindices[filteredindices < len(indices)]

    keypoints1, keypoints2 = keypoints1[filteredindices], keypoints2[filteredindices]
    
    filtereddistances = distances[filteredindices]  
    
    std_dev = float(np.std(filtereddistances))
    if std_dev == 0:
        std_dev = 0.0001
    
    # print(keypoints1.shape, keypoints2.shape, std_dev, filtereddistances.shape, filteredindices.shape)
    return keypoints1, keypoints2, std_dev

def findBestHomography(features1: np.ndarray, features2: np.ndarray):
    """ Find the best homography between two sets of keypoints """
    
    # MATCHING
    keypoints1, keypoints2, threshold = featureMatching(features1.T, features2.T, DISTANCE)    
    inliers = np.zeros(len(keypoints1), dtype=bool)
    # print(f"\nthreshold: {threshold}")

    # RANSAC
    _, inliers = ransac(keypoints1, keypoints2, RANSAC_ITER, RANSAC_THRESHOLD * threshold)

    num_inliers = np.sum(inliers).astype(int)
    # print(f"matches without sum: {len(inliers)}\n")
    # print(f"inliers with sum: {num_inliers}\n")
    # print(f"inliers: {inliers}\n")
    # Ensure that there are at least 4 inliers
    if num_inliers < 4:
        if DEBUG:
            print(f"\033[93mWarning: Not enough inliers. Number of inliers: {num_inliers}\033[m")
        return np.eye(3,3), inliers, keypoints1[inliers], keypoints2[inliers]

    # HOMOGRAPHY
    homography = findHomography(keypoints1[inliers], keypoints2[inliers])

    return homography, inliers, keypoints1[inliers], keypoints2[inliers]

def evaluateHomography(feat1: np.ndarray, feat2: np.ndarray, H: np.ndarray):
    # Warp the features of the second image
    warped = feat2
    for i, (x, y) in enumerate(feat2):
        aux = np.dot(H, np.array([x, y, 1]))
        warped[i] = (aux/aux[2])[:2]

    # Calculate sum of squared distances from each pair of keypoints
    dist = float(np.mean(np.sum((feat1 - warped)**2, axis=1)))

    return dist

def smart_homography(features):
    # Initialize H and fill diagonal with identity matrices
    H = [[np.empty((3, 3)) for i in range(len(features))] for j in range(len(features))]
    for i in range(len(H)):
        H[i][i] = np.eye(3)

    # Initialize progress bar

    # Versão Tomás
    matrix = np.zeros((len(features), len(features)))
    num_features = len(features)
    progress = np.arange(num_features)
    iter = 0
    
    while len(set(progress)) > 1: # Enquanto não se chegar ao fim de todas as linhas
        if not iter:
            print("Computing consecutive homographies...")
        elif iter == 1:
            print(f"Computing homographies with jumps...")
        bar = Progress(len(features)-1, "Progress:", True, True, False, True, True, 20)
        for i in range(num_features-1):
            for j in range(progress[i] + 1, num_features):
                
                best = progress[i]
                
                if matrix[best][j] == 0:
                    homo_aux, inliers, keypoints1, keypoints2 = findBestHomography(features[best], features[j])

                    if np.sum(inliers).astype(int) > 4:
                        error = evaluateHomography(keypoints1, keypoints2, homo_aux)
                        if (error > JUMP_THRESHOLD and j-i > 1):
                            progress[i] = j - 1
                            if DEBUG:
                                print(f"\nFailed in i = {i}-> b = {best}-> j = {j}")
                            break
                    else:
                        progress[i] = j - 1
                        if DEBUG:
                            print(f"\nFailed in i = {i}-> b = {best}-> j = {j}")
                        break

                    H[best][j] = homo_aux
                    matrix[best][j] = 1

                H[i][j] = np.dot(H[i][best], H[best][j])
                H[j][i] = np.linalg.inv(H[i][j])
                matrix[i][j] = 1
                matrix[j][i] = 1

                if j == num_features - 1:
                    progress[i] = j
            bar.update()
        
        iter += 1
        if iter == 100:
            print("Max iterations reached")
            break
    
    return H
    # Fim versão Tomás

def braindead_homography(features: np.ndarray):
    """
    Compute homographies between consecutive feature points.

    Parameters:
    -
    - features: An array of feature points, where each row represents
      a feature point and each column represents its coordinates and descriptors.

    Returns:
    -
    - H: A list of 2D arrays where each element represents a matrix H_ij
      representing the homography from feature point i to feature point j.
    """
    
    # Initialize H and fill diagonal with identity matrices
    H = [[np.empty((3, 3)) for i in range(len(features))] for j in range(len(features))]
    for i in range(len(H)):
        H[i][i] = np.eye(3)
    
    # Initialize progress bar
    bar = Progress(len(features)-1, "Computing homographies:", True, True, False, True, True, 20)
    
    # Compute homographies between consecutive feature points
    for i in range(len(features)-1):
        # Compute the upper triangular diagonal element
        h, inliers, keypoints1, keypoints2 = findBestHomography(features[i], features[i+1])
        H[i][i+1] = h/h[2,2]
        

        # Compute the lower triangular diagonal element
        h = np.linalg.inv(H[i][i+1])
        H[i+1][i] = h/h[2,2]
        
        for j in range(i-1, -1, -1):  # i=5 => j€[4,3,2,1,0]
            # Compute upper triangular elements above
            h = np.dot(H[i][i+1], H[j][i])
            H[j][i+1] = h/h[2,2]

            # Compute lower triangular elements to the left
            h = np.dot(H[i+1][i], H[i][j])
            H[i+1][j] = h/h[2,2]
        bar.update(i+1)
    return H

def output_map_H(features: np.ndarray, map_frame: int, map_H: np.ndarray) -> np.ndarray:
    """
    Compute and concatenate homographies from all frames to the map.

    Parameters:
    -
    - features: An array of feature points, where each row represents
      a feature point and each column represents its coordinates and descriptors.
    - map_frame: Index of the map frame for which homographies are computed.
    - map_H: Homography matrix representing the transformation from
      the map frame to the map.

    Returns:
    -
    - H: A 2D array where each column represents a flattened form of
      a homography matrix between the map frame and other feature points.
      The format of each homography is [map_frame, i, H_mi[0,0], H_mi[0,1], ..., H_mi[2,2]].
    """
    # Compute all homographies between feature points
    # all_H = braindead_homography(features)
    all_H = smart_homography(features)      # TODO
    
    # Concatenate homographies into a single array
    H = []
    for i in range(len(features)):
        if i == map_frame:
            homo = map_H.flatten()
        else:
            homo = np.dot(map_H, all_H[i][map_frame])
            homo = homo/homo[2,2]
        flat = np.hstack((np.array([0, i+1]), homo.flatten()))
        H.append(flat)

    # Convert the list of homographies into a 2D numpy array and transpose
    return np.array(H).T

def output_all_H(features: np.ndarray) -> np.ndarray:
    """
    Compute and concatenate all homographies between pairs of feature points.

    Parameters:
    -
    - features: An array of feature points, where each row represents
      a feature point and each column represents its coordinates and descriptors.

    Returns:
    -
    - H: A 2D array where each column represents a flattened form of
      a homography matrix between pairs of feature points. The format of each
      homography is [j, i, H_ji[0,0], H_ji[0,1], ..., H_ji[2,2]].
    """
    
    # Compute all homographies between feature points
    # all_H = braindead_homography(features)
    all_H = smart_homography(features)      # TODO

    # Concatenate homographies into a single array
    H = []
    for i in range(len(all_H)-1):
        for j in range(i+1, len(all_H)):
            # Create a flattened representation of the homography matrix
            flat = np.hstack((np.array([j+1, i+1]), all_H[j][i].flatten()))
            H.append(flat)
    
    # Convert the list of homographies into a 2D numpy array and transpose
    return np.array(H).T

def main():
    if len(sys.argv) != 2:
        raise TypeError("Usage: python compute_transform.py config_file.cfg")

    # Get the configuration file from the command-line argument
    cfg = Config(sys.argv[1])

    try:
        features = loadmat(cfg.keypoints_out)['features'][0, :]
    except FileNotFoundError:
        print(f"FileNotFoundError: File {cfg.keypoints_out} does not exist")
        exit(1)

    if cfg.transforms_params == 'all':
        H = output_all_H(features)

    elif cfg.transforms_params == 'map':
        m_i = 0
        map_frame = int(cfg.frame_number[m_i])
        map_H = findHomography(cfg.pts_in_frame[m_i], cfg.pts_in_map[m_i])
        map_H = map_H/map_H[2,2]
        H = output_map_H(features, map_frame, map_H)

    else:
        raise TypeError("Transforms type not recognized")

    savemat(cfg.transforms_out, {'H': H})

if __name__=='__main__':
    main()