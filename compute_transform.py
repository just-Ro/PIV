import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors                                                       #the goattt
from pivlib.cv import ransac, findHomography
from pivlib.config import Config


def feature_matching(frame1: np.ndarray, frame2: np.ndarray):
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

def findBestHomography(features1: np.ndarray, features2: np.ndarray):
    # MATCHING
    keypoints1, keypoints2 = feature_matching(features1.T, features2.T)
    # RANSAC
    _, inliers = ransac(keypoints1, keypoints2, 100, 0.1)   # 100 iterations, 0.1 threshold --- Sujeito a alteracoes
    # HOMOGRAPHY
    homography = findHomography(keypoints1[inliers], keypoints2[inliers])

    return homography

def mapHomographies(mapframe: int, seqHomographies: np.ndarray, allHomographies: np.ndarray) -> np.ndarray:
    """Calculate the homography between each frame and the map"""

    homographies = []

    for i in range(len(seqHomographies)):
        M = (i+1)*(i+2)/2
        #ESTA MAL, NAO ESTAMOS A TIRAR MATRIZ DA HOOMOGRAPHY
        if i > mapframe:
            hom = np.dot(mapHomography, np.linalg.inv(allHomographies[int(i*len(seqHomographies)+mapframe-M)]))
        else:
            hom = np.dot(mapHomography, allHomographies[int(i*len(seqHomographies)+mapframe-M)])

        homography = np.hstack((np.array([mapframe,i]),hom.flatten()))

        homographies.append(homography)

    return np.array(homographies).T

def allHomographies(seqHomographies: np.ndarray) -> np.ndarray:
    """Calculate the homography between each pair of frames"""

    homographies = []
    prev = np.identity(3)
    
    """
    [- a b c]
    [- - d e]
    [- - - f]
    [- - - -]
    
    
          map
           |
    [0 1 2 3 4 5 6]
    
    [- 0 0 1 0 0 0]
    [- - 0 1 0 0 0]
    [- - - 1 0 0 0]
    [- - - - 1 1 1]
    [- - - - - 0 0]
    [- - - - - - 0]
    [- - - - - - -]

    [01 02 03 04 05 06]
    [12 13 14 15 16] 
    [23 24 25 26] 
    [34 35 36]
    [45 46] 
    [56]

    [01 02 03 04 05 06 12 13 14 15 16 23 24 25 26 34 35 36 45 46 56]

    [03 13 23 I 34 35 36]
    [2 7 11 I 15 16 17]
    
    """

    for i in range(len(seqHomographies)-1):
        prev = seqHomographies[i]
        for j in range(i+1, len(seqHomographies)):
            homography = np.hstack((np.array([j,i]),prev.flatten()))

            homographies.append(homography)

            prev = np.dot(seqHomographies[j], prev)
    
    return np.array(homographies).T

# def combinations(features: np.ndarray) -> np.ndarray:
    
    
#     pass

def everyHomography(features: np.ndarray) :
        

    re


def sequentialHomographies(features: np.ndarray) -> np.ndarray:
    """Calculate the homography between each pair of consecutive frames"""

    homographies = []

    for i in range(len(features)-1):
        homography = findBestHomography(features[i], features[i+1])

        homographies.append(homography)
            
    return np.array(homographies)

def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_transform.py config_file.cfg")
        sys.exit(1)

    # Get the configuration file path from the command-line argument
    config_data = Config(sys.argv[1])
    config_data.show()

    features = loadmat(config_data.keypoints_out)['features']
    features = features[0, :]

    seqHomographies = sequentialHomographies(features)

    if config_data.transforms_params == 'all':
        allHomographies(seqHomographies)
    elif config_data.transforms_params == 'map':
        mapHomographies(seqHomographies)
    else:
        print("Transforms type not recognized")
        sys.exit(1)

if __name__=='__main__':
    main()