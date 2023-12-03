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
    keypoints1 = frame[largest][:, :2][indices] if largest == 0 else frame[~largest][:, :2]
    keypoints2 = frame[largest][:, :2][indices] if largest == 1 else frame[~largest][:, :2]

    # Falta meter as imagens lado a lado para ver melhor(está a funcionar bem) 
    """ # Display the pairs
    print('Keypoint Pairs:')
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

def mapHomographies(features):

    H_prev =np.eye(3)

    # feature[0,0]
    # feature[0,1]
    
    # for img in features:

def allHomographies(features):
    pass

def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_transform.py config_file.cfg")
        sys.exit(1)

    # Get the configuration file path from the command-line argument
    config_data = Config(sys.argv[1])

    features = loadmat(config_data.keypoints_out)['features']
    features = features[0, :]

    if config_data.transforms_type == 'all':
        allHomographies(features)
    elif config_data.transforms_type == 'map':
        mapHomographies(features)
    else:
        print("Transforms type not recognized")
        sys.exit(1)

if __name__=='__main__':
    main()