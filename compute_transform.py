import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat, savemat
from sklearn.neighbors import NearestNeighbors                                                       #the goattt
from pivlib.cv import ransac, findHomography
from pivlib.config import Config
from pivlib.utils import Progress


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
    """ Find the best homography between two sets of keypoints """
    
    # MATCHING
    keypoints1, keypoints2 = feature_matching(features1.T, features2.T)
    # RANSAC
    _, inliers = ransac(keypoints1, keypoints2, 100, 10)
    # HOMOGRAPHY
    homography = findHomography(keypoints1[inliers], keypoints2[inliers])

    return homography

def mapHomographies(mapHomography: np.ndarray, mapframe: int, everyHomographies) -> np.ndarray:
    """Calculate the homography between each frame and the map"""

    homographies = []
    for i in range(len(everyHomographies)-1):
 
        if i == mapframe:
            hom = mapHomography
        elif i > mapframe:
            hom = np.dot(mapHomography, np.linalg.inv(everyHomographies[mapframe][i]))
        else:
            hom = np.dot(mapHomography, everyHomographies[i][mapframe])

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
    """

    for i in range(len(seqHomographies)-1):
        prev = seqHomographies[i]
        for j in range(i+1, len(seqHomographies)):
            homography = np.hstack((np.array([j,i]),prev.flatten()))

            homographies.append(homography)

            prev = np.dot(seqHomographies[j], prev)
    
    return np.array(homographies).T

def everyHomography(seqHomographies: np.ndarray):
    """Calculate the homography between each pair of frames"""

    prev = np.identity(3)
    homographies = [[None] * (len(seqHomographies)+1)] *  (len(seqHomographies)+1)
    
    for i in range(len(seqHomographies)-1):
        prev = seqHomographies[i]
        for j in range(i+1, len(seqHomographies)):
            homographies[i][j] = prev
            prev = np.dot(seqHomographies[j], prev)

    return homographies

def sequentialHomographies(features: np.ndarray) -> np.ndarray:
    """Calculate the homography between each pair of consecutive frames"""

    homographies = []

    bar = Progress(len(features)-2, "Sequential:", display_title=True, display_fraction=True)
    
    for i in range(len(features)-1):
        homography = findBestHomography(features[i], features[i+1])

        homographies.append(homography)
        bar.update(i)
            
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

    print(f"Features shape: {features.shape}")

    seqHomographies = sequentialHomographies(features)

    if config_data.transforms_params == 'all':
        print("all")
        homographies = allHomographies(seqHomographies)
        print(homographies.shape)
        print(homographies)

    elif config_data.transforms_params == 'map':
        #mapframe = config_data.frame_number[0]
        #mapHomography = findHomography(config_data.pts_in_frame[:,0], config_data.pts_in_map[:,0])
        #test with map being the first frame
        mapframe = 0
        mapHomography = seqHomographies[0]
        everyhomography = everyHomography(seqHomographies)
        homographies = mapHomographies(mapHomography, mapframe, everyhomography)
    
    else:
        print("Transforms type not recognized")
        sys.exit(1)

    savemat(config_data.transforms_out, {'homographies': homographies})
    #DONEEEE agora é so testar e encontrar 500 erros ARDEU ARDEU ARDEU
    #POO TYPE BEAT
    #o nosso downfall
    #please write code to save us from this hell

if __name__=='__main__':
    main()