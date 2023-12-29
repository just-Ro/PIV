import cv2
import numpy as np
from pivlib.utils import Progress
from pivlib.config import Config
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import sys


STEPSIZE = 1
#Frame Limit = -1 means no limit
FRAME_LIMIT = 10
DOWNSCALE_FACTOR = 2
DISTANCE = 100

def video2array(filepath, frame_limit=-1, frame_step: int=1, scale: float=1):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"Error: Could not open video file {filepath}")
        return None

    frames = []
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check if the frame limit has been reached
        if frame_limit != -1 and len(frames) == frame_limit:
            break

        # Check if the current frame should be included
        if counter % frame_step == 0:
            if scale != 1:
                # Resize the frame
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            frames.append(frame_rgb)

        counter += 1
        
    cap.release()
    return np.array(frames)


def import_videos(filepaths, frame_limit=-1, frame_step: int=1, scale: float=1, verbose: bool=False):
    videos = []
    
    bar = Progress(len(filepaths), "Videos imported:", True, True)
    for i, path in enumerate(filepaths):
        vid = video2array(path, frame_limit, frame_step, scale)
        if vid is None:
            print("Error opening video file")
            exit()
        videos.append(vid)
        if verbose:
            bar.update(i+1)

    # Check that all videos have the same number of frames
    if len(set([vid.shape[0] for vid in videos])) != 1:
        print("Error: Videos have different number of frames")
        exit()
    
    return np.concatenate([arr[np.newaxis, :] for arr in videos], axis=0)


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


def features_from_videos(videos: np.ndarray) -> np.ndarray:

    features = np.empty((videos.shape[0], videos.shape[1]), dtype='object')
    bar = Progress(videos.shape[0]*videos.shape[1], "Frames analyzed:", True, True, False, True, True, 20)

    for v, video in enumerate(videos):
        for i, frame in enumerate(video):
            features[v,i] = feature_extraction(frame)
            bar.update(v*videos.shape[1]+i+1)

    return features


def feature_extraction(frame: np.ndarray):
    # Use SIFT to detect keypoints and descriptors
    sift = cv2.SIFT_create() # type: ignore
    keypoints, descriptors = sift.detectAndCompute(frame, None)

    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(frame, keypoints, frame, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # Display the frame with keypoints
    cv2.imshow('Frame with Keypoints', img_with_keypoints)
    cv2.waitKey(int(STEPSIZE))  # Adjust the wait time to control the speed of the video

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


def findHomography(keypoints1: np.ndarray, keypoints2: np.ndarray):
    """
    Find the homography between two sets of keypoints.
    Both sets of keypoints must have the same input shape \
        (num_features, num_descriptors) and the same number of descriptors.
    
    Parameters:
    -
    - keypoints1: Keypoints from the source image
    - keypoints2: Keypoints from the destination image
    
    Returns:
    -
    - H: The homography matrix between the two sets of keypoints
    """
    
    # Find the homography matrix
    H, _ = cv2.findHomography(keypoints1, keypoints2, cv2.RANSAC, 5.0) # type: ignore
    
    return H


def findallhomographies(features: np.ndarray):
    pass


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_video.py config_file.cfg")
        sys.exit(1)

    config_data = Config(sys.argv[1])

    #Thigs to do
    #Calibration
    #Stereo
    #Pose estimation
    # USE cv2.goodFeaturesToTrack !!!!!!!!!

    # Load videos to array
    videos = import_videos(config_data.videos, FRAME_LIMIT, STEPSIZE, 1/DOWNSCALE_FACTOR, True)

    # Extract features from all the frames in all the videos
    features = features_from_videos(videos)

    if config_data.transforms_type == "homography":
        if config_data.transforms_params == "all":
            pass
        elif config_data.transforms_params == "map":
            pass
        else:
            print("Error: Invalid transform parameter")
            exit()
    elif config_data.transforms_type == "rigid":
        pass
    elif config_data.transforms_type == "calibration":
        pass
    else:
        print("Error: Invalid transform type")
        exit()
    
    
    for frame in range(FRAME_LIMIT):
        # Find the nearest neighbors between two sets of keypoints
        keypoints1, keypoints2, std_dev = featureMatching(features[0, frame], features[1, frame], DISTANCE)

        # Find the homography matrix
        H = findHomography(keypoints1, keypoints2)



if __name__ == '__main__':
    main()
