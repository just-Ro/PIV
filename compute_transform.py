import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors                                                       #the goattt


class Config():
    def __init__(self, config_dict: dict):
        self.videos: str = config_dict['videos'][0][0]
        self.keypoints_out: str = config_dict['keypoints_out'][0][0]
        self.transforms_out: str = config_dict['transforms_out'][0][0]
        self.transforms_type: str = config_dict['transforms'][0][0]
        self.transforms_params: str = config_dict['transforms'][0][1]
        self.pts_in_frame: np.ndarray = np.array(config_dict['pts_in_frame'])[:, 1:].astype(int).reshape(2,-1,2)
        self.pts_in_map: np.ndarray = np.array(config_dict['pts_in_map'])[:, 1:].astype(int).reshape(2,-1,2)

    def show(self):
        print("videos:              ",self.videos)
        print("points in frame:     ",self.pts_in_frame)
        print("points in map:       ",self.pts_in_map)
        print("transforms type:     ",self.transforms_type)
        print("transforms params:   ",self.transforms_params)
        print("transforms out:      ",self.transforms_out)
        print("keypoints out:       ",self.keypoints_out)

def parse_config_file(file_path: str) -> dict:
    """Parse config file on file_path"""
    
    config_dict: dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Ignore comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Split the line into tokens
            tokens = line.split()
            
            # Extract parameter names and values
            param_name = tokens[0]
            param_values = [tokens[1:]]

            # Check if the token already exists in the dictionary
            if param_name in config_dict:
                # Add new values to the existing token
                config_dict[param_name].extend(param_values)
            else:
                # Create a new entry in the dictionary
                config_dict[param_name] = param_values

    return config_dict

def feat(frame1, frame2):
    
    
    
    pass

def feature_matching(frame1, frame2):
    """
    Find the nearest neighbors between two sets of keypoints
    - param src: Keypoints from the source image
    - param dst: Keypoints from the destination image
    - return: A list of corresponding keypoint pairs
    """
    frame1 = frame1.T
    frame2 = frame2.T
    flagi = False

    if frame1.shape[0] < frame2.shape[0]:
        temp = frame1
        frame1 = frame2
        frame2 = temp
        flagi = True

    features1 = frame1[:,2:]
    features2 = frame2[:,2:]

    # Create a NearestNeighbors model
    knn_model = NearestNeighbors(n_neighbors=1)
    knn_model.fit(features1)

    # Use kneighbors to find the nearest neighbors
    _, indices = knn_model.kneighbors(features2, n_neighbors=1)

    keypoints2 = frame2[:, :2][indices.flatten()]
    keypoints1 = frame1[:, :2]

    return keypoints1, keypoints2 if flagi else keypoints2, keypoints1

def main():
    if len(sys.argv) != 2:
        print("Usage: python myprogram.py config_file.cfg")
        sys.exit(1)

    # Get the configuration file path from the command-line argument
    config_data = Config(parse_config_file(sys.argv[1]))

        
    # Feature matching
    features = loadmat(config_data.keypoints_out)['features']
    print(features.shape())



    

if __name__=='__main__':
    main()