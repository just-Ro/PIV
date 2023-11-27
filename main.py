import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys
from pprint import pprint
from scipy.io import savemat

class Config():
    def __init__(self, config_dict: dict):
        self.videos: str = config_dict['videos'][0][0]
        self.keypoints_out: str = config_dict['keypoints_out'][0][0]
        self.transforms_out: str = config_dict['transforms_out'][0][0]
        self.transforms_type: str = config_dict['transforms'][0][0]
        self.transforms_params: str = config_dict['transforms'][0][1]
        self.pts_in_frame: np.array = np.array(config_dict['pts_in_frame'])[:, 1:].astype(int).reshape(2,-1,2)
        self.pts_in_map: np.array = np.array(config_dict['pts_in_map'])[:, 1:].astype(int).reshape(2,-1,2)

    def show(self):
        print("videos:              ",self.videos)
        print("points in frame:     ",self.pts_in_frame)
        print("points in map:       ",self.pts_in_map)
        print("transforms type:     ",self.transforms_type)
        print("transforms params:   ",self.transforms_params)
        print("transforms out:      ",self.transforms_out)
        print("keypoints out:       ",self.keypoints_out)


def feature_extraction(img):
    # Convert the image to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #SIFT
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)

    # Draw keypoints on the image
    img_with_keypoints = cv.drawKeypoints(gray_img, keypoints, None)

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
    print(features.shape)


    return features



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


def main():
    if len(sys.argv) != 2:
        print("Usage: python myprogram.py config_file.cfg")
        sys.exit(1)

    # Get the configuration file path from the command-line argument
    config_file_path = sys.argv[1]
    
    config_data = Config(parse_config_file(config_file_path))

    config_data.show()


    
    # video = cv.VideoCapture(config_data['videos'])
    video = cv.VideoCapture('Ihitaclip.mp4')
    if not video.isOpened():
        print("Error opening video file")
        exit()

    counter = 0

    # features = np.array([], dtype='numpy.ndarray')
    features = []
    while True:
        # Read the frame
        ret, frame = video.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        counter += 1
        if (counter % 30) == 0:
            # features = np.append(features, feature_extraction(frame))
            features.append(feature_extraction(frame))
            print(counter)

    # np.append(features, axis=0)

    print(len(features))
    print(features[0].shape)
    print(features[-1].shape)

    features = np.array(features, dtype='object')
    print(features.shape)

    # features = [arr.tolist() for arr in features]
    savemat('output.mat', {'features': features})

    # problema -> passar as coisas para cellarray


if __name__=='__main__':
    main()
    