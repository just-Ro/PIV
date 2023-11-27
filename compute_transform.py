import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2 as cv
import sys
from pprint import pprint

class Config():
    def __init__(self, config_dict: dict):
        self.videos: str = config_dict['videos'][0]
        self.keypoints_out: str = config_dict['keypoints_out'][0]
        #self.
        pass


def feature_extraction() -> np.ndarray:
    """Extract features from the images"""
    # Read an image from file
    image_path = "image.jpg"
    img = cv.imread(image_path)

    # Convert the image to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)

    # Draw keypoints on the image
    img_with_keypoints = cv.drawKeypoints(gray_img, keypoints, None)

    """ # Display the image with keypoints
    plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show() """


    keypoints_coord = []
    # store the keypoints coordinates
    for point in keypoints:
        keypoints_coord.append(point.pt)
    keypoints_coord = np.array(keypoints_coord)

    descriptors = np.array(descriptors)

    features = np.append(keypoints_coord, descriptors, axis=1)
    features = np.transpose(features)

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
    
    config_data = parse_config_file(config_file_path)

        print(pprint(config_data))
        feature_extraction()


if __name__=='__main__':
    main()