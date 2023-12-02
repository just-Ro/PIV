import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys
from pprint import pprint
from scipy.io import savemat, loadmat
from pivlib.utils import Progress

stepsize = 10

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

def feature_extraction(img):
    # Convert the image to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create a SIFT object
    sift = cv.SIFT.create()

    # detect keypoints and compute descriptors of the image
    keypoints, descriptors = sift.detectAndCompute(gray_img, None) # type: ignore

    # Draw keypoints on the image
    img_with_keypoints = cv.drawKeypoints(img, keypoints, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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
    #print(features.shape)

    return features


def main():
    if len(sys.argv) != 2:
        print("Usage: python myprogram.py config_file.cfg")
        sys.exit(1)

    # Get the configuration file path from the command-line argument
    config_data = Config(parse_config_file(sys.argv[1]))

    # video = cv.VideoCapture(config_data['videos'])
    video = cv.VideoCapture(config_data.videos)
    if not video.isOpened():
        print("Error opening video file")
        exit()

    counter = 0
    features = np.empty(int(video.get(cv.CAP_PROP_FRAME_COUNT)/stepsize), dtype='object')
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    
    bar = Progress(total_frames, "Frames analyzed:", True, False, True, True, 50)
    
    while True:
        ret, frame = video.read()
        
        # Break the loop if we have reached the end of the video
        if not ret:
            break

        counter += 1
        if (counter % stepsize) == 0:
            features[int(counter/stepsize)-1] = feature_extraction(frame)

        bar.update(counter)

    video.release()
    cv.destroyAllWindows()

    print(len(features))

    savemat(config_data.keypoints_out, {'features': features})

if __name__=='__main__':
    main()
    