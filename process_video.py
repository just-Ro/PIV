import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys
from pprint import pprint
from scipy.io import savemat, loadmat
from pivlib.utils import Progress
from pivlib.config import Config

STEPSIZE = 1
FRAME_LIMIT = 50
DOWNSCALE_FACTOR = 2

def feature_extraction(img):
    # Make a new image equal to the original image
    img_copy = img.copy()

    # Convert the image to grayscale
    gray_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY, img_copy)

    # Create a SIFT object
    sift = cv.SIFT.create()

    # detect keypoints and compute descriptors of the image
    keypoints, descriptors = sift.detectAndCompute(gray_img, None) # type: ignore
    
    # Draw keypoints on the image
    img_with_keypoints = cv.drawKeypoints(img_copy, keypoints, img_copy, flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)

    # Display the frame with keypoints
    cv.imshow('Frame with Keypoints', img_with_keypoints)
    cv.waitKey(int(STEPSIZE))  # Adjust the wait time to control the speed of the video
    
    #Display the image with keypoints
    #plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
    #plt.axis("off")
    #plt.show()

    keypoints_coord = []
    # store the keypoints coordinates
    for point in keypoints:
        keypoints_coord.append(point.pt)

    keypoints_coord = np.array(keypoints_coord)
    
    descriptors = np.array(descriptors)

    features = np.append(keypoints_coord, descriptors, axis=1)
    features = np.transpose(features)

    return features


def video2array(filepath):
    cap = cv.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return np.array(frames)


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_video.py config_file.cfg")
        sys.exit(1)

    config_data = Config(sys.argv[1])
    video_path = config_data.videos[0]

    video_array = video2array(video_path)
    if video_array is None:
        print("Error opening video file")
        exit()

    num_frames = min(int(video_array.shape[0] / STEPSIZE), FRAME_LIMIT)
    features = np.empty(num_frames, dtype='object')
    used_frames = np.empty(num_frames, dtype='object')

    bar = Progress(num_frames, "Frames analyzed:", True, True, False, True, True, 20)

    for i in range(num_frames):
        frame = video_array[i * STEPSIZE]
        frame = cv.resize(frame, (0, 0), fx=1 / DOWNSCALE_FACTOR, fy=1 / DOWNSCALE_FACTOR)
        used_frames[i] = frame
        features[i] = feature_extraction(frame)
        bar.update(i+1)

    cv.destroyAllWindows()

    savemat(config_data.keypoints_out, {'features': features})
    savemat("output/frames.mat", {'frames': used_frames})


if __name__=='__main__':
    main()