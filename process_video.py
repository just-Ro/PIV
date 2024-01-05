import numpy as np
import cv2 as cv
import sys
from scipy.io import savemat
from pivlib.utils import Progress
from pivlib.config import Config
from constants import *

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
    if DEBUG:
        cv.imshow('Frame with Keypoints', img_with_keypoints)
        cv.waitKey(int(STEPSIZE))  # Adjust the wait time to control the speed of the video

    keypoints_coord = []
    # store the keypoints coordinates
    for point in keypoints:
        keypoints_coord.append(point.pt)

    keypoints_coord = np.array(keypoints_coord)
    
    descriptors = np.array(descriptors)

    features = np.append(keypoints_coord, descriptors, axis=1)
    features = np.transpose(features)

    return features

def video2array(filepath, frame_limit=-1, frame_step: int=1, scale: float=1):
    cap = cv.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"Error: Could not open video file {filepath}")
        return None

    frames = []
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Check if the frame limit has been reached
        if frame_limit != -1 and len(frames) == frame_limit:
            break

        # Check if the current frame should be included
        if counter % frame_step == 0:
            if scale != 1:
                # Resize the frame
                frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)
            frames.append(frame_rgb)

        counter += 1
        
    cap.release()
    return np.array(frames)


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_video.py config_file.cfg")
        sys.exit(1)

    config_data = Config(sys.argv[1])

    
    # Load the video
    video_array = video2array(config_data.videos[0], FRAME_LIMIT, STEPSIZE, 1/DOWNSCALE_FACTOR)
    if video_array is None:
        print("Error opening video file")
        exit()

    num_frames = video_array.shape[0]
    features = np.empty(num_frames, dtype='object')
    used_frames = np.empty(num_frames, dtype='object')

    print(f"Downscaling the video by a factor of {DOWNSCALE_FACTOR}")
    print(f"Processing {num_frames} frames of the video, {STEPSIZE} frames apart")
    
    bar = Progress(num_frames, "Extracting features:", True, True, False, True, True, 20)

    # Extract features from each frame
    for i,frame in enumerate(video_array):
        if DEBUG:
            used_frames[i] = frame
        features[i] = feature_extraction(frame)
        bar.update(i+1)

    cv.destroyAllWindows()

    # Save the features to a .mat file
    savemat(config_data.keypoints_out, {'features': features})
    if DEBUG:
        savemat("output/frames.mat", {'frames': used_frames})


if __name__=='__main__':
    main()