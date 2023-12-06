import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys
from pprint import pprint
from scipy.io import savemat, loadmat
from pivlib.utils import Progress
from pivlib.config import Config

stepsize = 50

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
        print("Usage: python process_video.py config_file.cfg")
        sys.exit(1)

    # Get the configuration file path from the command-line argument
    config_data = Config(sys.argv[1])

    # video = cv.VideoCapture(config_data['videos'])
    video = cv.VideoCapture(config_data.videos)
    if not video.isOpened():
        print("Error opening video file")
        exit()

    counter = 0
    features = np.empty(int(video.get(cv.CAP_PROP_FRAME_COUNT)/stepsize), dtype='object')
    used_frames = np.empty(int(video.get(cv.CAP_PROP_FRAME_COUNT)/stepsize), dtype='object')
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    
    bar = Progress(total_frames, "Frames analyzed:", True, True, False, True, True, 20)
    
    while True:
        ret, frame = video.read()
        
        # Break the loop if we have reached the end of the video
        if not ret:
            break
        
        # Get the height and width of the image
        height, _= frame.shape[:2]
        # Define the region of interest (ROI) for the top half of the image
        roi = frame[:height//2, :]
        # Apply Gaussian blur to the ROI
        blurred_roi = cv.GaussianBlur(roi, (15, 15), 0)
        # Replace the top half of the original image with the blurred ROI
        frame[:height//2, :] = blurred_roi

        counter += 1
        if (counter % stepsize) == 0:
            features[int(counter/stepsize)-1] = feature_extraction(frame)
            used_frames[int(counter/stepsize)-1] = frame
        bar.update(counter)

    video.release()
    cv.destroyAllWindows()

    savemat(config_data.keypoints_out, {'features': features})
    savemat("frames.mat", {'frames': used_frames})

if __name__=='__main__':
    main()
    