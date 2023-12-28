import cv2
import pcl
import numpy as np
from pivlib.utils import Progress
from pivlib.config import Config
import sys

STEPSIZE = 1
FRAME_LIMIT = 3

def video2array(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return np.array(frames)

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_video.py config_file.cfg")
        sys.exit(1)

    config_data = Config(sys.argv[1])

    # Load videos to array
    videos = np.array([])
    for path in config_data.videos:
        vid = video2array(path)
        if vid is None:
            print("Error opening video file")
            exit()
        videos = np.append(videos, vid)

    # Check if videos are the same size
    if not np.all(videos.shape == videos[0].shape):
        print("Error: Videos are not the same size")
        exit()
    
    print(videos.shape)
    # Select video frames to process
    for i in range(videos.shape[0]):
        videos[i] = videos[i][:FRAME_LIMIT*STEPSIZE:STEPSIZE, :, :, :]
    print(videos.shape)
        
    num_frames = min(int(videos.shape[1] / STEPSIZE), FRAME_LIMIT)

    #Thigs to do
    #Calibration
    #Stereo
    #Pose estimation
    # USE cv2.goodFeaturesToTrack !!!!!!!!!


    

if __name__ == '__main__':
    main()
