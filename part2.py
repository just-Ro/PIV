import cv2
import pcl
import numpy as np
from pivlib.utils import Progress
from pivlib.config import Config
import sys

STEPSIZE = 1
FRAME_LIMIT = 3

def video2array(filepath, frame_limit=-1, frame_step: int=1):
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
        
        # Check if the current frame should be included
        if counter % frame_step == 0 and (len(frames) < frame_limit or frame_limit == -1):
            frames.append(frame_rgb)

        counter += 1

    cap.release()
    return np.array(frames)

def import_videos(filepaths, frame_limit=-1, frame_step: int=1):
    videos = []
    for path in filepaths:
        vid = video2array(path, frame_limit, frame_step)
        if vid is None:
            print("Error opening video file")
            exit()
        videos.append(vid)

    # Check that all videos have the same number of frames
    if len(set([vid.shape[0] for vid in videos])) != 1:
        print("Error: Videos have different number of frames")
        exit()
    
    return np.concatenate([arr[np.newaxis, :] for arr in videos], axis=0)


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_video.py config_file.cfg")
        sys.exit(1)

    config_data = Config(sys.argv[1])

    # Load videos to array
    videos = import_videos(config_data.videos, FRAME_LIMIT, STEPSIZE)
    
    

    #Thigs to do
    #Calibration
    #Stereo
    #Pose estimation
    # USE cv2.goodFeaturesToTrack !!!!!!!!!


    

if __name__ == '__main__':
    main()
