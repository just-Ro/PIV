import cv2
import numpy as np
from pivlib.utils import Progress, addWeighted
from pivlib.config import Config
from matplotlib import pyplot as plt
import sys
from scipy.io import loadmat
from constants import *


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

def showFrames(frame1, frame2):
    # Create an empty image to concatenate the two images side by side
    concatenated_image = np.zeros((max(frame1.shape[0], frame2.shape[0]), frame1.shape[1] + frame2.shape[1], 3), dtype=np.uint8)

    # Copy the images into the concatenated image
    concatenated_image[:frame1.shape[0], :frame1.shape[1]] = frame1
    concatenated_image[:frame2.shape[0], frame2.shape[1]:] = frame2

    # Show the concatenated image with lines
    #plt.title(f"Image {frame_number1} and Image {frame_number2}")
    plt.imshow(concatenated_image)
    plt.show()
    # plt.savefig('frames.png')

def stitch(frame1, frame2):
    img = addWeighted(frame1, 0.5, frame2, 0.5)
    
    # Show the concatenated image with lines
    plt.imshow(img)
    plt.show()
    # plt.savefig('stitched.png')

def print_color(text, color):
    colors = {
        'RESET': '\033[0m',
        'BRIGHT': '\033[1m',
        'DIM': '\033[2m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'CYAN': '\033[96m',
    }
    print(f"{colors[color]}{text}{colors['RESET']}", end="")
    return

def menu():
    print("============================")
    print("\033[1mChoose what to do: \033[0m")
    print("show frames ", end="")
    print_color("[1]\n", 'RED')
    print("show homography ", end="")
    print_color("[2]\n", 'GREEN')
    print("stitch ", end="")
    print_color("[3]\n", 'CYAN')
    print("exit ", end="")
    print_color("[Enter]\n", 'YELLOW')
    print("============================")
    print("Input: ", end="")


def main():
    if len(sys.argv) != 2:
        print("Usage: python homography_probe.py config_file.cfg")
        sys.exit(1)

    cfg = Config(sys.argv[1])

    print("============================")
    print("\033[1mWELCOME! \033[0m")

    # Load videos to array
    videos = import_videos(cfg.videos, FRAME_LIMIT, STEPSIZE, 1/DOWNSCALE_FACTOR, True)

    try:
        homographies = loadmat(cfg.transforms_out)['homographies'].T
    except FileNotFoundError:
        print(f"FileNotFoundError: File {cfg.transforms_out} does not exist")
        exit(1)

    # Find original Homography matrix size
    homo_size = videos[0].shape[0]
    
    H = []
    if cfg.transforms_params == 'all':
        # Initialize original Homography matrix
        H = [[np.empty((3, 3)) for i in range(homo_size)] for j in range(homo_size)]
        for i in range(len(H)):
            H[i][i] = np.eye(3)

        for homo in homographies:
            j, i = int(homo[0]-1), int(homo[1]-1)
            params = homo[2:].reshape(3,3)
            H[j][i] = params
            H[i][j] = np.linalg.inv(params)
    
    elif cfg.transforms_params == 'map':
        H = [np.empty((3, 3)) for i in range(homo_size)]

        for homo in homographies:
            i = int(homo[1]-1)
            H[i] = homo[2:].reshape(3,3)
    
    
    video_num = int(input("Choose video: "))
    while video_num < 0 or video_num > videos.shape[0]:
        print("Choose again")
        video_num = int(input("Choose video: "))

    vid = video_num - 1
    vid = videos[vid]
    if cfg.transforms_params == 'all':
        while True:
            menu()
            action = str(input())
            if action == "1":
                try:
                    print("============================")
                    frame1 = int(input("Enter frame1: "))-1
                    frame2 = int(input("Enter frame2: "))-1
                    print("============================")
                    if frame1 >= homo_size or frame2 >= homo_size:
                        print("Frame number out of bounds")
                        continue
                    showFrames(vid[frame1],vid[frame2])
                except:
                    continue
            elif action == "2":
                try:
                    print("============================")
                    frame1 = int(input("Enter frame1: "))-1
                    frame2 = int(input("Enter frame2: "))-1
                    print("============================")
                    if frame1 >= homo_size or frame2 >= homo_size:
                        print("Frame number out of bounds")
                        continue
                    shape = vid[frame2].shape[:2][::-1]
                    homo = cv2.warpPerspective(vid[frame2],H[frame2][frame1],shape)
                    showFrames(vid[frame2],homo)
                except:
                    continue
            elif action == "3":
                try:
                    print("============================")
                    frame1 = int(input("Enter frame1: "))-1
                    frame2 = int(input("Enter frame2: "))-1
                    print("============================")
                    if frame1 >= homo_size or frame2 >= homo_size:
                        print("Frame number out of bounds")
                        continue
                    shape = vid[frame2].shape[:2][::-1]
                    frame2_to_1 = cv2.warpPerspective(vid[frame2], H[frame2][frame1], shape)
                    stitch(vid[frame1],frame2_to_1)
                except:
                    continue
            elif action.isnumeric():
                continue
            else:
                break
            
    elif cfg.transforms_params == 'map':
        while True:
            menu()
            action = str(input())
            mapframe = cfg.frame_number[0]
            if action == "1":
                try:
                    print("============================")
                    frame1 = int(input("Choose frame: "))-1
                    print("============================")
                    if frame1 >= homo_size:
                        print("Frame number out of bounds")
                        continue
                    showFrames(vid[frame1],vid[mapframe])
                except:
                    continue
            elif action == "2":
                try:
                    print("============================")
                    frame1 = int(input("Choose frame: "))-1
                    print("============================")
                    if frame1 >= homo_size:
                        print("Frame number out of bounds")
                        continue
                    shape = vid[frame1].shape[:2][::-1]
                    homo = cv2.warpPerspective(vid[frame1],H[frame1],shape) # type: ignore
                    showFrames(vid[frame1],homo)
                except:
                    continue
            elif action == "3":
                try:
                    print("============================")
                    frame1 = int(input("Choose frame: "))-1
                    print("============================")
                    if frame1 >= homo_size:
                        print("Frame number out of bounds")
                        continue
                    shape = vid[frame1].shape[:2][::-1]
                    frame1_to_map = cv2.warpPerspective(vid[frame1], H[frame1], shape) # type: ignore
                    #stitch(vid[mapframe],frame1_to_map)
                    shape = vid[mapframe].shape[:2][::-1]
                    #warp mapframe to map
                    mapframe_to_map = cv2.warpPerspective(vid[mapframe], H[mapframe], shape)
                    stitch(mapframe_to_map ,frame1_to_map)

                except:
                    continue
            elif action.isnumeric():
                continue
            else:
                break
    else:
        raise TypeError("Transforms type not recognized")


if __name__ == "__main__":
    main()