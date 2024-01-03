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

def reverse_triangular_sum(target_sum):
    # Coefficients of the quadratic equation
    a = 1/2
    b = -1/2
    c = -target_sum

    # Calculate the roots of the quadratic equation
    roots = np.roots([a, b, c])

    # Select the real positive root
    real_roots = roots[np.isreal(roots)].real
    positive_root = real_roots[real_roots > 0]

    if len(positive_root) == 0:
        return None  # No real positive root

    return int(positive_root[0])

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

def showHomography(frame1, homography: np.ndarray):
    concatenated_image = np.zeros((frame1.shape[0], 2*frame1.shape[1], 3), dtype=np.uint8)
    
    # print(f"image2.shape[:2][::-1] = {image2.shape[:2][::-1]}")
    dst = cv2.warpPerspective(frame1, homography, frame1.shape[:2][::-1])
    # Draw the transformed image side by side with the first image

    # Copy the transformed image into the empty space
    concatenated_image[:frame1.shape[0], :frame1.shape[1]] = frame1
    concatenated_image[:dst.shape[0], dst.shape[1]:] = dst
    
    # Show the concatenated image with lines
    plt.title(f"Image warped with homography")
    plt.imshow(concatenated_image)
    plt.show()

def stitch(frame1, frame2, homography: np.ndarray):
    # Create an empty image to concatenate the two images side by side
    concatenated_image = np.zeros((max(frame1.shape[0], frame2.shape[0]), frame1.shape[1] + frame2.shape[1], 3), dtype=np.uint8)

    # Copy the images into the concatenated image
    concatenated_image[:frame1.shape[0], :frame1.shape[1]] = frame1
    concatenated_image[:frame2.shape[0], frame2.shape[1]:] = frame2

    # print(f"frame2.shape[:2][::-1] = {frame2.shape[:2][::-1]}")
    dst = cv2.warpPerspective(frame2, homography, frame2.shape[:2][::-1])

    img = addWeighted(frame1, 0.5, dst, 0.5)
    
    # Show the concatenated image with lines
    plt.imshow(img)
    plt.show()

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
    print_color("[any other key]\n", 'YELLOW')
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
    homo_size = reverse_triangular_sum(len(homographies))
    if homo_size is None:
        print("Homography with size 0")
        exit()
    
    # Initialize original Homography matrix
    H = [[np.empty((3, 3)) for i in range(homo_size)] for j in range(homo_size)]
    for i in range(len(H)):
        H[i][i] = np.eye(3)
    
    for homo in homographies:
        j, i = homo[:2]
        params = homo[2:].reshape(3,3)
        H[int(j)][int(i)] = params
        H[int(i)][int(j)] = np.linalg.inv(params)
    
    
    video_num = int(input("Choose video: "))
    while video_num < 0 or video_num > videos.shape[0]:
        print("Choose again")
        video_num = int(input("Choose video: "))

    vid = video_num - 1
    vid = videos[vid]

    while True:
        menu()
        action = str(input())
        #action = str(input("Choose what to do:\nshow frames [1]\nshow homography [2]\nstitch [3]\n> "))
        if action == "1":
            try:
                print("============================")
                frame1 = int(input("Enter frame1: "))
                frame2 = int(input("Enter frame2: "))
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
                frame1 = int(input("Enter frame1: "))
                frame2 = int(input("Enter frame2: "))
                print("============================")
                if frame1 >= homo_size or frame2 >= homo_size:
                    print("Frame number out of bounds")
                    continue
                showHomography(vid[frame2],H[frame2][frame1])
            except:
                continue
        elif action == "3":
            try:
                print("============================")
                frame1 = int(input("Enter frame1: "))
                frame2 = int(input("Enter frame2: "))
                print("============================")
                if frame1 >= homo_size or frame2 >= homo_size:
                    print("Frame number out of bounds")
                    continue
                stitch(vid[frame1],vid[frame2],H[frame2][frame1])
            except:
                continue
        elif action.isnumeric():
            continue
        else:
            break


if __name__ == "__main__":
    main()