import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_pts(img):
    # plot image and save coordinates clicked with the mouse
    # input: image
    # output: coordinates clicked
    
    pts = []
    # mouse callback function
    def get_coordinates(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            pts.append([x,y])
    
    # Create a black image, a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',get_coordinates) # type: ignore
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return pts

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

def main():
    # filename = "resources/viana.png"
    filename = "resources/Tesla/TeslaVC_carreiraVIDEOS/2023-07-23_11-36-50-front.mp4"
    
    video = video2array(filename, frame_limit=1, frame_step=15, scale=1/2)
    if video is None:
        print("Error opening video file")
        exit()
    
    frame = video[0]
    
    # frame = cv2.imread(filename)
    
    print(get_pts(frame))
    
    pass

if __name__ == "__main__":
    main()