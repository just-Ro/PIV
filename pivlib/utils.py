import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

class Progress():
    """
    A simple class for displaying progress during iterations.

    Parameters:
    -
    - goal (int): The total number of iterations (default is 100).
    - title (str): Title to display before the progress information (default is "Progress:").
    - display_title (bool): Whether to display the title (default is False).
    - display_fraction (bool): Whether to display the iteration fraction (e.g., "3/100") (default is False).
    - display_percent (bool): Whether to display the progress percentage (default is False).
    - display_eta (bool): Wether to display the remaining time (default is False).
    - display_bar (bool): Whether to display a progress bar (default is False).
    - bar_length (int): Length of the progress bar (default is 50).

    Methods:
    -
    - getstr(iteration: int) -> str:
        Returns a formatted string representing the progress information at a given iteration.

    - update(iteration: int):
        Prints the formatted progress string for a given iteration on the same line.
        Clears the line and prints a newline when the goal is reached.


    """
    
    def __init__(self, 
                 goal = 100, 
                 title = "Progress:", 
                 display_title = False, 
                 display_fraction = False, 
                 display_percent = False, 
                 display_eta = False, 
                 display_bar = False, 
                 bar_length = 50):
        """
        Initializes the Progress object with specified parameters.

        
        Parameters:
        -
        - goal: The total number of iterations (default is 100).
        - title: Title to display before the progress information (default is "Progress:").
        - display_title: Whether to display the title (default is False).
        - display_fraction: Whether to display the iteration fraction (e.g., "3/100") (default is False).
        - display_percent: Whether to display the progress percentage (default is False).
        - display_eta: Wether to display the remaining time (default is False).
        - display_bar: Whether to display a progress bar (default is False).
        - bar_length: Length of the progress bar (default is 50).
        """
        
        self.string = title
        self.goal=goal
        self.title = display_title
        self.frac=display_fraction
        self.per=display_percent
        self.bar=display_bar
        self.size=bar_length
        self.eta=display_eta
        
        self.__pre_calc__ = 1/goal
        self.__start_time__ = time.time()
        self.__prev_time__ = self.__start_time__

    def getstr(self, iteration: int) -> str:
        """
        Returns a formatted string representing the progress information at a given iteration.

        Parameters:
        -
        - iteration: The current iteration number.
        
        Returns:
        -
        - string: A formatted string representing the progress information.
        """
        
        string = ""
        if self.title:
            string += self.string.strip() + " "
        if self.frac:
            string += f"{iteration}/{self.goal} "
        if self.per:
            string += f"{100*iteration*self.__pre_calc__:.2f}% "
        if self.bar:
            full_bar = int(self.size*iteration*self.__pre_calc__)
            string += f"[{'■' * full_bar}{'□' * (self.size - full_bar)}] "
        if self.eta:

            self.__prev_time__ = time.time()
            elapsed_time = self.__prev_time__ - self.__start_time__

            remaining_time = elapsed_time / ((iteration if iteration>0 else self.goal)*self.__pre_calc__) - elapsed_time
            string += f"{int(remaining_time)}s  "
        
        return string
    
    def update(self, iteration: int):
        """
        Prints the formatted progress string for a given iteration on the same line.
        Clears the line and prints a newline when the goal is reached.

        Parameters:
        -
        - param iteration: The current iteration number.
        """
        
        print(f"\r{self.getstr(iteration)}", end='', flush=True)
        if iteration == self.goal:
            print()

def warpPerspective(src: np.ndarray, H: np.ndarray, dst_size) -> np.ndarray:
    """
    Apply a perspective transformation (warp) to an image using a homography matrix.

    Parameters:
    - src: Source image, a 2D or 3D NumPy array.
    - H: Homography matrix, a 3x3 matrix representing the perspective transformation.
    - dst_size: Size of the destination image, specified as (width, height).

    Returns:
    - dst: Warped image, a 2D or 3D NumPy array.
    """
    # Generate an empty image
    width, height = dst_size
    channel = src.shape[2] if src.ndim > 2 else 1
    dst = np.zeros((height, width, channel), dtype=src.dtype)
    
    # Iterate over the destination image
    for qy in range(height):
        for qx in range(width):
            # Calculate the inverse mapping using the homography matrix H
            p = np.dot(np.linalg.inv(H), [qx, qy, 1])
            
            px, py = int(p[0]/p[-1] + 0.5), int(p[1]/p[-1] + 0.5)
            
            # Check if the source pixel is within bounds
            if px >= 0 and py >= 0 and px < src.shape[1] and py < src.shape[0]:
                dst[qy, qx] = src[py, px]
    
    return dst

def showTransformations(frame_number: int, homography: np.ndarray, features1: np.ndarray, features2: np.ndarray, mask: np.ndarray):
    """
    Displays the transformations between both pictures.
    """
    # Load the video
    mat_file = scipy.io.loadmat("used_frames.mat")

    # Print the keys (variable names) in the MATLAB file
    print("Variables in the MATLAB file:")
    print(mat_file.keys())

    frames = np.array(mat_file['frames'])
    image1 = frames[frame_number]
    image2 = frames[frame_number + 1]
    # Create an empty image to concatenate the two images side by side
    concatenated_image = np.zeros((max(frames[frame_number].shape[0], frames[frame_number + 1].shape[0]), frames[frame_number].shape[1] + frames[frame_number+1].shape[1], 3), dtype=np.uint8)

    # Copy the images into the concatenated image
    concatenated_image[:image1.shape[0], :image1.shape[1]] = image1
    concatenated_image[:image2.shape[0], image1.shape[1]:] = image2

    # Draw lines between corresponding keypoints
    for p1, p2 in zip(features1, features2):
        # Shift the x-coordinate for the second image since it's concatenated next to the first image
        p2[0] += image1.shape[1]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='green', linewidth=1)

    # Show the concatenated image with lines
    plt.imshow(concatenated_image)
    plt.show()

    # Draw the inliers (which are the points that were used to compute the homography)
    inlier_image = concatenated_image.copy()

    src_pts = features1[mask]
    dst_pts = features2[mask]

    for p1, p2 in zip(src_pts, dst_pts):
        # Draw a circle at each inlier
        plt.scatter(*p1, color='red')
        plt.scatter(*p2, color='red')
        
        # Draw a line between the inliers
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=1)

    # Show the image with inliers
    plt.imshow(inlier_image)
    plt.show()

    dst = warpPerspective(image2, homography, image2.shape[:2][::-1])
    # Draw the transformed image side by side with the first image
    with_transform = concatenated_image.copy()

    # Copy the transformed image into the empty space
    with_transform[:image2.shape[0], :image2.shape[1]] = image2
    with_transform[:dst.shape[0], dst.shape[1]:] = dst

    # Show the concatenated image with lines
    plt.imshow(with_transform)
    plt.show()

    exit()
