import time
import numpy as np


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
        
        self.prev_iteration = 0
        
        self.__pre_calc__ = 1/goal
        self.__start_time__ = time.time()
        self.__prev_time__ = self.__start_time__
        
        # Check if the terminal can print ■ and □
        if self.bar:
            try:
                print("\r■□",end='')
                print("\r  ",end='\r')
            except UnicodeEncodeError:
                self.bar=False
                print("Warning: Terminal does not support Unicode characters. Progress bar disabled.")


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
            if iteration == self.goal:
                string += f"[\033[92m{'■' * full_bar}{'□' * (self.size - full_bar)}\033[0m] "
            else:
                string += f"[\033[93m{'■' * full_bar}{'□' * (self.size - full_bar)}\033[0m] "
        if self.eta:

            self.__prev_time__ = time.time()
            elapsed_time = self.__prev_time__ - self.__start_time__

            remaining_time = elapsed_time / ((iteration if iteration>0 else self.goal)*self.__pre_calc__) - elapsed_time
            string += f"{int(remaining_time)}s  "
        
        return string
    
    def update(self, iteration: int=-1):
        """
        Prints the formatted progress string for a given iteration on the same line.
        Clears the line and prints a newline when the goal is reached.

        Parameters:
        -
        - param iteration: The current iteration number.
        """
        if iteration == -1:
            self.prev_iteration += 1
            iteration = self.prev_iteration
        
        print(f"\r{self.getstr(iteration)}", end='', flush=True)
        if iteration == self.goal:
            print("")

def addWeighted(src1: np.ndarray, alpha: float, src2: np.ndarray, beta: float, gamma: float = 0.0) -> np.ndarray:
    """
    Blends two images with specified weights and an optional bias.

    Parameters:
    -
    - src1: First input array (image).
    - alpha: Weight of the first image elements.
    - src2: Second input array (image).
    - beta: Weight of the second image elements.
    - gamma: Scalar added to each sum (optional, default is 0.0).

    Returns:
    -
    - dst: Resulting blended image.
    """
    # Ensure input arrays have the same shape
    assert src1.shape == src2.shape, "Input images must have the same shape"

    # Perform the weighted summation
    return np.clip(src1 * alpha + src2 * beta + gamma, 0, 255).astype(np.uint8)
