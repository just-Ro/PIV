import time


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

