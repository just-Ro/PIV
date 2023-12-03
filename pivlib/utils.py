


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
    
    def __init__(self, goal = 100, title = "Progress:", display_title = False, display_fraction = False, display_percent = False, display_bar = False, bar_length = 50):
        """
        Initializes the Progress object with specified parameters.

        
        Parameters:
        -
        - param goal: The total number of iterations (default is 100).
        - param title: Title to display before the progress information (default is "Progress:").
        - param display_title: Whether to display the title (default is False).
        - param display_fraction: Whether to display the iteration fraction (e.g., "3/100") (default is False).
        - param display_percent: Whether to display the progress percentage (default is False).
        - param display_bar: Whether to display a progress bar (default is False).
        - param bar_length: Length of the progress bar (default is 50).
        """
        
        self.string = title
        self.goal=goal
        self.pre_calc = 1/goal
        self.title = display_title
        self.frac=display_fraction
        self.per=display_percent
        self.bar=display_bar
        self.size=bar_length

    def getstr(self, iteration: int) -> str:
        """
        Returns a formatted string representing the progress information at a given iteration.

        Parameters:
        -
        - param iteration: The current iteration number.
        
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
            string += f"{100*iteration*self.pre_calc:.2f}% "
        if self.bar:
            progress = int(self.size*iteration*self.pre_calc)
            string += f"[{'■' * progress}{'□' * (self.size - progress)}]"

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

