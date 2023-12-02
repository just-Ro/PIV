


class Progress():
    def __init__(self, goal = 100, title = "Progress:", display_title = False, display_fraction = False, display_percent = False, display_bar = False, bar_length = 50):
        
        self.string = title
        self.goal=goal
        self.pre_calc = 1/goal
        self.title = display_title
        self.frac=display_fraction
        self.per=display_percent
        self.bar=display_bar
        self.size=bar_length

    def getstr(self, iteration: int) -> str:
        
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
        print(f"\r{self.getstr(iteration)}", end='', flush=True)
        if iteration == self.goal:
            print()
        