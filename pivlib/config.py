import numpy as np


class Config():

    def __init__(self, config_file: str):
        self.config_dict = parse_config_file(config_file)
        self.videos: str = self.config_dict['videos'][0][0]
        self.keypoints_out: str = self.config_dict['keypoints_out'][0][0]
        self.transforms_out: str = self.config_dict['transforms_out'][0][0]
        self.transforms_type: str = self.config_dict['transforms'][0][0]
        self.transforms_params: str = self.config_dict['transforms'][0][1]
        self.pts_in_frame: np.ndarray = np.array(self.config_dict['pts_in_frame'])[:, 1:].astype(int).reshape(1,-1,2)  # TODO: check if this is correct
        self.pts_in_map: np.ndarray = np.array(self.config_dict['pts_in_map'])[:, 1:].astype(int).reshape(1,-1,2)
        self.frame_number: np.ndarray = np.array(self.config_dict['pts_in_frame'])[:, 0].astype(int)

    def show(self):
        print("videos:              ",self.videos)
        print("points in frame:     ",self.pts_in_frame)
        print("frame number:        ",self.frame_number)
        print("points in map:       ",self.pts_in_map)
        print("transforms type:     ",self.transforms_type)
        print("transforms params:   ",self.transforms_params)
        print("transforms out:      ",self.transforms_out)
        print("keypoints out:       ",self.keypoints_out)

def parse_config_file(file_path: str) -> dict:
    """Parse config file on file_path"""
    
    config_dict: dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Ignore comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Split the line into tokens
            tokens = line.split()
            
            # Extract parameter names and values
            param_name = tokens[0]
            param_values = [tokens[1:]]

            # Check if the token already exists in the dictionary
            if param_name in config_dict:
                # Add new values to the existing token
                config_dict[param_name].extend(param_values)
            else:
                # Create a new entry in the dictionary
                config_dict[param_name] = param_values

    return config_dict