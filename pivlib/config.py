import numpy as np


class Config():

    def __init__(self, config_file: str):
        # Config1
        self.config_dict = parse_config_file(config_file)
        self.videos: str = self.config_dict['videos'][0]
        self.keypoints_out: str = self.config_dict['keypoints_out'][0][0] if 'keypoints_out' in self.config_dict else ""
        self.transforms_out: str = self.config_dict['transforms_out'][0][0]
        self.transforms_type: str = self.config_dict['transforms'][0][0]
        self.transforms_params: str = self.config_dict['transforms'][0][1]
        self.frame_number: np.ndarray = np.array(self.config_dict['pts_in_frame'])[:, 0].astype(int)-1
        self.pts_in_frame: np.ndarray = np.array(self.config_dict['pts_in_frame'])[:, 1:].astype(int).reshape(self.frame_number.size,-1,2)
        self.pts_in_map: np.ndarray = np.array(self.config_dict['pts_in_map'])[:, 1:].astype(int).reshape(self.frame_number.size,-1,2)
        # Config2
        self.cams: int = int(self.config_dict['cams'][0][0]) if 'cams' in self.config_dict else 0
        self.intrinsics: np.ndarray = np.array(self.config_dict['intrinsics']).astype(float) if 'intrinsics' in self.config_dict else np.array([])
        if self.intrinsics.size > 0:
            self.intrinsics = np.array([convert_to_intrinsic_matrix(row) for row in self.intrinsics])

    def show(self):
        print("cams:                ",self.cams)
        print("videos:              ",self.videos)
        print("intrinsics:          ",self.intrinsics)
        print("points in frame:     ",self.pts_in_frame)
        print("frame number:        ",self.frame_number)
        print("points in map:       ",self.pts_in_map)
        print("transforms type:     ",self.transforms_type)
        print("transforms params:   ",self.transforms_params)
        print("transforms out:      ",self.transforms_out)
        print("keypoints out:       ",self.keypoints_out)

def convert_to_intrinsic_matrix(intrinsics):
    fx, fy, cx, cy = intrinsics
    intrinsic_matrix = np.array([[fx, 0,  cx],
                                 [0,  fy, cy],
                                 [0,  0,  1]])
    return intrinsic_matrix

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