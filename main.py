import numpy as np
import scipy as sp
import sys
import configparser

def parse_config_file(file_path):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(file_path)

    # Access values
    videos_path = config.get('INPUT PARAMETERS', 'videos path')
    keypoints_out = config.get('OUTPUT PARAMETERS', 'keypoints_out')
    transforms_type = config.get('OUTPUT PARAMETERS', 'transforms')
    transforms_params = config.get('OUTPUT PARAMETERS', 'transforms_out')

    # Process optional parameters
    image_map = config.get('INPUT PARAMETERS', 'image_map', fallback=None)

    return {
        'videos_path': videos_path,
        'keypoints_out': keypoints_out,
        'transforms_type': transforms_type,
        'transforms_params': transforms_params,
        'image_map': image_map
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python myprogram.py config_file.cfg")
        sys.exit(1)

    # Get the configuration file path from the command-line argument
    config_file_path = sys.argv[1]
    
    config_data = parse_config_file(config_file_path)

    # Your program logic goes here using the config_data dictionary
    # For example, you can print the parsed values
    print(f'Videos Path: {config_data["videos_path"]}')
    print(f'Keypoints Output: {config_data["keypoints_out"]}')
    print(f'Transforms Type: {config_data["transforms_type"]}')
    print(f'Transforms Params: {config_data["transforms_params"]}')
    print(f'Image Map (optional): {config_data["image_map"]}')

if __name__=='__main__':
    main()