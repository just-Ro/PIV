import subprocess

def call_process_video():
    try:
        subprocess.run(["python", "process_video.py", "config/config1.cfg"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error calling process_video.py: {e}")

def call_compute_transform():
    try:
        subprocess.run(["python", "compute_transform.py", "config/config1.cfg"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error calling compute_transform.py: {e}")

if __name__ == '__main__':
    # run process_video.py
    call_process_video()
    
    # run compute_transform.py
    call_compute_transform()