import os
import time
import subprocess
from datetime import datetime


def capture_image(save_path):
    """
    Capture image using rpicam-still
    """
    cmd = [
        "rpicam-still",
        "-n",                 # no preview
        "-t", "1",            # almost instant capture
        "-o", save_path
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def collect_gaze_dataset(save_root="data", images_per_direction=50):

    directions = ["left", "right", "up", "down", "center"]

    # Create directories
    for direction in directions:
        os.makedirs(os.path.join(save_root, direction), exist_ok=True)

    print("\nDataset collection started")
    print(f"Images per direction: {images_per_direction}")
    print("Look at the requested direction when prompted\n")

    for direction in directions:

        input(f"\nPress ENTER when ready for: {direction.upper()}")

        print(f"Capturing {images_per_direction} images for {direction}...")

        for i in range(images_per_direction):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            filename = f"{direction}_{timestamp}.jpg"

            save_path = os.path.join(save_root, direction, filename)

            success = capture_image(save_path)

            if success:
                print(f"[{i+1}/{images_per_direction}] Saved: {save_path}")
            else:
                print("Capture failed")

            time.sleep(0.3)

        print(f"{direction.upper()} collection completed")

    print("\nDataset collection finished")


if __name__ == "__main__":
    collect_gaze_dataset(images_per_direction=50)
