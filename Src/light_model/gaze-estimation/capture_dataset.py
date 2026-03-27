import os
import time
import subprocess
from datetime import datetime
from PIL import Image

def capture_image(save_path, size=448):
    raw_path = save_path.replace(".jpg", "_raw.jpg")
    cmd = [
        "rpicam-still",
        "-n",
        "-t", "1",
        "--width", "640",
        "--height", "480",
        "-o", raw_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return False

    # Center crop 480x480
    try:
        img = Image.open(raw_path)
        w, h = img.size          # 640 x 480
        left = (w - size) // 2  # 80
        top  = (h - size) // 2  # 0
        img.crop((left, top, left + size, top + size)).save(save_path, quality=95)
        os.remove(raw_path)
        return True
    except Exception as e:
        print(f"Crop failed: {e}")
        return False


def collect_gaze_dataset(save_root="my_captures", images_per_direction=50):

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
