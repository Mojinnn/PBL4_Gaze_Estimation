import os
import subprocess
from datetime import datetime


def capture_video(save_path, width=640, height=480, framerate=20):
    """
    Capture smooth video until Ctrl+C
    """
    cmd = [
        "rpicam-vid",
        "-n",
        "-t", "0",
        "--width", str(width),
        "--height", str(height),
        "--framerate", str(framerate),
        "--bitrate", "10000000",
        "--intra", "30",        
        "--flush",                       
        "--codec", "h264",
        "--level", "4.2",               
        "-o", save_path
    ]
    try:
        print(f"Recording... Press Ctrl+C to stop")
        print(f"Saving to: {save_path}")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print(f"\nRecording stopped")
        print(f"Saved: {save_path}")
    except subprocess.CalledProcessError as e:
        print(f"Capture failed: {e.stderr.decode()}")


if __name__ == "__main__":
    os.makedirs("data_video", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("data_video", f"video_{timestamp}.h264")
    capture_video(save_path)
