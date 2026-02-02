#%%
from PIL import Image
import numpy as np
import onnxruntime
import cv2
import time
from datetime import datetime
import os

def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

#%%
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_sample(session, image_file, categories, result_file):
    output = session.run([], {'input':preprocess_image(image_file, 224, 224)})[0]
    output = output.flatten()
    output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:5]
    
    print(f"Results:")
    for catid in top5_catid:
        print(f"  {categories[catid]}: {output[catid]:.4f}")
    
    # write the result to a file
    with open(result_file, "w") as f:
        for catid in top5_catid:
            f.write(categories[catid] + " " + str(output[catid]) + "\n")

#%%
if __name__ == "__main__":
    # Get absolute path for capture folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    capture_folder = os.path.join(script_dir, "capture")
    os.makedirs(capture_folder, exist_ok=True)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Capture folder: {capture_folder}")
    
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    
    # Create Inference Session
    session = onnxruntime.InferenceSession(
        "mobilenet_v2_float.onnx",
        providers=["CPUExecutionProvider"]
    )
    
    print("\nStarting continuous prediction. Press Ctrl+C to stop.")
    print(f"Images and results will be saved to '{capture_folder}' folder\n")
    
    try:
        while True:
            print("="*50)
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"image_{timestamp}.jpg"
            result_filename = f"result_{timestamp}.txt"
            
            # Use absolute paths
            image_path = os.path.join(capture_folder, image_filename)
            result_path = os.path.join(capture_folder, result_filename)
            
            print(f"Capturing: {image_filename}")
            
            # Get capture from camera with absolute path
            import subprocess
            subprocess.run([
                "rpicam-still",
                "-o", image_path,
                "--width", "640",
                "--height", "480",
                "-n",           # no preview
                "-t", "1"       # timeout 1ms (immediate capture)
            ], check=True, capture_output=True, text=True)
            
            # Wait for file to be written
            time.sleep(0.3)
            
            # Check if file was created
            if not os.path.exists(image_path):
                print(f"ERROR: File not created at {image_path}")
                # Check current directory
                if os.path.exists(image_filename):
                    print(f"Found file in current directory, moving it...")
                    import shutil
                    shutil.move(image_filename, image_path)
                else:
                    print("Skipping this capture...")
                    time.sleep(3)
                    continue
            
            print(f"Image saved ({os.path.getsize(image_path)} bytes)")
            
            # Run inference
            run_sample(session, image_path, categories, result_path)
            print(f"Results saved\n")
            
            # Wait before next capture
            print("Waiting 3 seconds...\n")
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        print(f"All files saved in: {capture_folder}")
