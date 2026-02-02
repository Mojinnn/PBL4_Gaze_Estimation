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
    
    print(f'Results for {image_file}:')
    for catid in top5_catid:
        print(categories[catid], output[catid])
    
    # write the result to a file
    with open(result_file, "w") as f:
        for catid in top5_catid:
            f.write(categories[catid] + " " + str(output[catid]) + " \r")



#%%
if __name__ == "__main__":
    capture_folder = "capture"
    os.makedirs(capture_folder, exist_ok=True)
    
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Create Inference Session
    session = onnxruntime.InferenceSession(
        "mobilenet_v2_float.onnx",
        providers=["CPUExecutionProvider"]
    )

    # Get capture from camera
    #import subprocess

    #subprocess.run([
    #    "rpicam-still",
    #    "-o", image_path,
    #    "--width", "640",
    #    "--height", "480",
    #    "-n"            # no preview
    #], check=True)

    # Run inference
    #run_sample(session, "capture.jpg", categories)

    print('Starting continuous prediction. Press Ctrl+C to stop.')
    print(f'Images and results will be saved to {capture_folder} folder')
    
    try:
        while True:
            print('\n'+'='*50)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"image_{timestamp}.jpg"
            result_filename = f"result_{timestamp}.txt"
            
            image_path = os.path.join(capture_folder, image_filename)
            result_path = os.path.join(capture_folder, result_filename)
            
            print(f'Capturing image:{image_filename}')
            
            import subprocess
            subprocess.run([
                "rpicam-still",
                "-o", "capture.jpg",
                "--width", "640",
                "--height", "480",
                "-n"
            ], check=True)
            
            if os.path.exists(image_path):
                print(f"Image saved successfully: {image_path}")
                file_size = os.path.getsize(image_path)
                print(f"File size: {file_size} bytes")
            else:
                print(f"ERROR: Image file not found at {image_path}")
                print("Skipping this capture...")
                time.sleep(3)
                continue
            
            time.sleep(0.5)
            
            print('Running prediction: ')
            run_sample(session, image_path, categories, result_path)
            print(f"Results saved to: {result_filename}")
            
            print("\nWaiting 3 seconds before next capture...")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n\nStopped by user. Exiting...")
        print(f"All captures saved in '{capture_folder}' folder")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
            
