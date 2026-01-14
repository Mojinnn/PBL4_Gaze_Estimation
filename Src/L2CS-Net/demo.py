# import argparse
# import pathlib
# import numpy as np
# import cv2
# import time

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torchvision import transforms
# import torch.backends.cudnn as cudnn
# import torchvision

# from PIL import Image
# from PIL import Image, ImageOps

# from face_detection import RetinaFace

# from l2cs import select_device, draw_gaze, getArch, Pipeline, render

# CWD = pathlib.Path.cwd()

# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(
#         description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
#     parser.add_argument(
#         '--device',dest='device', help='Device to run model: cpu or gpu:0',
#         default="cpu", type=str)
#     parser.add_argument(
#         '--snapshot',dest='snapshot', help='Path of model snapshot.', 
#         default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
#     parser.add_argument(
#         '--cam',dest='cam_id', help='Camera device id to use [0]',  
#         default=0, type=int)
#     parser.add_argument(
#         '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
#         default='ResNet50', type=str)

#     args = parser.parse_args()
#     return args

# if __name__ == '__main__':
#     args = parse_args()

#     cudnn.enabled = True
#     arch=args.arch
#     cam = args.cam_id
#     # snapshot_path = args.snapshot

#     gaze_pipeline = Pipeline(
#         weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
#         arch='ResNet50',
#         device = select_device(args.device, batch_size=1)
#     )
     
#     cap = cv2.VideoCapture(cam)

#     # Check if the webcam is opened correctly
#     if not cap.isOpened():
#         raise IOError("Cannot open webcam")

#     with torch.no_grad():
#         while True:

#             # Get frame
#             success, frame = cap.read()    
#             start_fps = time.time()  

#             if not success:
#                 print("Failed to obtain frame")
#                 time.sleep(0.1)

#             # Process frame
#             results = gaze_pipeline.step(frame)

#             # Visualize output
#             frame = render(frame, results)
           
#             myFPS = 1.0 / (time.time() - start_fps)
#             cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

#             cv2.imshow("Demo",frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             success,frame = cap.read()  
    

import argparse
import cv2
import numpy as np
import torch
from l2cs import Pipeline, render

def parse_args():
    parser = argparse.ArgumentParser(description='Gaze estimation demo')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu or cuda:0)')
    parser.add_argument('--snapshot', type=str, required=True, help='Path to model snapshot')
    parser.add_argument('--cam', type=int, default=0, help='Camera ID')
    parser.add_argument('--arch', type=str, default='ResNet50', help='Model architecture')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize pipeline
    print(f"Loading model from {args.snapshot}...")
    print(f"Using device: {args.device}")
    
    try:
        gaze_pipeline = Pipeline(
            weights=args.snapshot,
            arch=args.arch,
            device=torch.device(args.device)
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open camera
    print(f"Opening camera {args.cam}...")
    cap = cv2.VideoCapture(args.cam)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.cam}")
        return
    
    print("Camera opened successfully!")
    print("Press 'q' to quit")
    print("\nLooking for faces...")
    
    frame_count = 0
    faces_detected_last = False
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame from camera")
            break
        
        frame_count += 1
        
        try:
            # Process frame
            results = gaze_pipeline.step(frame)
            
            # Check if faces were detected by checking the bboxes attribute
            if results is not None and hasattr(results, 'bboxes') and len(results.bboxes) > 0:
                if not faces_detected_last:
                    print(f"Face detected! (frame {frame_count})")
                    faces_detected_last = True
                
                # Render gaze estimation
                frame = render(frame, results)
            else:
                # No faces detected
                if faces_detected_last:
                    print(f"Lost face at frame {frame_count}")
                    faces_detected_last = False
                
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Frame {frame_count}: Still looking for faces...")
                
                # Display message on frame
                cv2.putText(frame, "No face detected - Position yourself in front of camera", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        except ValueError as e:
            # Handle the "need at least one array to stack" error
            if "need at least one array to stack" in str(e):
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: No face detected")
                cv2.putText(frame, "No face detected - Move closer to camera", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                print(f"Error processing frame: {e}")
        
        except Exception as e:
            if frame_count % 30 == 0:  # Only print occasionally to avoid spam
                print(f"Error: {e}")
        
        # Display the frame
        cv2.imshow('L2CS-Net Gaze Estimation', frame)
        
        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQuitting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == '__main__':
    main()