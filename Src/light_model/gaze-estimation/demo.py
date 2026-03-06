import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque
import argparse


class FaceDetector:
    """Simple face detector using OpenCV's Haar Cascade"""
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Haar cascade cho mắt
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        return faces


class GazeEstimator:
    """Gaze estimation using ONNX model"""
    def __init__(self, model_path, input_size=(224, 224)):
        """
        Initialize gaze estimator
        Args:
            model_path: Path to ONNX model
            input_size: Input size for the model (height, width)
        """
        # Tạo session với optimization cho RPi
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # RPi4 có 4 cores
        
        # Sử dụng CPU provider (tối ưu cho RPi)
        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        
        # Get output names
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Bins configuration for Gaze360 dataset
        self.num_bins = 90
        self.bin_width = 4
        
        print(f"Model loaded: {model_path}")
        print(f"Input name: {self.input_name}")
        print(f"Output names: {self.output_names}")
    
    def preprocess(self, face_img):
        """Preprocess face image for model input"""
        # Resize
        img = cv2.resize(face_img, self.input_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess(self, outputs):
        """Convert model outputs to gaze angles"""
        # Outputs: [yaw_pred, pitch_pred, yaw_reg, pitch_reg]
        yaw_pred = outputs[0][0]  # Classification output
        pitch_pred = outputs[1][0]  # Classification output
        
        # Softmax
        yaw_exp = np.exp(yaw_pred - np.max(yaw_pred))
        yaw_prob = yaw_exp / np.sum(yaw_exp)
        
        pitch_exp = np.exp(pitch_pred - np.max(pitch_pred))
        pitch_prob = pitch_exp / np.sum(pitch_exp)
        
        # Get predicted bin
        yaw_bin = np.argmax(yaw_prob)
        pitch_bin = np.argmax(pitch_prob)
        
        # Convert bin to angle
        yaw = (yaw_bin * self.bin_width - 180)
        pitch = (pitch_bin * self.bin_width - 90)
        
        return yaw, pitch
    
    def estimate(self, face_img):
        """Estimate gaze direction from face image"""
        # Preprocess
        input_data = self.preprocess(face_img)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # Postprocess
        yaw, pitch = self.postprocess(outputs)
        
        return yaw, pitch


def draw_gaze_arrow(frame, face_bbox, yaw, pitch, length=150, color=(0, 255, 0)):
    """Draw gaze direction arrow on frame"""
    x, y, w, h = face_bbox
    
    # Điểm bắt đầu (giữa khuôn mặt)
    start_x = x + w // 2
    start_y = y + h // 2
    
    # Chuyển đổi góc sang radian
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    
    # Tính điểm kết thúc của mũi tên
    dx = length * np.sin(yaw_rad) * np.cos(pitch_rad)
    dy = -length * np.sin(pitch_rad)
    
    end_x = int(start_x + dx)
    end_y = int(start_y + dy)
    
    # Vẽ mũi tên
    cv2.arrowedLine(frame, (start_x, start_y), (end_x, end_y), color, 3, tipLength=0.3)
    
    return frame


class FPSCounter:
    """FPS counter with moving average"""
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)
    
    def update(self):
        """Update with current timestamp"""
        self.timestamps.append(time.time())
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.timestamps) < 2:
            return 0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0
        
        return len(self.timestamps) / time_diff


def main():
    parser = argparse.ArgumentParser(description='Gaze Estimation Demo for Raspberry Pi 4')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height (default: 480)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Target FPS for camera (default: 15)')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without display (headless mode)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Gaze Estimation Demo for Raspberry Pi 4")
    print("=" * 50)
    
    # Initialize components
    print("\n[1/3] Initializing face detector...")
    face_detector = FaceDetector()
    
    print("[2/3] Loading gaze estimation model...")
    gaze_estimator = GazeEstimator(args.model)
    
    print("[3/3] Opening camera...")
    cap = cv2.VideoCapture(args.camera)
    
    # Set camera properties for RPi optimization
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm độ trễ
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("\nCamera opened successfully!")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Target FPS: {args.fps}")
    print("\nPress 'q' to quit\n")
    
    fps_counter = FPSCounter()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            frame_count += 1
            start_time = time.time()
            
            # Detect faces
            faces = face_detector.detect(frame)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_img = frame[y:y+h, x:x+w]
                
                if face_img.size == 0:
                    continue
                
                # Estimate gaze
                try:
                    yaw, pitch = gaze_estimator.estimate(face_img)
                    
                    # Draw results
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    draw_gaze_arrow(frame, (x, y, w, h), yaw, pitch)
                    
                    # Display angles
                    text = f"Yaw: {yaw:.1f} Pitch: {pitch:.1f}"
                    cv2.putText(frame, text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Update FPS counter
            fps_counter.update()
            current_fps = fps_counter.get_fps()
            
            # Display FPS and info
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            if not args.no_display:
                cv2.imshow('Gaze Estimation - Raspberry Pi 4', frame)
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                inference_time = (time.time() - start_time) * 1000
                print(f"Frame {frame_count}: FPS={current_fps:.1f}, "
                      f"Inference={inference_time:.1f}ms, Faces={len(faces)}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == '__main__':
    main()
