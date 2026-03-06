#!/usr/bin/env python3
"""
Gaze Estimation with LED Control for Raspberry Pi
Using rpicam-still for image capture
"""

import cv2
import numpy as np
import time
import argparse
import sys
import os
import subprocess
from datetime import datetime
import mediapipe as mp

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("Warning: RPi.GPIO not found. Running in simulation mode.")
    GPIO = None

# Cấu hình GPIO pins cho 4 LED
LED_PINS = {
    'left': 17,    # GPIO 17
    'right': 27,   # GPIO 27 
    'up': 22,      # GPIO 22
    'down': 23     # GPIO 23
}

class LEDController:
    
    def __init__(self, use_gpio=True):
        self.use_gpio = use_gpio and GPIO is not None
        self.current_direction = None
        
        if self.use_gpio:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            for direction, pin in LED_PINS.items():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            
            print("GPIO initialized successfully")
            print(f"LED Pins: Left={LED_PINS['left']}, Right={LED_PINS['right']}, " 
                  f"Up={LED_PINS['up']}, Down={LED_PINS['down']}")
        else:
            print("Running in simulation mode (no GPIO)")
    
    def turn_off_all(self):
        if self.use_gpio:
            for pin in LED_PINS.values():
                GPIO.output(pin, GPIO.LOW)
        self.current_direction = None
    
    def set_direction(self, direction):
        """Bật LED theo hướng nhìn"""
        if direction == self.current_direction:
            return
        
        self.turn_off_all()
        
        if direction in LED_PINS:
            if self.use_gpio:
                GPIO.output(LED_PINS[direction], GPIO.HIGH)
            print(f"Direction: {direction.upper()}")
            self.current_direction = direction
    
    def cleanup(self):
        if self.use_gpio:
            self.turn_off_all()
            GPIO.cleanup()
            print("GPIO cleaned up")


class GazeEstimator:
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.input_size = (448, 448)
        
        # Threshold
        self.thresholds = {
            'yaw': 15,
            'pitch': 15
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            self.model = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            print(f"ONNX model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return False
    
    def preprocess(self, face_image):
        img = cv2.resize(face_image, self.input_size)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, face_image):
        input_tensor = self.preprocess(face_image)
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_tensor})
        
        # Outputs: [yaw_pred, pitch_pred, yaw_reg, pitch_reg]
        yaw_pred = outputs[0][0]
        pitch_pred = outputs[1][0]
        
        # Softmax
        yaw_exp = np.exp(yaw_pred - np.max(yaw_pred))
        yaw_prob = yaw_exp / np.sum(yaw_exp)
        
        pitch_exp = np.exp(pitch_pred - np.max(pitch_pred))
        pitch_prob = pitch_exp / np.sum(pitch_exp)
        
        # Get predicted bin
        yaw_bin = np.argmax(yaw_prob)
        pitch_bin = np.argmax(pitch_prob)
        
        # Convert bin to angle (90 bins, 4 degrees each)
        yaw = (yaw_bin * 4 - 180)
        pitch = (pitch_bin * 4 - 90)
        
        return pitch, yaw
    
    def estimate_gaze(self, face_image):
        if self.model is None:
            return None, None
        
        try:
            pitch, yaw = self.predict(face_image)
            return pitch, yaw
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None
    
    def get_direction(self, pitch, yaw):
        if abs(yaw) > self.thresholds['yaw']:
            if yaw > 0:
                return 'right'
            else:
                return 'left'
        
        if abs(pitch) > self.thresholds['pitch']:
            if pitch > 0:
                return 'down'
            else:
                return 'up'
        
        return 'center'


class FaceDetector:
    
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(
            model_selection=0,  # 0 = short range
            min_detection_confidence=0.5
        )
        print("MediaPipe Face Detector initialized")
    
    def detect(self, frame):
        """Detect face and return largest bounding box"""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if results.detections:
            h, w, _ = frame.shape
            
            boxes = []
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                
                boxes.append((x, y, bw, bh))
            
            # Biggest face
            face = max(boxes, key=lambda x: x[2] * x[3])
            return face
        
        return None


class RpiCameraCapture:
    
    def __init__(self, width=640, height=480, temp_file='temp_capture.jpg'):
        self.width = width
        self.height = height
        self.temp_file = temp_file
        
        self._check_rpicam()
    
    def _check_rpicam(self):
        """Check rpicam-still command"""
        try:
            result = subprocess.run(
                ['rpicam-still', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            print("rpicam-still found")
            return True
        except FileNotFoundError:
            print("rpicam-still not found!")
            print("Install with: sudo apt install rpicam-apps")
            return False
        except Exception as e:
            print(f" Error checking rpicam-still: {e}")
            return False
    
    def capture(self):
        try:
            # Run rpicam-still command
            result = subprocess.run([
                'rpicam-still',
                '-o', self.temp_file,
                '--width', str(self.width),
                '--height', str(self.height),
                '-n',  # no preview
                '-t', '1',  # timeout 1ms (chụp ngay)
                '--immediate'  # capture immediately
            ], 
            capture_output=True,
            timeout=5,
            check=True)
            
            if os.path.exists(self.temp_file):
                frame = cv2.imread(self.temp_file)
                return frame
            else:
                print(f"Capture file not found: {self.temp_file}")
                return None
                
        except subprocess.TimeoutExpired:
            print("Camera capture timeout")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Camera capture failed: {e}")
            return None
        except Exception as e:
            print(f"Capture error: {e}")
            return None
    
    def cleanup(self):
        try:
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='Gaze Estimation with LED Control - Using rpicam-still'
    )
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to ONNX model file')
    parser.add_argument('--width', type=int, default=640, 
                       help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, 
                       help='Camera height (default: 480)')
    parser.add_argument('--interval', type=float, default=0.5,
                       help='Interval between captures in seconds (default: 0.5)')
    parser.add_argument('--no-gpio', action='store_true', 
                       help='Disable GPIO (simulation mode)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print detailed info')
    parser.add_argument('--save-images', action='store_true',
                       help='Save captured images with timestamp')
    parser.add_argument('--save-dir', type=str, default='captures',
                       help='Directory to save images (default: captures)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("  Gaze Estimation LED Control - Raspberry Pi 4")
    print("  Using rpicam-still for image capture")
    print("="*70)
    
    if args.save_images:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Images will be saved to: {args.save_dir}/")
    
    # Khởi tạo các components
    print("\n[1/4] Initializing LED Controller...")
    led_controller = LEDController(use_gpio=not args.no_gpio)
    
    print("\n[2/4] Loading Gaze Estimation Model...")
    gaze_estimator = GazeEstimator(args.model)
    
    print("\n[3/4] Initializing Face Detector...")
    face_detector = FaceDetector()
    
    print("\n[4/4] Setting up Camera...")
    camera = RpiCameraCapture(width=args.width, height=args.height)
    
    print("\n" + "="*70)
    print("  System Ready! Press Ctrl+C to exit")
    print(f"  Capture interval: {args.interval}s")
    print("  Pipeline: CAPTURE → DETECT FACE → PREDICT GAZE → UPDATE LED")
    print("="*70 + "\n")
    
    frame_count = 0
    prediction_count = 0
    total_capture_time = 0
    total_prediction_time = 0
    start_time = time.time()
    
    try:
        while True:
            frame_count += 1
            loop_start = time.time()
            
            print(f"\n{'='*50}")
            print(f"Frame {frame_count}")
            
            # ========== STEP 1: CAPTURE ==========
            capture_start = time.time()
            frame = camera.capture()
            capture_time = (time.time() - capture_start) * 1000
            total_capture_time += capture_time
            
            if frame is None:
                print("Failed to capture image")
                time.sleep(args.interval)
                continue
            
            print(f"Image captured ({capture_time:.1f}ms)")
            
            # Lưu ảnh nếu được yêu cầu
            if args.save_images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                save_path = os.path.join(args.save_dir, f"frame_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
            
            # ========== STEP 2: FACE DETECTION ==========
            detect_start = time.time()
            face_bbox = face_detector.detect(frame)
            detect_time = (time.time() - detect_start) * 1000
            
            if face_bbox is not None:
                x, y, w, h = face_bbox
                face_image = frame[y:y+h, x:x+w]
                
                print(f"Face detected ({detect_time:.1f}ms) - Size: {w}x{h}")
                
                # ========== STEP 3: GAZE ESTIMATION ==========
                predict_start = time.time()
                pitch, yaw = gaze_estimator.estimate_gaze(face_image)
                predict_time = (time.time() - predict_start) * 1000
                total_prediction_time += predict_time
                
                if pitch is not None and yaw is not None:
                    prediction_count += 1
                    
                    print(f"Gaze predicted ({predict_time:.1f}ms)")
                    print(f" Pitch: {pitch:6.2f}° | Yaw: {yaw:6.2f}°")
                    
                    # ========== STEP 4: CONTROL LED WITH GAZE ==========
                    direction = gaze_estimator.get_direction(pitch, yaw)
                    
                    if direction != 'center':
                        led_controller.set_direction(direction)
                    else:
                        led_controller.turn_off_all()
                    
                    print(f"  Direction: {direction.upper()}")
                else:
                    print("Gaze prediction failed")
                    led_controller.turn_off_all()
            else:
                print(f"No face detected ({detect_time:.1f}ms)")
                led_controller.turn_off_all()
            
            # ========== STATISTICS ==========
            loop_time = (time.time() - loop_start) * 1000
            print(f"\nLoop time: {loop_time:.1f}ms")
            
            # Print statistics each 10 frames
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                avg_capture = total_capture_time / frame_count
                avg_prediction = total_prediction_time / max(prediction_count, 1)
                fps = frame_count / elapsed
                
                print(f"\n{'='*50}")
                print(f"Statistics (Frame {frame_count}):")
                print(f"  Avg Capture Time: {avg_capture:.1f}ms")
                print(f"  Avg Prediction Time: {avg_prediction:.1f}ms")
                print(f"  Success Rate: {prediction_count}/{frame_count} ({100*prediction_count/frame_count:.1f}%)")
                print(f"  Average FPS: {fps:.2f}")
                print(f"{'='*50}")
            
            # ========== WAIT INTERVAL ==========
            elapsed = time.time() - loop_start
            sleep_time = args.interval - elapsed
            
            if sleep_time > 0:
                if args.verbose:
                    print(f"Waiting {sleep_time:.2f}s...")
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\n Exiting gracefully...")
    
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        camera.cleanup()
        led_controller.cleanup()
        
        # Final statistics
        if frame_count > 0:
            elapsed = time.time() - start_time
            print("\n" + "="*70)
            print("  Final Statistics")
            print("="*70)
            print(f"  Total Runtime: {elapsed:.1f}s")
            print(f"  Total Frames: {frame_count}")
            print(f"  Total Predictions: {prediction_count}")
            print(f"  Success Rate: {100*prediction_count/frame_count:.1f}%")
            print(f"  Avg Capture Time: {total_capture_time/frame_count:.1f}ms")
            if prediction_count > 0:
                print(f"  Avg Prediction Time: {total_prediction_time/prediction_count:.1f}ms")
            print(f"  Average FPS: {frame_count/elapsed:.2f}")
            print("="*70)
        
        print("\n✓ All resources released")
        print("\nGoodbye!\n")


if __name__ == '__main__':
    main()
    
    
'''
# Change resolution
python3 gaze_led_rpi.py \
  --model weights/mobileone_s0_gaze.onnx \
  --width 640 --height 480

# Change interval between captures
python3 gaze_led_rpi.py \
  --model weights/mobileone_s0_gaze.onnx \
  --interval 0.3

# Save image
python3 gaze_led_rpi.py \
  --model weights/mobileone_s0_gaze.onnx \
  --save-images \
  --save-dir my_captures

# Verbose mode
python3 gaze_led_rpi.py \
  --model weights/mobileone_s0_gaze.onnx \
  --verbose

# Test no GPIO
python3 gaze_led_rpi.py \
  --model weights/mobileone_s0_gaze.onnx \
  --no-gpio


# Balance accuracy and 
python3 gaze_led_rpi.py \
  --model weights/mobileone_s0_gaze.onnx \
  --width 640 \
  --height 480 \
  --interval 0.5

'''
