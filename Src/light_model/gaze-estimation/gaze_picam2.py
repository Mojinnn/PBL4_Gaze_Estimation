import cv2
import numpy as np
import time
import argparse
import sys
import os

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not found!")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not found. Running in simulation mode.")

# GPIO pins
LED_PINS = {'left': 17, 'right': 27, 'up': 22, 'down': 23}

class LEDController:
    def __init__(self, use_gpio=True):
        self.use_gpio = use_gpio and GPIO_AVAILABLE
        self.current_direction = None
        
        if self.use_gpio:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for direction, pin in LED_PINS.items():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            print("✓ GPIO initialized")
        else:
            print("✓ Simulation mode (no GPIO)")
    
    def turn_off_all(self):
        if self.use_gpio:
            for pin in LED_PINS.values():
                GPIO.output(pin, GPIO.LOW)
        self.current_direction = None
    
    def set_direction(self, direction):
        if direction == self.current_direction:
            return
        self.turn_off_all()
        if direction in LED_PINS:
            if self.use_gpio:
                GPIO.output(LED_PINS[direction], GPIO.HIGH)
            print(f"→ {direction.upper()}")
            self.current_direction = direction
    
    def cleanup(self):
        if self.use_gpio:
            self.turn_off_all()
            GPIO.cleanup()

class GazeEstimator:
    def __init__(self, model_path, model_type='onnx'):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.input_size = (224, 224)
        self.thresholds = {'yaw': 15, 'pitch': 15}
        self._load_model()
    
    def _load_model(self):
        if self.model_type == 'onnx':
            try:
                import onnxruntime as ort
                self.model = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                print(f"✓ Model loaded: {self.model_path}")
                return True
            except Exception as e:
                print(f"✗ Model error: {e}")
                return False
    
    def preprocess(self, face_image):
        img = cv2.resize(face_image, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)
    
    def estimate_gaze(self, face_image):
        if self.model is None:
            return None, None
        try:
            input_tensor = self.preprocess(face_image)
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: input_tensor})
            return float(outputs[0][0]), float(outputs[1][0])
        except:
            return None, None
    
    def get_direction(self, pitch, yaw):
        if abs(yaw) > self.thresholds['yaw']:
            return 'right' if yaw > 0 else 'left'
        if abs(pitch) > self.thresholds['pitch']:
            return 'down' if pitch > 0 else 'up'
        return 'center'

class FaceDetector:
    def __init__(self):
        possible_paths = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ]

        cascade_path = None
        for p in possible_paths:
            if os.path.exists(p):
                cascade_path = p
                break

        if cascade_path is None:
            raise RuntimeError(
                "Haar cascade not found.\n"
                "Try: sudo apt install python3-opencv"
            )

        self.detector = cv2.CascadeClassifier(cascade_path)
        print(f"✓ Face detector loaded: {cascade_path}")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        return max(faces, key=lambda x: x[2] * x[3]) if len(faces) > 0 else None

def main():
    parser = argparse.ArgumentParser(description='Gaze Estimation LED Control')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--type', type=str, default='onnx', choices=['onnx', 'pytorch'])
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--skip-frames', type=int, default=2)
    parser.add_argument('--no-gpio', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    print("="*60)
    print("  Gaze Estimation LED Control - Pi Camera")
    print("="*60)
    
    if not PICAMERA2_AVAILABLE:
        print("✗ picamera2 not installed!")
        print("Run: sudo apt-get install -y python3-picamera2")
        return
    
    # Initialize components
    led = LEDController(use_gpio=not args.no_gpio)
    gaze = GazeEstimator(args.model, args.type)
    face_det = FaceDetector()
    
    # Initialize camera
    print(f"\n✓ Opening camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={"FrameRate": args.fps}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print(f"✓ Camera ready: {args.width}x{args.height} @ {args.fps}fps")
    
    print("\n" + "="*60)
    print("  Running! Press Ctrl+C to exit")
    print("="*60 + "\n")
    
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            frame_count += 1
            if frame_count % args.skip_frames != 0:
                continue
            
            # Detect face
            face_bbox = face_det.detect(frame)
            
            if face_bbox is not None:
                x, y, w, h = face_bbox
                face_image = frame[y:y+h, x:x+w]
                
                # Estimate gaze
                pitch, yaw = gaze.estimate_gaze(face_image)
                
                if pitch is not None and yaw is not None:
                    direction = gaze.get_direction(pitch, yaw)
                    
                    if direction != 'center':
                        led.set_direction(direction)
                    else:
                        led.turn_off_all()
                    
                    if args.verbose:
                        print(f"Pitch: {pitch:6.2f}°, Yaw: {yaw:6.2f}° → {direction.upper()}")
            else:
                led.turn_off_all()
            
            # FPS counter
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_counter / (time.time() - fps_start)
                print(f"FPS: {fps:.1f}")
                fps_counter = 0
                fps_start = time.time()
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n\n✓ Exiting...")
    
    finally:
        picam2.stop()
        picam2.close()
        led.cleanup()
        print("✓ Cleanup complete\n")

if __name__ == '__main__':
    main()
