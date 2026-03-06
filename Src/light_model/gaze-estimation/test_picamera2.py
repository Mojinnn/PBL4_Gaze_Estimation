from picamera2 import Picamera2
import cv2
import time

print("Initializing camera...")
picam2 = Picamera2()

# Configure camera
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)

print("Starting camera...")
picam2.start()
time.sleep(2)

print("Capturing 10 frames...")
for i in range(10):
    frame = picam2.capture_array()
    print(f"Frame {i+1}: shape={frame.shape}")
    time.sleep(0.1)

print("Saving test image...")
frame = picam2.capture_array()
# Convert RGB to BGR for OpenCV
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imwrite('test_picamera2.jpg', frame_bgr)
print("✓ Image saved: test_picamera2.jpg")

picam2.stop()
print("✓ Test completed successfully!")
