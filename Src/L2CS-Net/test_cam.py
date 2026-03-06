import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
print("ret:", ret)

if ret:
    print("Frame shape:", frame.shape)

cap.release()
