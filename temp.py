import cv2

for i in range(0, 32):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    ok, frame = cap.read()
    print(f"Video{i} -> opened={cap.isOpened()}, frame={ok}")
    cap.release()
