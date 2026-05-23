import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')

found = []
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"OK index {i}: {int(w)}x{int(h)}")
        found.append(i)
        cap.release()
    else:
        print(f"NO index {i}")
        cap.release()

print(f"FOUND: {found}")
