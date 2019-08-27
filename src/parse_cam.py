import numpy as np
import cv2
from face_detect import detect_face_and_eyes

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

frames = []
size = None
size = None
while(True):
    ret, frame = cap.read()
    h, w, l = frame.shape
    size = (w, h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new_frame, eyes = detect_face_and_eyes(frame, gray)

    cv2.imshow('frame', new_frame)

    frames.append(new_frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

out = cv2.VideoWriter('../captured/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frames)):
    out.write(frames[i])
out.release()

cap.release()
cv2.destroyAllWindows()
