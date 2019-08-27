import numpy as np
import cv2
from face_detect import detect_face_and_eyes

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
 
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new_frame, eyes = detect_face_and_eyes(frame, gray)

    cv2.imshow('frame', new_frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()