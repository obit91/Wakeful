import numpy as np
import cv2
from facial_detection import detect_face_and_eyes, reshape_eyes_for_model, predict_eyes, EYES_OPEN, EYES_CLOSED
from generate_eyes_model import generate_model, TRAINED_PATH

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

frames = []
size = None
size = None
ret = False
trained_model = generate_model()
trained_model.load_weights(TRAINED_PATH)

closed_counter = 0
open_counter = 0
display_frames = 0
while True:
    ret, frame = cap.read()
    if ret:
        h, w, l = frame.shape
        size = (w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_frame, roi_gray, eyes = detect_face_and_eyes(frame, gray)

        if len(eyes) == 2:
            left_eye = eyes[0]
            right_eye = eyes[1]
            left_eye_resized, right_eye_resized = reshape_eyes_for_model(roi_gray, left_eye, right_eye)
            result = predict_eyes(trained_model, left_eye_resized, right_eye_resized)

            prev_closed_counter = closed_counter

            if result == EYES_CLOSED:
                closed_counter += 1
                open_counter -= 1
            else:
                open_counter += 1
                closed_counter -= 1

            if closed_counter % fps == 0 and closed_counter > 0:
                if prev_closed_counter < closed_counter:
                    display_frames = fps
                    closed_counter = 0
                    open_counter = 0


            closed_counter = 0 if closed_counter < 0 else closed_counter
            open_counter = 0 if open_counter < 0 else open_counter

            print('[open, closed] - [%s, %s]' % (open_counter, closed_counter))

        if display_frames > 0:
            display_frames -= 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(new_frame, 'Sleeping', (80, 350), font, 4, (2500, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', new_frame)

        frames.append(new_frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    else:
        break

if ret:
    out = cv2.VideoWriter('../captured/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

cap.release()
cv2.destroyAllWindows()
