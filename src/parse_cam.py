import cv2
from facial_detection import detect_face_and_eyes, yawn_detection, reshape_eyes_for_model, predict_eyes
from facial_detection import EYES_OPEN, EYES_CLOSED
from generate_eyes_model import generate_model, TRAINED_PATH
from sound_manager import SoundManager
import os

CAPTURED = '../captured/'
YAWN_RATIO = 0.06

if not os.path.isdir(CAPTURED):
   os.mkdir(CAPTURED)


cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

frames = []
size = None
size = None
ret = False
trained_model = generate_model()
trained_model.load_weights(TRAINED_PATH)

closed_counter = 0
open_counter = 0
sleep_frames = 0
yawn_frames = 0

while True:
    ret, frame = cap.read()
    if ret:
        h, w, l = frame.shape
        size = (w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_frame, roi_gray, eyes, face_dims = detect_face_and_eyes(frame, gray)
        ratio, new_frame = yawn_detection(new_frame, face_dims)

        if ratio > YAWN_RATIO:
            yawn_frames = fps

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
                    sleep_frames = fps
                    closed_counter = 0
                    open_counter = 0

            closed_counter = 0 if closed_counter < 0 else closed_counter
            open_counter = 0 if open_counter < 0 else open_counter

            print('[open, closed, ratio] - [%s, %s, %s]' % (open_counter, closed_counter, ratio))

        frame_with_msg = new_frame
        # add a sleeping warning to the frame
        if sleep_frames > 0:
            if sleep_frames == fps:
                SoundManager(sleep_detected=True)
            cv2.putText(new_frame, 'Sleeping', (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2, cv2.LINE_AA)
            sleep_frames -= 1
        elif yawn_frames > 0:
            if yawn_frames == fps:
                SoundManager(sleep_detected=False)
            cv2.putText(new_frame, 'Drowsing', (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 140, 255), 2, cv2.LINE_AA)
            yawn_frames -= 1

        cv2.imshow('frame', new_frame)

        frames.append(new_frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break
    else:
        break

if ret:
    out = cv2.VideoWriter(CAPTURED + 'project.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

cap.release()
cv2.destroyAllWindows()
