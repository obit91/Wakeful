import cv2


def get_path(file_name): return cv2.data.haarcascades + file_name


face_cascade = cv2.CascadeClassifier(get_path('haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(get_path('haarcascade_eye.xml'))


def detect_face_and_eyes(img, gray):

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = None
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x, y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return img, eyes

