import cv2
from . import generate_eyes_model

def get_path(file_name): return cv2.data.haarcascades + file_name


face_cascade = cv2.CascadeClassifier(get_path('haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(get_path('haarcascade_eye.xml'))


def detect_face_and_eyes(img, gray):
    """
    Receives an image and a grayscale version of it as an input.
    Draws a rectangle over the first face and pair of eyes within it.
    :param img: An image we want to analyze.
    :param gray: A grayscale version of the image input.
    :return: The image along with the coordinates of the eyes.
    """
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = None
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(img,(x, y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        return img, eyes
    return None, None


def get_eyes(eyes):
    """
    Receives the roi of eyes and returns them only if they are one pair.
    :param eyes: a list of eye rois.
    :return: A tuple containing the left and right eyes from a single face.
    """
    left_eye = None
    right_eye = None
    if len(eyes) == 2:
        left_eye = eyes[0]
        right_eye = eyes[1]
    return left_eye, right_eye


def reshape_eyes_for_model(roi_gray, left_eye, right_eye):
    """
    Converts input eyes into the shape of the trained model.
    :param roi_gray: A roi of the original grayscale face.
    :param left_eye: The left eye roi.
    :param right_eye: The right eye roi.
    :return: The resized left and right eyes, according to the trained model shape.
    """
    # crop the eye rectangle from the frame
    ex, ey, ew, eh = left_eye
    left_eye_rect = roi_gray[ey: ey + eh, ex:ex + ew]
    left_eye_resized = cv2.resize(left_eye_rect, (generate_eyes_model.WIDTH, generate_eyes_model.HEIGHT))

    ex, ey, ew, eh = right_eye
    # crop the eye rectangle from the frame
    right_eye_rect = roi_gray[ey: ey + eh, ex:ex + ew]
    right_eye_resized = cv2.resize(right_eye_rect, (generate_eyes_model.WIDTH, generate_eyes_model.HEIGHT))

    return left_eye_resized, right_eye_resized
