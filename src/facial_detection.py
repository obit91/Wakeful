import cv2
from generate_eyes_model import WIDTH, HEIGHT, load_trained_model
import numpy as np

EYES_OPEN = 'open'
EYES_CLOSED = 'closed'

# The prediction threshold is very low because there are almost no false-positives for open eye detection, On the other
# hand, closed eye detection amounts to values far below 0.1.
PREDICTION_THRESHOLD = 0.1


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
    eyes = list()
    roi_gray = None
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # cv2.rectangle(img,(x, y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return img, roi_gray, eyes


def reshape_eyes_for_model(roi_gray, left_eye, right_eye):
    """
    Converts input eyes into the shape of the trained model.
    :param roi_gray: A roi of the original grayscale face.
    :param left_eye: The left eye roi.
    :param right_eye: The right eye roi.
    :return: The resized left and right eyes, according to the trained model shape.
    """
    left_eye_resized = fit_eye_for_cnn(roi_gray, left_eye)
    right_eye_resized = fit_eye_for_cnn(roi_gray, right_eye, True)
    return left_eye_resized, right_eye_resized


def fit_eye_for_cnn(roi_gray, eye, right_ind=False):
    """
    Crops the eye to the model shape, normalizes it and returns the eye in the shape trained.
    :param roi_gray: A roi of the face in grayscale.
    :param eye: The coordinates of an eye.
    :param right_ind: If this is a right eye we flip it, since we only trained on left eyes.
    :return: A resized normalized eye fitting for the model.
    """
    # crop the eye rectangle from the frame
    ex, ey, ew, eh = eye
    eye_rect = roi_gray[ey: ey + eh, ex:ex + ew]
    eye_resized = cv2.resize(eye_rect, (WIDTH, HEIGHT))
    if right_ind:
        eye_resized = cv2.flip(eye_resized, 1)

    # fit the eyes to the cnn
    normalized_eye = eye_resized.astype('float32')
    normalized_eye /= 255
    normalized_eye = np.expand_dims(normalized_eye, axis=2)
    normalized_eye = np.expand_dims(normalized_eye, axis=0)
    return normalized_eye


def predict_eyes(trained_model, left_eye, right_eye):
    """
    Predicts whether the eyes are open or closed.
    The prediction is based on the sum of predictions of both the left and the right eyes.
    :param trained_model: A trained model that classifies if eyes are closed or open.
    :param left_eye: The left eye of a grayscale face.
    :param right_eye: The right eye of a grayscale face.
    :return: 'open' if the eyes are open, 'closed' otherwise.
    """
    left_prediction = trained_model.predict(left_eye)
    right_prediction = trained_model.predict(right_eye)
    # print('[left, right]: [%s %s]' % (left_prediction, right_prediction))
    if left_prediction + right_prediction > PREDICTION_THRESHOLD:
        return EYES_OPEN
    else:
        return EYES_CLOSED

