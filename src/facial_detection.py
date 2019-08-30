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
    face_dims = None
    if len(faces) > 0:
        face_dims = (x, y, w, h) = faces[0]
        cv2.rectangle(img,(x, y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return img, roi_gray, eyes, face_dims


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


def yawn_detection(original_frame, face_dims):
    """
    Detects the mouth within a face.
    :param original_frame: The original frame (before grayscale and crop).
    :param face_dims: (x, y, w, h) dimensions of a face.
    :return:
    """
    if face_dims is None:
        return 0, original_frame

    (x, y, w, h) = face_dims

    # Isolate the ROI as the mouth region
    width_one_corner = int((x + (w / 4)))
    width_other_corner = int(x + ((3 * w) / 4))
    height_one_corner = int(y + (11 * h / 16))
    height_other_corner = int(y + h)

    # Indicate the region of interest as the mouth by highlighting it in the window.
    cv2.rectangle(original_frame, (width_one_corner, height_one_corner), (width_other_corner, height_other_corner),
                  (0, 0, 255), 2)

    # mouth region
    mouth_region = original_frame[height_one_corner:height_other_corner, width_one_corner:width_other_corner]

    # Area of the bottom half of the face rectangle
    rect_area = (w * h) / 2

    ratio = 0
    if len(mouth_region) > 0:
        ratio = mouth_threshold(mouth_region, rect_area)

    return ratio, original_frame


def mouth_threshold(mouth_region, rect_area):
    """
    Thresholds the image and converts it to binary
    :param mouth_region: Mouth region within the face (a slice of the face).
    :param rect_area: The area of the bottom half of the face.
    :return: A ratio between the mouth and the face.
    """
    imgray = cv2.equalizeHist(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY))
    ret, thresh = cv2.threshold(imgray, 64, 255, cv2.THRESH_BINARY)

    # Finds contours in a binary image
    # Constructs a tree like structure to hold the contours
    # Contouring is done by having the contoured region made by of small rectangles and storing only the end points
    # of the rectangle
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return_value = calculate_contours(mouth_region, contours)

    # return_value[0] => second_max_count
    # return_value[1] => Area of the contoured region.
    # second_max_count = return_value[0]
    contour_area = return_value[1]

    ratio = contour_area / rect_area

    # Draw contours in the image passed. The contours are stored as vectors in the array.
    # -1 indicates the thickness of the contours. Change if needed.
    # if isinstance(second_max_count, np.ndarray) and len(second_max_count) > 0:
    #    cv2.drawContours(mouth_region, [second_max_count], 0, (255, 0, 0), -1)

    return ratio


def calculate_contours(image, contours):
    """
    Find the second largest contour in the ROI;
    Largest is the contour of the bottom half of the face.
    Second largest is the lips and mouth when yawning.
    """
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    max_area = 0
    second_max = 0
    max_count = 0
    secondmax_count = 0
    for i in contours:
        count = i
        area = cv2.contourArea(count)
        if max_area < area:
            second_max = max_area
            max_area = area
            secondmax_count = max_count
            max_count = count
        elif (second_max < area):
            second_max = area
            secondmax_count = count

    return [secondmax_count, second_max]
