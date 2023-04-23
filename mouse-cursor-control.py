from imutils import face_utils
from utils import *
import numpy as np
import imutils
import dlib
import cv2
from aur import *

# Thresholds and consecutive frame length for triggering the mouse action.

MOUTH_AR_THRESH = 0.06
MOUTH_AR_CONSECUTIVE_FRAMES = 15
EYE_AR_THRESH = 0.22
EYE_AR_CONSECUTIVE_FRAMES = 12
WINK_AR_DIFF_THRESH = 0.02
WINK_AR_CLOSE_THRESH = 0.14
WINK_CONSECUTIVE_FRAMES = 10

# Initialize the frame counters for each action as well as
# booleans used to indicate if action is performed or not
MOUTH_COUNTER = 0
EYE_COUNTER = 0
WINK_COUNTER = 0
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
GREEN_COLOR = (0, 0, 255)
RED_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)
elbow = False
wrist = False
shoulder = False
s1 = 90
s3 = 70
s4 = 180
s5 = 90
s6 = 120
s7 = 30

# Initialize Dlib's face detector (HOG-based) and then create
# the facial landmark predictor
shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# Grab the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Video capture
vid = cv2.VideoCapture(0)

resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h
while True:

    # Grab the frame from the threaded video file stream, resize and convert it to grayscale
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(20) & 0xFF
        continue

    # Determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    mouth = shape[mStart:mEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    nose = shape[nStart:nEnd]

    # Because I flipped the frame, left is right, right is left.
    temp = leftEye
    leftEye = rightEye
    rightEye = temp

    # Average the mouth aspect ratio together for both eyes
    mar = mouth_aspect_ratio(mouth)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    nose_point = (nose[3, 0], nose[3, 1])

    # Compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

    for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
        cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)

    # Check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter

    if diff_ear > WINK_AR_DIFF_THRESH:

        if leftEAR < rightEAR:

            if leftEAR < EYE_AR_THRESH:
                WINK_COUNTER += 1

                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    print("tt")
                    elbow = True
                    wrist = False
                    shoulder = False

                    WINK_COUNTER = 0

        elif leftEAR > rightEAR:
            if rightEAR < EYE_AR_THRESH:
                WINK_COUNTER += 1

                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                    elbow = False
                    wrist = True
                    shoulder = False
                    WINK_COUNTER = 0
        else:
            WINK_COUNTER = 0
    else:
        if ear <= EYE_AR_THRESH:
            EYE_COUNTER += 1

            if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                shoulder = True
                elbow = False
                wrist = False
                EYE_COUNTER = 0
        else:
            EYE_COUNTER = 0
            WINK_COUNTER = 0

    if mar > MOUTH_AR_THRESH:
        MOUTH_COUNTER += 1

        if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
            INPUT_MODE = not INPUT_MODE
            MOUTH_COUNTER = 0
            ANCHOR_POINT = nose_point

    else:
        MOUTH_COUNTER = 0

    if INPUT_MODE:
        cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        x, y = ANCHOR_POINT
        nx, ny = nose_point
        w, h = 35, 35
        multiple = 1
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
        cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)
        dir = direction(nose_point, ANCHOR_POINT, w, h)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        drag = 18

        if shoulder:
            cv2.putText(frame, 'shoulder', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
            if dir == 'right' and s1 > 0:
                rotateservo(9, s1 - 2)
                s1 = s1 - 2
                print(s1)

                if s1 - 2 <= 0:
                    print(s1)
                    rotateservo(9, 0)
                    s1 = 0

            if dir == 'left' and s1 < 180:
                rotateservo(9, s1 + 2)
                s1 = s1 + 2
                print(s1)

                if s1 + 2 >= 180:
                    rotateservo(9, 180)
                    print(s1)
                    s1 = 180

            if dir == 'up' and s3 < 180:
                rotateservo(3, s3 + 2)
                s3 = s3 + 2
                print(s3)

                if s3 + 2 >= 180:
                    rotateservo(3, 180)
                    print(s3)
                    s3 = 180

            elif dir == 'down' and s3 > 0:
                rotateservo(3, s3 - 2)
                s3 = s3 - 2
                print(s3)

                if s3 - 2 <= 0:
                    print(s3)
                    rotateservo(3, 0)
                    s3 = 0
        if elbow:
            cv2.putText(frame, 'elbow', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
            if dir == 'left' and s5 > 0:
                rotateservo(5, s5 - 2)
                s5 = s5 - 2
                print(s5)

                if s5 - 2 <= 0:
                    print(s5)
                    rotateservo(5, 0)
                    s5 = 0

            if dir == 'right' and s5 < 180:
                rotateservo(5, s5 + 2)
                s5 = s5 + 2
                print(s5)

                if s5 + 2 >= 180:
                    rotateservo(5, 180)
                    print(s5)
                    s5 = 180

            elif dir == 'down' and s4 < 180:
                rotateservo(4, s4 + 2)
                s4 = s4 + 2
                print(s4)

                if s4 + 2 >= 180:
                    rotateservo(4, 180)
                    print(s4)
                    s4 = 180

            elif dir == 'up' and s4 > 0:
                rotateservo(4, s4 - 2)
                s4 = s4 - 2
                print(s4)

                if s4 - 2 <= 0:
                    print(s4)
                    rotateservo(4, 0)
                    s4 = 0

        if wrist:
            cv2.putText(frame, 'wrist', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
            if dir == 'right' and s7 > 0:
                rotateservo(7, s7 - 2)
                s7 = s7 - 2
                print(s7)

                if s7 - 2 <= 0:
                    print(s7)
                    rotateservo(7, 0)
                    s7 = 0

            if dir == 'left' and s7 < 180:
                rotateservo(7, s7 + 2)
                s7 = s7 + 2
                print(s7)

                if s7 + 2 >= 180:
                    rotateservo(7, 180)
                    print(s7)
                    s7 = 180

            elif dir == 'up' and s6 < 180:
                rotateservo(6, s6 + 2)
                s6 = s6 + 2
                print(s6)

                if s6 + 2 >= 180:
                    rotateservo(6, 180)
                    print(s6)
                    s6 = 180

            elif dir == 'down' and s6 > 0:
                rotateservo(6, s6 - 2)
                s6 = s6 - 2
                print(s6)

                if s6 - 2 <= 0:
                    print(s6)
                    rotateservo(6, 0)
                    s6 = 0
    # Show the frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(20) & 0xFF

    # If the `Esc` key was pressed, break from the loop
    if key == 27:
        break

# Cleanup
cv2.destroyAllWindows()
vid.release()
