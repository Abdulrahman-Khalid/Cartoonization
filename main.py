import datetime
import argparse
from imutils import video, face_utils, resize
import time
import dlib
import cv2

# args
args = argparse.ArgumentParser()
args.add_argument("-m", "--model", required=True,
                  help="path to facial landmark model")
args = args.parse_args()

# initialize dlib’s HOG Linear SVM-based face detector
detector = dlib.get_frontal_face_detector()

# load the facial landmark predictor from disk
predictor = dlib.shape_predictor(args.model)

# video stream
vs = video.VideoStream().start()

# loop until user exits
while (cv2.waitKey(1) & 0xFF) != ord("q"):
    # frame -> resize -> gray
    frame = vs.read()
    frame = resize(frame, width=500)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = detector(gray_frame, 0)

    for face in faces:
        # determine the facial landmarks
        face = predictor(gray_frame, face)

        # facial landmarks (x, y) -> np array
        face = face_utils.shape_to_np(face)

        for (x, y) in face:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # show the frame
    cv2.imshow("Frame", frame)

# cleanup
cv2.destroyAllWindows()
vs.stop()
