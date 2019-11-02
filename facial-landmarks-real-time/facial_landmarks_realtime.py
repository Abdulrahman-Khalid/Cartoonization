# how to use it
# python facial_landmarks_realtime.py --shape-predictor ../model/shape_predictor.dat

from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
arg = argparse.ArgumentParser()
arg.add_argument("-p", "--shape-predictor", required=True,
                 help="path to facial landmark predictor")
args = vars(arg.parse_args())

# initialize dlib’s HOG Linear SVM-based face detector
# and then load the facial landmark predictor from disk
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream
vs = VideoStream().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the video stream, resize it to
    # 500 pixels, and convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the faces that have been detected
    for rect in rects:
        # determine the facial landmarks for the every face region, then convert
        # the facial landmark (x, y) to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y) for the facial landmarks and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the "q" key was pressed, break the loop
    if key == ord("q"):
        break

# cleanup
cv2.destroyAllWindows()
vs.stop()
