#!/usr/bin/env python3.7
import datetime
import argparse
from imutils import video, resize
import time
import cv2

import dlib_detector
import custom_detector

# args
args = argparse.ArgumentParser()
args.add_argument('-m', '--model', required=True,
                  help='path to facial landmark model')
args.add_argument('--algo', required=True, choices=['DLIB', 'CUSTOM'])
args = args.parse_args()

# choose detector
detector = dlib_detector.DlibDetector if args.algo == 'DLIB' else custom_detector.CustomDetector
detector = detector(args.model)

# video stream
vs = video.VideoStream().start()

# loop until user exits
while (cv2.waitKey(1) & 0xFF) != ord('q'):
    # frame -> resize -> gray
    frame = vs.read()
    frame = resize(frame, width=500)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.extract_faces(frame)

    for face in faces:
        for (x, y) in face:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('frame', frame)

# cleanup
cv2.destroyAllWindows()
vs.stop()
