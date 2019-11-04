#!/usr/bin/env python3.7
import datetime
import argparse
from imutils import video, resize
import time
import cv2
import math

import dlib_detector
import custom_detector

# args
args = argparse.ArgumentParser()
args.add_argument('-m', '--model', required=True,
                  help='path to facial landmark model')
args.add_argument('--algo', required=True, choices=['dlib', 'custom'])
args = args.parse_args()

# choose detector
detector = dlib_detector.DlibDetector if args.algo == 'dlib' else custom_detector.CustomDetector
detector = detector(args.model)

# video stream
vs = video.VideoStream().start()

fps_sum = 0
num_frames = 0
max_fps = 0
min_fps = math.inf

# loop until user exits
while (cv2.waitKey(1) & 0xFF) != ord('q'):
    time_start = time.time()

    # frame -> resize -> gray
    frame = vs.read()
    frame = resize(frame, width=500)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.extract_faces(frame)

    for face in faces:
        for (x, y) in face:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('frame', frame)

    fps = 1 / (time.time() - time_start)
    fps_sum += fps
    num_frames += 1
    max_fps = max(max_fps, fps)
    min_fps = min(min_fps, fps)

    print('Avg FPS = {:3.2f}, Max FPS = {:3.2f}, Min FPS = {:3.2f}\r'.format(
        fps_sum/num_frames, max_fps, min_fps), end='')

# cleanup
cv2.destroyAllWindows()
vs.stop()

print()
