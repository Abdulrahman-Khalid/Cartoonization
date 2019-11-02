# how to use ?
# python facial_landmarks.py --shape-predictor shape_predictor.dat --image images/test1.jpg 
# could detect more than one face at the same image

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# initializes dlib’s pre-trained face detector
detector = dlib.get_frontal_face_detector()
# loads the facial landmark predictor using the path to the supplied --shape-predictor
predictor = dlib.shape_predictor(args["shape_predictor"])

# first decect face in image
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
