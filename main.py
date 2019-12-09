#!/usr/bin/env python3.7
import sys
import datetime
import argparse
import time
import cv2
import math
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

import dlib_detector
import custom_detector
import gen_ui
import emoji_window


def cirle_features(frame, faces):
    for face in faces:
        for (x, y) in face:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    return frame


def ndarray_to_qimage(image: np.ndarray):
    height, width, colors = image.shape
    bytesPerLine = 3 * width

    image = QtGui.QImage(image.data,
                         width,
                         height,
                         bytesPerLine,
                         QtGui.QImage.Format_RGB888)

    image = image.rgbSwapped()
    return image


class VideoRecorder(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()

        if not read:
            print('Error reading from the webcam', file=sys.stderr)
            exit(1)

        self.image_data.emit(cv2.flip(data, 1))


class FeaturesFrameWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.image = QtGui.QImage()

    def update_img(self, image, faces):
        self.image = cirle_features(image, faces)
        self.image = ndarray_to_qimage(self.image)

        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class CartoonizationWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.image = QtGui.QImage()

    def update_img(self, image, gray_image, faces):
        # TODO: edit image with stickers
        self.image = ndarray_to_qimage(image)

        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class Window(gen_ui.Ui_MainWindow):
    def __init__(self, args, emoji_world, detector):
        self.parent = QtWidgets.QMainWindow()

        self.setupUi(self.parent)

        self.detector = detector

        # widgets
        self.widgetFrame = FeaturesFrameWidget(self.widgetFrame)
        self.widget2D = CartoonizationWidget(self.widget2D)
        self.emoji_world = emoji_world

        # fps calculations
        self.fps_sum = 0
        self.num_frames = 0
        self.max_fps = 0
        self.min_fps = math.inf

        # recorder
        self.vr = VideoRecorder()
        self.vr.image_data.connect(self.image_data_slot)
        self.vr.start_recording()

    def show(self):
        self.parent.show()

    def image_data_slot(self, frame):
        time_start = time.time()

        # frame -> resize -> gray
        frame = cv2.resize(frame, (500, int(
            500/frame.shape[1]*frame.shape[0])), interpolation=cv2.INTER_AREA)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detection
        faces = self.detector.extract_faces(gray_frame)

        # update widgets
        self.widgetFrame.update_img(frame.copy(), faces)
        self.widget2D.update_img(frame.copy(), gray_frame.copy(), faces)
        self.emoji_world.update_img(gray_frame, faces)

        self.update_fps(time_start)

    def update_fps(self, time_start):
        fps = 1 / (time.time() - time_start)
        self.fps_sum += fps
        self.num_frames += 1
        self.max_fps = max(self.max_fps, fps)
        self.min_fps = min(self.min_fps, fps)

        fps_msg = 'Avg FPS = {:3.2f}, Max FPS = {:3.2f}, Min FPS = {:3.2f}'.format(
            self.fps_sum/self.num_frames, self.max_fps, self.min_fps)

        print(fps_msg, '\r', end='')
        self.statusbar.showMessage(fps_msg)


# parse args
args = argparse.ArgumentParser()
args.add_argument('-m', '--model', required=True,
                  help='path to facial landmark model')
args.add_argument('--algo', required=True, choices=['dlib', 'custom'])
args = args.parse_args()

# choose detector
detector = dlib_detector.DlibDetector if args.algo == 'dlib' else custom_detector.CustomDetector
detector = detector(args.model)

# define app
app = QtWidgets.QApplication(sys.argv)

em_window = QtWidgets.QMainWindow()
em_window.setGeometry(-1, -1, 411, 301)
emoji_world = emoji_window.MyWorld()
pandaWidget = emoji_window.QPanda3DWidget(emoji_world)
em_window.setCentralWidget(pandaWidget)

ui = Window(args, emoji_world, detector)

# run app
em_window.show()
ui.show()
exit_code = app.exec_()
print()  # to keep the fps line
sys.exit(exit_code)
