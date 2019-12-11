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


def cut_src(src, x, y, xmax, ymax):
    if x > xmax or y > ymax:
        return None

    if x < 0:
        src = src[-x:, :, :]
        x = 0

    if y < 0:
        src = src[:, -y:, :]
        y = 0

    if x+src.shape[0] > xmax:
        src = src[:xmax-x, :, :]

    if y+src.shape[1] > ymax:
        src = src[:, :ymax-y, :]

    if src.shape[0] == 0 or src.shape[1] == 0:
        return None
    return src


def cut_dst(dst, src, x, y):
    x, y = max(x, 0), max(y, 0)
    return dst[x:x+src.shape[0], y:y+src.shape[1], :]


def img_blit(dst, src, cx=0, cy=0):
    x, y = cx-src.shape[0]//2, cy-src.shape[1]//2
    x, y = min(x, dst.shape[0]), min(y, dst.shape[1])

    # cut src
    src = cut_src(src, x, y, dst.shape[0], dst.shape[1])

    if src is None:
        return dst

    # cut dst
    final_dst = dst
    dst = cut_dst(dst, src, x, y)

    # alpha
    alpha_img = src[:, :, 3] / 255
    alpha_dst = 1 - alpha_img

    # blit
    for c in range(0, 3):
        dst[:, :, c] = (alpha_img * src[:, :, c] +
                        alpha_dst * dst[:, :, c])

    return final_dst


def img_scale(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)


def img_rotate_center(mat, angle):
    angle = -math.degrees(angle)

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))


def get_slope(a, b):
    ax, ay = a
    bx, by = b
    if ay == by:
        return 0
    return (ay-by)/(ax-bx)


def get_dist(a, b):
    ax, ay = a
    bx, by = b
    return math.sqrt((ay-by)**2+(ax-bx)**2)


def get_half(a, b):
    ax, ay = a
    bx, by = b
    return ((ay+by)//2, (ax+bx)//2)


def put_sticker(image, faces, p_glasses, p_mustache):
    for face in faces:
        dst = get_dist(face[36], face[45])

        scl = dst/150
        scl = scl if scl != 0 else 1

        slope = get_slope(face[46-1], face[43-1])

        glasses = img_rotate_center(p_glasses.copy(), slope)
        glasses = img_scale(glasses, scl, scl)
        image = img_blit(image, glasses, face[27][1], face[27][0])

        scl = dst/470
        scl = scl if scl != 0 else 1

        mustache = img_rotate_center(p_mustache.copy(), slope)
        mustache = img_scale(mustache, scl, scl)
        image = img_blit(image, mustache, *get_half(face[51], face[33]))

    return image


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

        self.glasses = cv2.imread('data/glasses.png', cv2.IMREAD_UNCHANGED)
        self.mustache = cv2.imread('data/mustache.png', cv2.IMREAD_UNCHANGED)

        assert self.glasses.shape[2] == 4
        assert self.mustache.shape[2] == 4

    def update_img(self, image, gray_image, faces):
        image = put_sticker(
            image, faces, self.glasses.copy(), self.mustache.copy())

        self.image = ndarray_to_qimage(image)

        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class Window(gen_ui.Ui_MainWindow):
    def __init__(self, args, detector):
        self.parent = QtWidgets.QMainWindow()

        self.setupUi(self.parent)

        self.detector = detector
        self.last_faces = [[(0, 0)] * 68]

        # widgets
        self.widgetFrame = FeaturesFrameWidget(self.widgetFrame)
        self.widget2D = CartoonizationWidget(self.widget2D)

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
        if len(faces) > 0:
            self.last_faces = faces

        # update widgets
        self.widgetFrame.update_img(frame.copy(), self.last_faces)
        self.widget2D.update_img(
            frame.copy(), gray_frame.copy(), self.last_faces)

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

ui = Window(args, detector)

# run app
ui.show()
exit_code = app.exec_()
print()  # to keep the fps line
sys.exit(exit_code)
