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

import hog_detector
import violajones_detector
import dlib_detector
import gen_ui
import constants


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
    return cv2.resize(img, None, fx=fx if fx != 0 else 1, fy=fy if fy != 0 else 1, interpolation=cv2.INTER_CUBIC)


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


def put_sticker(image, faces, p_glasses, p_mustache, p_hat, draw_hat, draw_glasses, draw_mustache):
    for face in faces:
        dst = get_dist(face[36], face[45])
        slope = get_slope(face[46-1], face[43-1])

        if draw_glasses:
            scl = dst/150
            glasses = img_rotate_center(p_glasses.copy(), slope)
            glasses = img_scale(glasses, scl, scl)
            image = img_blit(image, glasses, face[27][1], face[27][0])

        if draw_mustache:
            scl = dst/470
            mustache = img_rotate_center(p_mustache.copy(), slope)
            mustache = img_scale(mustache, scl, scl)
            image = img_blit(image, mustache, *get_half(face[51], face[33]))

        if draw_hat:
            scl = dst/800
            hat = img_rotate_center(p_hat.copy(), slope)
            hat = img_scale(hat, scl, scl)
            image = img_blit(image, hat, int(face[27][1]-dst), face[27][0])

    return image


def cirle_features(frame, faces):
    for face in faces:
        for (x, y) in face:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    return frame


def draw_rects(frame, rects):
    for rect in rects:
        frame = cv2.rectangle(frame, (rect.left(), rect.top()),
                              (rect.right(), rect.bottom()), (0, 0, 255), 2)

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

    def update_img(self, image, faces, rects):
        self.image = cirle_features(image, faces)
        self.image = draw_rects(image, rects)
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

        self.glasses = cv2.imread(
            constants.STICKERS_GLASSES_PATH, cv2.IMREAD_UNCHANGED)
        self.mustache = cv2.imread(
            constants.STICKERS_MUSTACHE_PATH, cv2.IMREAD_UNCHANGED)
        self.hat = cv2.imread(constants.STICKERS_HAT_PATH,
                              cv2.IMREAD_UNCHANGED)

        assert self.glasses.shape[2] == 4
        assert self.mustache.shape[2] == 4
        assert self.hat.shape[2] == 4

    def update_img(self, image, gray_image, faces, draw_hat, draw_glasses, draw_mustache):
        image = put_sticker(
            image, faces, self.glasses, self.mustache, self.hat, draw_hat, draw_glasses, draw_mustache)

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

        self.hat.setChecked(True)
        self.glasses.setChecked(True)
        self.mustache.setChecked(True)

    def show(self):
        self.parent.show()

    def image_data_slot(self, frame):
        time_start = time.time()

        # frame -> resize -> gray
        frame = cv2.resize(frame, (500, int(
            500/frame.shape[1]*frame.shape[0])), interpolation=cv2.INTER_AREA)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detection
        faces, rects = self.detector.extract_faces(gray_frame)

        # update widgets
        self.widgetFrame.update_img(frame.copy(), faces, rects)
        self.widget2D.update_img(
            frame.copy(), gray_frame.copy(), faces, self.hat.isChecked(), self.glasses.isChecked(), self.mustache.isChecked())

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
args.add_argument('algorithm', choices=['hog', 'violajones', 'dlib'])
args = args.parse_args()

# choose detector
detector = None
if args.algorithm == 'hog':
    detector = hog_detector.HogDetector
elif args.algorithm == 'violajones':
    detector = violajones_detector.ViolaJonesDetector
else:
    detector = dlib_detector.DlibDetector

detector = detector(constants.DLIB_MODEL_PATH)

# define app
app = QtWidgets.QApplication(sys.argv)

ui = Window(args, detector)

# run app
ui.show()
exit_code = app.exec_()
print()  # to keep the fps line
sys.exit(exit_code)
