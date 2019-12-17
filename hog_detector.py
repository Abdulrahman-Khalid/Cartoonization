import dlib
import numpy as np
import skimage
from imutils import face_utils
from skimage import feature
import joblib
import cv2

import constants


def _non_max_suppression_slow(bboxes, thres):
    if len(bboxes) == 0:
        return []

    pick = []

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > thres:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return bboxes[pick]


def _apply_window(img, patch_width, patch_height, step_i, step_j):
    for i in range(0, img.shape[0] - patch_width, step_i):
        for j in range(0, img.shape[1] - patch_width, step_j):
            patch = img[i:i + patch_width, j:j + patch_height]

            yield (i, j), patch


def _detect_with_hog(model, gray_frame, step_i, step_j):
    patch_width, patch_height = 62, 47

    indices, patches = zip(
        *_apply_window(gray_frame, patch_width, patch_height, step_i, step_j))
    patches_hog = np.array([feature.hog(patch) for patch in patches])

    labels = model.predict(patches_hog)
    indices = np.array(indices)

    bboxes = np.array([(i, j, i+patch_width, j+patch_height)
                       for i, j in indices[labels == 1]])
    return _non_max_suppression_slow(bboxes, 0.3)


def _bbox_to_dlib_rectangle(bbox: ((int, int), (int, int)), scl) -> dlib.rectangle:
    left, top, right, bottom = bbox
    return dlib.rectangle(int(left//scl), int(top//scl), int(right//scl), int(bottom//scl))


class HogDetector:
    def __init__(self, model_path: str):
        self.hog_model = joblib.load(constants.HOG_MODEL_PATH)

        # load the facial landmark predictor from disk
        self.dlib_segmentation = dlib.shape_predictor(model_path)

    def detect(self, frame, scl, step_i, step_j):
        ''' Given grayscale-frame return [bounding-box], where bounding-box is ((x0, y0), (x1, y1)) '''
        frame = skimage.transform.rescale(frame.copy(), scl)
        return _detect_with_hog(self.hog_model, frame, step_i, step_j)

    def extract_faces(self, frame: np.ndarray) -> [[(int, int)]]:
        ''' Given gray scale image (2D np array), return array of faces in it '''
        scale = .29
        step_vertical = 20
        step_horizontal = 5

        rects = [_bbox_to_dlib_rectangle(rect, scale)
                 for rect in self.detect(frame, scale, step_vertical, step_horizontal)]
        return [face_utils.shape_to_np(self.dlib_segmentation(frame, rect)) for rect in rects], rects
