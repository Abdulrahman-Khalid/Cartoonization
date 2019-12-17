import dlib
import numpy as np
import skimage
from imutils import face_utils
from skimage import feature
import joblib

import constants


def _non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]


def _sliding_window(img, patch_size=(62, 47),
                    istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = skimage.transform.resize(patch, patch_size)
            yield (i, j), patch


def _detect_with_hog(model, gray_frame):
    indices, patches = zip(*_sliding_window(gray_frame))
    patches_hog = np.array([feature.hog(patch) for patch in patches])

    labels = model.predict(patches_hog)
    indices = np.array(indices)

    Ni, Nj = (62, 47)

    bboxes = np.array([(i, j, i+Ni, j+Nj) for i, j in indices[labels == 1]])
    return _non_max_suppression_slow(bboxes, 0.3)


def _bbox_to_dlib_rectangle(bbox: ((int, int), (int, int))) -> dlib.rectangle:
    (left, top), (right, bottom) = bbox
    return dlib.rectangle(left, top, right, bottom)


class HogDetector:
    def __init__(self, model_path: str):
        self.hog_model = joblib.load(constants.HOG_MODEL_PATH)

        # load the facial landmark predictor from disk
        self.dlib_segmentation = dlib.shape_predictor(model_path)

    def detect(self, frame):
        ''' Given grayscale-frame return [bounding-box], where bounding-box is ((x0, y0), (x1, y1)) '''
        return _detect_with_hog(self.hog_model, frame)

    def extract_faces(self, frame: np.ndarray) -> [[(int, int)]]:
        ''' Given gray scale image (2D np array), return array of faces in it '''
        return [face_utils.shape_to_np(self.dlib_segmentation(frame, _bbox_to_dlib_rectangle(rect))) for rect in self.detect(frame)]
