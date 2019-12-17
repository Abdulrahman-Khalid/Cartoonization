import dlib
import numpy as np
from imutils import face_utils

import violajones.Utils as utils
from hog_detector import _non_max_suppression_slow
import constants as c


def _bbox_to_dlib_rectangle(bbox: ((int, int), (int, int))) -> dlib.rectangle:
    left, top, right, bottom = bbox
    return dlib.rectangle(left, top, right, bottom)


class ViolaJonesDetector:
    def __init__(self, model_path: str):
        print('Inintailizing ViolaJones custom detector...')

        print("Loading Classifiers...")
        classifiers = utils.load_classifiers(c.classifiersFileName)
        print(len(classifiers), "Classifiers are loaded")
        self.classifiers_stages = utils.classifiers_to_classifiers_stages(
            classifiers)

        # load the facial landmark predictor from disk
        self.dlib_segmentation = dlib.shape_predictor(model_path)

        print('Done Inintailizing ViolaJones custom detector')

    def detect(self, frame):
        ''' Given grayscale-frame return [bounding-box], where bounding-box is ((x0, y0), (x1, y1)) '''
        height, width = frame.shape[:2]
        return utils.detect_faces(frame, width, height, self.classifiers_stages)

    def extract_faces(self, frame: np.ndarray) -> [[(int, int)]]:
        ''' Given gray scale image (2D np array), return array of faces in it '''
        rects = _non_max_suppression_slow(self.detect(frame), .3)
        rects = [_bbox_to_dlib_rectangle(bbox) for bbox in rects]
        return [face_utils.shape_to_np(self.dlib_segmentation(frame, rect)) for rect in rects], rects
