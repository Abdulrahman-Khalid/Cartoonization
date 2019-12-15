import dlib
import numpy as np
from imutils import face_utils


class CustomDetector:
    def __init__(self, model_path: str):
        # initialize dlib’s HOG Linear SVM-based face detector
        self._dlib_detector_todo = dlib.get_frontal_face_detector()

        # load the facial landmark predictor from disk
        self.dlib_segmentation = dlib.shape_predictor(model_path)

    def detect(self, frame):
        # TODO
        return self._dlib_detector_todo(frame, 0)

    def extract_faces(self, frame: np.ndarray) -> [[(int, int)]]:
        ''' Given gray scale image (2D np array), return array of faces in it '''
        return [face_utils.shape_to_np(self.dlib_segmentation(frame, bbox)) for bbox in self.detect(frame)]
