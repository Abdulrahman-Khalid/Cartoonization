import dlib
import numpy as np
from imutils import face_utils


class DlibDetector:
    def __init__(self, model_path: str):
        # initialize dlib’s HOG Linear SVM-based face detector
        self.detector = dlib.get_frontal_face_detector()

        # load the facial landmark predictor from disk
        self.predictor = dlib.shape_predictor(model_path)

    def extract_faces(self, frame: np.ndarray) -> [[(int, int)]]:
        ''' Given gray scale image (2D np array), return array of faces in it '''
        return [face_utils.shape_to_np(self.predictor(frame, face)) for face in self.detector(frame, 0)]
