import dlib
import numpy as np
from imutils import face_utils

import violajones.Utils as utils
import violajones.IntegralImage as IImg
import constants as c


class ViolaJonesDetector:
    def __init__(self, model_path: str):
        print('Inintailizing ViolaJones custom detector...')

        print("Loading Classifiers...")
        classifiers = utils.load_classifiers(c.classifiersFileName)
        print(len(classifiers), "Classifiers are loaded")
        self.classifiers_stages = utils.classifiers_to_classifiers_stages(
            classifiers)

        # initialize dlib’s HOG Linear SVM-based face detector
        self._dlib_detector_todo = dlib.get_frontal_face_detector()

        # load the facial landmark predictor from disk
        self.dlib_segmentation = dlib.shape_predictor(model_path)

        print('Done Inintailizing ViolaJones custom detector')

    def detect(self, frame):
        ''' Given grayscale-frame return [bounding-box], where bounding-box is ((x0, y0), (x1, y1)) '''
        width, height = frame.shape[:2]
        iimage = IImg.get_integral_image(frame)

        return utils.detect_faces(iimage, width,
                                  height, self.classifiers_stages)

    def extract_faces(self, frame: np.ndarray) -> [[(int, int)]]:
        ''' Given gray scale image (2D np array), return array of faces in it '''
        return [face_utils.shape_to_np(self.dlib_segmentation(frame, bbox)) for bbox in self.detect(frame)]
