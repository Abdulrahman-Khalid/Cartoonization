import numpy as np

class CustomDetector:
    def __init__(self, model_path: str):
        # TODO: read model (trained data)
        raise NotImplementedError

    def extract_faces(self, frame: np.ndarray) -> [[(int, int)]]:
        ''' Given gray scale image (2D np array), return array of faces in it '''
        # TODO
        raise NotImplementedError