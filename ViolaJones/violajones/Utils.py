import os
import pickle
from functools import partial
from violajones.HaarLikeFeature import FeatureType
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_images(path):
    images = []
    for img in tqdm(os.listdir(path), unit="image"):
        if img.endswith('.jpg') or img.endswith('.png'):
            imgArr = np.array(Image.open(
                (os.path.join(path, img))), dtype=np.float64)
            imgArr /= imgArr.max()
            images.append(imgArr)
    return images


def save_classifiers(classifiers, classifiersFileName):
    with open(classifiersFileName, 'wb') as outputFile:
        pickle.dump(classifiers, outputFile, pickle.HIGHEST_PROTOCOL)


def load_classifiers(classifiersFileName):
    with open(classifiersFileName, 'rb') as inputFile:
        return pickle.load(inputFile)


def isFaceImg(integralImg, classifiers):
    return 1 if sum([classifier.get_vote(integralImg) for classifier in classifiers]) >= 0 else 0


def isFaceImgs(integralImgs, classifiers):
    return list(map(partial(isFaceImg, classifiers=classifiers), integralImgs))


def casCadingIsFaceImg(integralImg, classifiers):
    for classifier in classifiers:
        if classifier.get_vote(integralImg) < 0:
            return 0
    return 1


def casCadingIsFaceImgs(integralImgs, classifiers):
    return list(map(partial(casCadingIsFaceImg, classifiers=classifiers), integralImgs))
