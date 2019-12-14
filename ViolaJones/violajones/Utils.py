import os
import pickle
from functools import partial
from violajones.HaarLikeFeature import HaarLikeFeature
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


def compute_feature(self, box, featureChosen, integralImg):
    # scaling features
    boxSize = box[0]  # area
    # @ TODO the calFeatures file
    # featureChosen = features[featureIndex] # features should be from the calFeatures file

    areaPos_i = box[1]
    areaPos_j = box[2]
    sampleSize = 24
    scale = np.sqrt(boxSize) / sampleSize

    # scaling the i and j of the feature
    i = round(scale*featureChosen.topleft[1])  # topleft[1] rows
    j = round(scale*featureChosen.topleft[0])  # topleft[0] cols

    # abs_i and abs_j will be used to calculate the integral image result
    # indicate the feature position inside the real frame
    abs_i = areaPos_i + i
    abs_j = areaPos_j + j

    # getting the haar feature width and height
    # we will check on the feature pattern to get the width
    isHoriztonal = featureChosen.featureType == FeatureType.THREE_HORIZONTAL or featureChosen.featureType == FeatureType.TWO_HORIZONTAL
    isFour = featureChosen.featureType == FeatureType.FOUR
    width = featureChosen.width*3 if isHoriztonal else featureChosen.width
    width += featureChosen.width if isFour else 0  # as feature five width is at 5,6
    # we will check on the feature pattern to get the height
    height = featureChosen.height if isHoriztonal else featureChosen.height + \
        featureChosen.width
    # feature five height is at 3,4 while feature three and four their heights is at 3,4,5
    # vertical features
    height += featureChosen.height if not(isFour or isHoriztonal) else 0

    # original area of the feature
    originArea = width*height

    # scaling the width and the height of the feature
    width = round(scale*width)
    height = round(scale*height)

    # scaling the feature pattern one i.e. 1x2 feature
    if(featureChosen.featureType == FeatureType.TWO_HORIZONTAL):
        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        height = height if height < (
            np.sqrt(boxSize) - i) else (np.sqrt(boxSize) - i)

        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        # we should make sure that width is divisible by 2 after scaling
        width = width if width % 2 == 0 else width + 1

        while (width > np.sqrt(boxSize) - j):
            width -= 2

    # scaling the feature pattern two i.e. 1x3 feature
    elif(featureChosen.featureType == FeatureType.THREE_HORIZONTAL):
        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        height = height if height < (
            np.sqrt(boxSize) - i) else (np.sqrt(boxSize) - i)
        # we should make sure that width is divisible by 3 after scaling
        width = width if width % 3 == 0 else (
            (width + 2 if width % 3 == 1 else width + 1))

        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        while (width > np.sqrt(boxSize) - j):
            width -= 3

    # scaling the feature pattern one i.e. 2x1 feature
    elif(featureChosen.featureType == FeatureType.TWO_VERTICAL):

        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        width = width if width < (
            np.sqrt(boxSize) - j) else (np.sqrt(boxSize) - j)

        '''p
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        # we should make sure that height is divisible by 2 after scaling
        height = height if height % 2 == 0 else height + 1

        while (height > np.sqrt(boxSize) - i):
            height -= 2

    # scaling the feature pattern one i.e. 3x1 feature
    elif(featureChosen.featureType == FeatureType.THREE_VERTICAL):

        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        width = width if (width < (np.sqrt(boxSize) - j)
                          ) else (np.sqrt(boxSize) - j)

        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        # we should make sure that height is divisible by 3 after scaling
        height = height if height % 3 == 0 else (
            (height + 2 if height % 3 == 1 else height + 1))

        while (height > np.sqrt(boxSize) - i):
            height -= 3

    # scaling the feature pattern one i.e. 2x2 feature
    else:

        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        # we should make sure that width is divisible by 2 after scaling
        width = width if width % 2 == 0 else width + 1

        while (width > np.sqrt(boxSize) - j):
            width -= 2

        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        # we should make sure that height is divisible by 2 after scaling
        height = height if height % 2 == 0 else height + 1

        while (height > np.sqrt(boxSize) - i):
            height -= 2

    new_feature = HaarLikeFeature(featureChosen.featureType, (abs_j, abs_i),
                                  width, height, featureChosen.threshold, featureChosen.polarity)
    score = new_feature.get_score(integralImg)
    # rescale the feature to its original scale
    # multiply the originArea by 2
    reScale = originArea/(width*height)

    featureResult = score * reScale
    return featureResult
