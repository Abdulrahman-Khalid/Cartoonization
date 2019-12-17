import base64
import os
import pickle
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import constants as c
import violajones.IntegralImage as IImg
from violajones.HaarLikeFeature import FeatureType, HaarLikeFeature


def load_images(path):
    images = []
    for img in tqdm(os.listdir(path), unit="image"):
        if img.endswith('.jpg') or img.endswith('.png'):
            imgArr = np.array(Image.open(
                (os.path.join(path, img))), dtype=np.float64)
            imgArr /= imgArr.max()
            images.append(imgArr)
    return images


def classifiers_to_classifiers_stages(classifiers):
    stages = [13, 21, 41, 51, 61, 71, 78, 51, 101, 111, 121, 131,
              141, 151, 161, 171, 181, 191, 201, 201, 201, 111,
              201, 201, 201, 201, 201, 201, 201, 201, 201]
    length_to_split = []
    stop = 0
    for s in stages:
        if(stop + s > len(classifiers)):
            length_to_split.append(len(classifiers)-s)
            break
        else:
            length_to_split.append(s)

    # length_to_split = [13] * int(len(classifiers)/13)
    # if(len(classifiers) % 13 != 0):
    #     length_to_split.append(len(classifiers) % 13)
    Inputt = iter(classifiers)
    return [list(islice(Inputt, num)) for num in length_to_split]


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


def cascadingIsFace(box, integralImg, classifiers_stages):
    # return True if sum([classifier.get_vote(integralImg) for classifier in classifiers_stages]) >= 0 else False
    s = 0
    for stage in classifiers_stages:
        # s += sum([classifier.get_vote_cascade(compute_feature(box, classifier, integralImg))
        #           for classifier in classifiers_stages[i]])
        for classifier in stage:
            featureResult, new_feature = compute_feature(
                box, classifier, integralImg)
            s += new_feature.get_vote_cascade(featureResult)
        # cascading
        if(s < 0):
            return False  # not face don't continue
    return True if s >= 0 else False


def detect_faces(frame, frameWidth, frameHeight, classifiers_stages, bbox_to_dlib_rectangle):
    iimage = IImg.get_integral_image(frame)

    rects = []
    minFrame = frameWidth if frameWidth < frameHeight else frameHeight
    for b in range(c.minSize, minFrame+1, c.sizeStep):
        for w in range(0, frameWidth-b+1, c.stepSizeW):
            for h in range(0, frameHeight-b+1, c.stepSizeH):
                if(cascadingIsFace([b**2, h, w], iimage, classifiers_stages)):
                    rects.append(bbox_to_dlib_rectangle(((h, w), (h+b, w+b))))

    return rects


def compute_feature(box, featureChosen, integralImg):
    # scaling features
    boxSize = box[0]  # area
    # @ TODO the calFeatures file
    # featureChosen = features[featureIndex] # features should be from the calFeatures file

    areaPos_i = box[1]
    areaPos_j = box[2]
    sampleSize = 24
    scale = np.sqrt(boxSize) / sampleSize

    # scaling the i and j of the feature
    i = int(scale*featureChosen.topLeft[1] + 0.5)  # topLeft[1] rows
    j = int(scale*featureChosen.topLeft[0] + 0.5)  # topLeft[0] cols

    # abs_i and abs_j will be used to calculate the integral image result
    # indicate the feature position inside the real frame
    abs_i = areaPos_i + i
    abs_j = areaPos_j + j

    # getting the haar feature width and height
    # we will check on the feature pattern to get the width
    width = featureChosen.width
    # if(featureChosen.featureType == FeatureType.THREE_HORIZONTAL):
    #     width *= 3
    # elif (featureChosen.featureType == FeatureType.TWO_HORIZONTAL or featureChosen.featureType == FeatureType.FOUR):
    #     width *= 2
    height = featureChosen.height
    # if(featureChosen.featureType == FeatureType.THREE_VERTICAL):
    #     height *= 3
    # elif (featureChosen.featureType == FeatureType.TWO_VERTICAL or featureChosen.featureType == FeatureType.FOUR):
    #     height *= 2
    # original area of the feature
    originArea = width*height

    # scaling the width and the height of the feature
    width = int(scale*width + 0.5)
    height = int(scale*height + 0.5)

    # scaling the feature pattern one i.e. 1x2 feature
    if(featureChosen.featureType == FeatureType.TWO_HORIZONTAL):
        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''

        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        # we should make sure that width is divisible by 2 after scaling

        while (width > np.sqrt(boxSize) - j):
            width -= 2

    # scaling the feature pattern two i.e. 1x3 feature
    elif(featureChosen.featureType == FeatureType.THREE_HORIZONTAL):
        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        while (width > np.sqrt(boxSize) - j):
            width -= 3

    # scaling the feature pattern one i.e. 2x1 feature
    elif(featureChosen.featureType == FeatureType.TWO_VERTICAL):

        '''p
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        # we should make sure that height is divisible by 2 after scaling
        while (height > np.sqrt(boxSize) - i):
            height -= 2

    # scaling the feature pattern one i.e. 3x1 feature
    elif(featureChosen.featureType == FeatureType.THREE_VERTICAL):
        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        # we should make sure that height is divisible by 3 after scaling
        while (height > np.sqrt(boxSize) - i):
            height -= 3

    # scaling the feature pattern one i.e. 2x2 feature
    else:

        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        # we should make sure that width is divisible by 2 after scaling
        while (width > np.sqrt(boxSize) - j):
            width -= 2

        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        # we should make sure that height is divisible by 2 after scaling
        while (height > np.sqrt(boxSize) - i):
            height -= 2

    new_feature = HaarLikeFeature(featureChosen.featureType, (abs_j, abs_i),
                                  width, height, featureChosen.threshold, featureChosen.polarity)
    # print(new_feature)
    score = new_feature.get_score(integralImg)
    # rescale the feature to its original scale
    # multiply the originArea by 2
    reScale = originArea/(width*height)

    featureResult = score * reScale
    return featureResult, new_feature
