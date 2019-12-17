import os
import pickle
from itertools import islice
from functools import partial
from violajones.HaarLikeFeature import HaarLikeFeature
# from violajones.HaarLikeFeature import FeatureType
from violajones.Feature import FeatureType, Feature
import violajones.IntegralImage as IImg
import numpy as np
from PIL import Image
from tqdm import tqdm
import constants as c
import copy
import random
from datetime import datetime


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


def cascadingIsFace(box, frameGrayScale, integralImg, classifiers_stages, normalizedVariance):
    # return True if sum([classifier.get_vote(integralImg) for classifier in classifiers_stages]) >= 0 else False
    prediction = 0
    for i, stage in enumerate(classifiers_stages):
        # For each feature in that layer:
        for j, feature in enumerate(stage):
            feature_value = compute_feature(
                box, feature, frameGrayScale, integralImg)/normalizedVariance
            vote = 1 if feature_value > feature.threshold else -1
            vote *= feature.polarity
            vote += c.tweaks[i]
            if(feature.weight == 0):
                if(j == 0):
                    return True if vote > 0 else False
            prediction += vote * np.log((1/feature.weight) - 1)
        if (prediction < 0):
            # if not face return false
            return False
    return True


def normalize_image(image):
    max_px = np.max(image)
    min_px = np.min(image)
    std = max_px - min_px
    mean = image.mean()
    image = (image - mean) / std
    return image


def detect_faces(frameGrayScale, iimage, frameWidth, frameHeight, classifiers_stages):
    rects = []
    minFrame = frameWidth if frameWidth < frameHeight else frameHeight
    for b in range(c.minSize, minFrame+1, c.sizeStep):
        for w in range(0, frameWidth-b+1, c.stepSizeW):
            for h in range(0, frameHeight-b+1, c.stepSizeH):
                summation1 = int(
                    np.sum(np.power(frameGrayScale[h:h+b+1, w:w+b+1], 2)))
                summation2 = int(np.sum(frameGrayScale[h:h+b+1, w:w+b+1]))
                area = b**2
                num = ((summation2/area)**2) - (summation1/area)
                std = 0
                if(num > 0):
                    std = np.sqrt(num)
                if(std < 1):
                    continue
                if(cascadingIsFace([area, h, w], frameGrayScale, iimage, classifiers_stages, std/1e4)):
                    rects.append(((h, w), (h+b, w+b)))
    return rects


def compute_feature(box, featureChosen, frameGrayScale, integralImg):
    # scaling features
    boxSize = box[0]  # area
    sideLength = int(np.sqrt(boxSize))
    areaPos_i = box[1]
    areaPos_j = box[2]
    # @ TODO the calFeatures file
    # featureChosen = features[featureIndex] # features should be from the calFeatures file
    sampleSize = 24
    scale = sideLength / sampleSize

    # scaling the i and j of the feature
    i = int(scale*featureChosen.topLeft[1] + 0.5)  # topLeft[1] rows
    j = int(scale*featureChosen.topLeft[0] + 0.5)  # topLeft[0] cols

    # abs_i and abs_j will be used to calculate the integral image result
    # indicate the feature position inside the real frame
    abs_i = areaPos_i + i
    abs_j = areaPos_j + j

    width = featureChosen.width
    height = featureChosen.height
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
        height = height if height < (
            sideLength - i) else (sideLength - i)
        width = width if width % 2 == 0 else (width + 1)
        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        # we should make sure that width is divisible by 2 after scaling

        while (width > 2 and width > sideLength - j):
            width -= 2

    # scaling the feature pattern two i.e. 1x3 feature
    elif(featureChosen.featureType == FeatureType.THREE_HORIZONTAL):
        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        height = height if height < (
            sideLength - i) else (sideLength - i)
        # we should make sure that width is divisible by 3 after scaling
        width = width if width % 3 == 0 else (
            (width + 2 if width % 3 == 1 else width + 1))
        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        while (width > 3 and width > sideLength - j):
            width -= 3

    # scaling the feature pattern one i.e. 2x1 feature
    elif(featureChosen.featureType == FeatureType.TWO_VERTICAL):
        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        width = width if width < (
            sideLength - j) else (sideLength - j)
        '''p
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        # we should make sure that height is divisible by 2 after scaling
        height = height if height % 2 == 0 else height + 1
        # we should make sure that height is divisible by 2 after scaling
        while (height > 2 and height > sideLength - i):
            height -= 2

    # scaling the feature pattern one i.e. 3x1 feature
    elif(featureChosen.featureType == FeatureType.THREE_VERTICAL):
        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        width = width if (width < (sideLength - j)
                          ) else (sideLength - j)

        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        # we should make sure that height is divisible by 3 after scaling
        height = height if height % 3 == 0 else (
            (height + 2 if height % 3 == 1 else height + 1))

        while (height > 3 and height > sideLength - i):
            height -= 3

    # scaling the feature pattern one i.e. 2x2 feature
    else:

        '''
        the width of the feature may exceeds the box's size - j as
        boxSize - j is the maximum side the feature's width can hold
        '''
        width = width if width % 2 == 0 else width + 1
        # we should make sure that width is divisible by 2 after scaling
        while (width > 2 and width > sideLength - j):
            width -= 2

        '''
        the height of the feature may exceeds the box's size - i as
        boxSize - i is the maximum side the feature's height can hold
        '''
        height = height if height % 2 == 0 else height + 1
        # we should make sure that height is divisible by 2 after scaling
        while (height > 2 and height > sideLength - i):
            height -= 2

    new_feature = Feature(featureChosen.featureType, (abs_j, abs_i),
                          int(width), int(height), featureChosen.weight, featureChosen.threshold, featureChosen.polarity)
    # print(new_feature)
    score = new_feature.get_score(integralImg)
    # rescale the feature to its original scale
    # multiply the originArea by 2
    if(featureChosen.featureType == FeatureType.THREE_HORIZONTAL or featureChosen.featureType == FeatureType.THREE_VERTICAL):
        mean = IImg.sum_region(integralImg, (areaPos_j, areaPos_i),
                               (areaPos_j+sideLength, areaPos_i+sideLength))/boxSize
        score -= mean*width*height/3
    reScale = originArea/(width*height)
    featureResult = score * reScale
    return featureResult


def doubleCheckIsFace(box, integralImg, classifiers_stages):
    s = 0
    for stage in classifiers_stages:
        for classifier in stage:
            featureResult, new_feature = compute_feature(
                box, classifier, integralImg)
            s += new_feature.get_vote_cascade(featureResult)
    return True if s >= 0 else False


def pickBest(rects):
    bests = []
    bests = copy.deepcopy(rects)

    size = len(rects)
    # i = 0
    # while i < size:
    now = datetime.now()
    current = int(now.strftime("%S"))
    randn = current % len(rects)
    print(randn)
    thsh = np.sqrt(2) * abs(rects[randn][0][0] - rects[randn][1][0])
    thsh = 2.5*thsh
    j = 0
    while j < size:
        if j != randn:
            X_diff = (rects[randn][0][0] - rects[j][0][0])**2
            Y_diff = (rects[randn][0][1] - rects[j][0][1])**2
            dist = np.sqrt(X_diff + Y_diff)
            if dist < thsh:
                bests.remove(rects[j])
        j += 1

    return (bests)
