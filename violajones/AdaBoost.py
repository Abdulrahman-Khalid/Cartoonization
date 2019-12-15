from multiprocessing import Pool
from functools import partial

from tqdm import tqdm
import numpy as np

from violajones.Utils import save_classifiers
from violajones.HaarLikeFeature import HaarLikeFeature
from violajones.HaarLikeFeature import FeatureTypes
from constants import classifiersFileName


def adaBoostLearn(facesIntegrals, nonFacesIntegrals, minFeatureWidth=1, maxFeatureWidth=-1, minFeatureHeight=1, maxFeatureHeight=-1, classifiersNum=-1):
    """
    TODO List
    attentional cascading
    select optimal threshold for each feature
    """
    facesIntegralsLength = len(facesIntegrals)
    nonFacesIntegralsLength = len(nonFacesIntegrals)
    imgsTotalCount = facesIntegralsLength + nonFacesIntegralsLength
    imgHeight, imgWidth = facesIntegrals[0].shape

    # Maximum feature width and height default to image width and height
    maxFeatureHeight = imgHeight if maxFeatureHeight == -1 else maxFeatureHeight
    maxFeatureWidth = imgWidth if maxFeatureWidth == -1 else maxFeatureWidth

    # Create initial weights and labels
    positiveWeights = np.ones(facesIntegralsLength) * 1. / \
        (2 * facesIntegralsLength)
    negativeWeights = np.ones(nonFacesIntegralsLength) * \
        1. / (2 * nonFacesIntegralsLength)
    weights = np.hstack((positiveWeights, negativeWeights))
    labels = np.hstack((np.ones(facesIntegralsLength),
                        np.ones(nonFacesIntegralsLength) * -1))

    images = facesIntegrals + nonFacesIntegrals

    # Create features for all sizes and locations
    features = create_features(imgHeight, imgWidth, minFeatureWidth,
                               maxFeatureWidth, minFeatureHeight, maxFeatureHeight)
    featuresCount = len(features)
    featureIndecies = list(range(featuresCount))

    classifiersNum = featuresCount if classifiersNum == -1 else classifiersNum

    print('Calculating scores for images...')

    votes = np.zeros((imgsTotalCount, featuresCount))
    with Pool(processes=None) as p:
        votesFaces = list(tqdm(
            p.imap(partial(get_feature_vote, features=features), facesIntegrals), total=len(facesIntegrals), unit="image"))
        votesNonFaces = list(tqdm(
            p.imap(partial(get_feature_vote, features=features), nonFacesIntegrals), total=len(nonFacesIntegrals), unit="image"))

    votes = np.array(votesFaces+votesNonFaces)
    print('Haar like features are Extracted successfully!\n')

    # select best classifiers
    classifiers = []
    print('Selecting classifiers...')
    for _ in tqdm(range(classifiersNum), unit="classifier"):
        classificationErrors = np.zeros(len(featureIndecies))
        # normalize weights
        weights *= 1. / np.sum(weights)
        # select best classifier based on the weighted error
        for index in range(len(featureIndecies)):
            # classifier error is the sum of image weights where the classifier is right
            error = sum(map(
                lambda imgIndex: weights[imgIndex] if labels[imgIndex] != votes[imgIndex, featureIndecies[index]] else 0, range(imgsTotalCount)))
            classificationErrors[index] = error

        # get best feature with smallest error
        minErrorIndex = np.argmin(classificationErrors)
        bestError = classificationErrors[minErrorIndex]
        bestFeatureIndex = featureIndecies[minErrorIndex]

        # set feature weight
        bestFeature = features[bestFeatureIndex]
        # amount of say
        bestFeature.weight = 0.5 * \
            np.log((1 - bestError) / bestError)
        classifiers.append(bestFeature)
        save_classifiers(classifiers, classifiersFileName)
        # update image weights
        weights = np.array(list(map(lambda imgIndex: weights[imgIndex] * np.sqrt((1-bestError)/bestError) if labels[imgIndex]
                                    != votes[imgIndex, bestFeatureIndex] else weights[imgIndex] * np.sqrt(bestError/(1-bestError)), range(imgsTotalCount))))

        # remove feature (a feature can't be selected twice)
        featureIndecies.remove(bestFeatureIndex)

    return classifiers


def create_features(imgHeight, imgWidth, minFeatureWidth, maxFeatureWidth, minFeatureHeight, maxFeatureHeight):
    print('Creating haar like features...')
    features = []
    for feature in FeatureTypes:
        # FeatureTypes are just tuples
        featureStartWidth = max(minFeatureWidth, feature[0])
        for featureWidth in range(featureStartWidth, maxFeatureWidth, feature[0]):
            featureStartHeight = max(minFeatureHeight, feature[1])
            for featureHeight in range(featureStartHeight, maxFeatureHeight, feature[1]):
                for x in range(imgWidth - featureWidth):
                    for y in range(imgHeight - featureHeight):
                        features.append(HaarLikeFeature(
                            feature, (x, y), featureWidth, featureHeight, 0.11, -1))
                        # features.append(HaarLikeFeature(
                        #     feature, (x, y), featureWidth, featureHeight, 0, 1))
                        # features.append(HaarLikeFeature(
                        #     feature, (x, y), featureWidth, featureHeight, 0, -1))
    print(str(len(features)) + ' features are created!')
    return features


def get_feature_vote(image, features):
    votes = []
    for feature in features:
        votes.append(feature.get_vote(image))
    return votes
