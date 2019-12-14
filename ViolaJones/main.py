import sys
import os
import violajones.AdaBoost as AdaBoost
import violajones.Utils as utils
import violajones.IntegralImage as IImg
import constants as c


def testClassifiers(classifiers, classifiers_stages, facesTestingPath, nonFacesTestingPath):
    print('Loading faces testing data set...')
    facesTesting = utils.load_images(c.facesTestingPath)
    facesTestingIntegral = list(
        map(IImg.get_integral_image, facesTesting))
    print(str(len(facesTesting)) +
          ' Testing faces data set are loaded!\n')
    print('Loading testing non faces data set...')
    nonFacesTesting = utils.load_images(c.nonFacesTestingPath)
    nonFacesTestingIntegral = list(
        map(IImg.get_integral_image, nonFacesTesting))
    print(str(len(nonFacesTesting)) +
          ' Testing non faces data set are loaded!\n')

    print('Testing selected classifiers ...')
    correctFacesCount = 0
    correctNonFacesCount = 0

    # comment this part
    correctFacesCount = sum(list(len(utils.detect_faces(
        face, 24, 24, classifiers_stages)) for face in facesTestingIntegral))
    correctNonFacesCount = len(nonFacesTesting) - sum(list(len(utils.detect_faces(
        nonFace, 24, 24, classifiers_stages)) for nonFace in nonFacesTestingIntegral))

    # uncomment this part
    # correctFacesCount = sum(utils.isFaceImgs(
    #     facesTestingIntegral, classifiers))
    # correctNonFacesCount = len(
    #     nonFacesTesting) - sum(utils.isFaceImgs(nonFacesTestingIntegral, classifiers))

    print('Accuracy:-\nFaces: ' + str(correctFacesCount) + '/' + str(len(facesTesting))
          + '  (' + str((float(correctFacesCount) / len(facesTesting))
                        * 100) + '%)\nNon Faces: '
          + str(correctNonFacesCount) + '/' +
          str(len(nonFacesTesting)) + '  ('
          + str((float(correctNonFacesCount) / len(nonFacesTesting)) * 100) + '%)')


def main():
    if not os.path.exists(c.classifiersFileName) or (len(sys.argv) >= 2 and sys.argv[len(sys.argv)-1] == 'relearn'):
        print('Loading faces training data set...')
        facesTraining = utils.load_images(c.facesTrainingPath)
        facesTrainingIntegral = list(
            map(IImg.get_integral_image, facesTraining))
        print(str(len(facesTraining)) +
              ' Training faces data set are loaded!\n')
        print('Loading non faces training data set...')
        nonFacesTraining = utils.load_images(c.nonFacesTrainingPath)
        nonFacesTrainingIntegral = list(
            map(IImg.get_integral_image, nonFacesTraining))
        print(str(len(nonFacesTraining)) +
              ' Training non faces data set are loaded!\n')

        classifiers = AdaBoost.adaBoostLearn(facesTrainingIntegral, nonFacesTrainingIntegral,
                                             c.minFeatureHeight, c.maxFeatureHeight, c.minFeatureWidth, c.maxFeatureWidth, c.classifiersNum)

        utils.save_classifiers(classifiers, c.classifiersFileName)
        classifiers_stages = utils.classifiers_to_classifiers_stages(
            classifiers)
        testClassifiers(classifiers, classifiers_stages,
                        c.facesTestingPath, c.nonFacesTestingPath)

    elif os.path.exists(c.classifiersFileName) and (len(sys.argv) >= 2 and sys.argv[len(sys.argv)-1] == 'test'):
        print("Loading Classifiers ...")
        classifiers = utils.load_classifiers(c.classifiersFileName)
        print(len(classifiers), "Classifier are loaded")
        classifiers_stages = utils.classifiers_to_classifiers_stages(
            classifiers)
        testClassifiers(classifiers, classifiers_stages,
                        c.facesTestingPath, c.nonFacesTestingPath)
        # f = open(c.classifiersFileASNumbers, "w")
        # for cl in classifiers:
        #     f.write(str(cl.featureType) + " " + str(cl.topLeft) + " " + str(cl.width) +
        #             " " + str(cl.height) + " " + str(cl.threshold) + " " + str(cl.polarity) + "\n")
    elif os.path.exists(c.classifiersFileName):
        print("Loading Classifiers ...")
        classifiers = utils.load_classifiers(c.classifiersFileName)
        print(len(classifiers), "Classifier are loaded")
        classifiers_stages = utils.classifiers_to_classifiers_stages(
            classifiers)
        # TODO Run Camera
        while(True):
            # TODO get frame
            # TODO get frameWidth and frameHeight
            # TODO get frameGrayScale
            iimage = utils.get_integral_image(frameGrayScale)
            rects = utils.detect_faces(iimage, frameWidth,
                                       frameHeight, classifiers_stages)


main()
