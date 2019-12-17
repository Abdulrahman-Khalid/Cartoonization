import sys
import os
import violajones.AdaBoost as AdaBoost
import violajones.Utils as utils
import violajones.IntegralImage as IImg
import constants as c
import cv2
import numpy as np


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
    elif os.path.exists(c.classifiersFileName) and (len(sys.argv) >= 2 and sys.argv[len(sys.argv)-1] == 'me'):
        print("Loading Classifiers ...")
        classifiers_stages = utils.load_classifiers(c.classifiersFileName)
        print(len(classifiers_stages), "stages are loaded")
        img = cv2.imread('./test1.jpg')
        frame = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA)
        frame = np.copy(img)
        frameGrayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frameGrayScale = frameGrayScale.astype('uint8')
        print("max: ", np.max(frameGrayScale))
        print("min: ", np.min(frameGrayScale))
        # mean = frameGrayScale.mean()
        # std = frameGrayScale.std()
        # normalized_img = (frameGrayScale - mean)/std
        iimage = IImg.get_integral_image(frameGrayScale)
        frameWidth = frameGrayScale.shape[1]
        frameHeight = frameGrayScale.shape[0]
        rects = utils.detect_faces(frameGrayScale, iimage, frameWidth,
                                   frameHeight, classifiers_stages)
        modifiedImage = np.copy(frame)
        # for i in range(len(rects)):
        #     area = abs(rects[i][0][0]-rects[i][1][0])**2
        #     h = rects[i][0][0]
        #     w = rects[i][0][1]
        #     if(not utils.doubleCheckIsFace([area, h, w], iimage, classifiers_stages)):
        #         rects.pop(i)
        print("rects before non max suppression: ", len(rects))
        new_rects = utils.non_max_supp(rects)
        print("rects after non max suppression: ", len(new_rects))
        for rect in new_rects:
            modifiedImage = cv2.rectangle(
                frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        cv2.imshow("Test", modifiedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif os.path.exists(c.classifiersFileName):
        print("Main ...")


main()
