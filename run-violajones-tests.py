import violajones.Utils as utils
import violajones.IntegralImage as IImg
import constants as c
from violajones_detector import _bbox_to_dlib_rectangle

import numpy as np
import cv2

print("Loading Classifiers...")
classifiers = utils.load_classifiers(c.classifiersFileName)
print(len(classifiers), "Classifier are loaded")
classifiers_stages = utils.classifiers_to_classifiers_stages(
    classifiers)

print('Loading faces testing data set...')
facesTesting = utils.load_images(c.facesTestingPath)
facesTestingIntegral = list(
    map(IImg.get_integral_image, facesTesting))
print(len(facesTesting), 'Testing faces data set are loaded!')

print('Loading testing non faces data set...')
nonFacesTesting = utils.load_images(c.nonFacesTestingPath)
nonFacesTestingIntegral = list(
    map(IImg.get_integral_image, nonFacesTesting))
print(len(nonFacesTesting), 'Testing non faces data set are loaded!')

print('Testing selected classifiers...')
correctFacesCount = 0
correctNonFacesCount = 0

correctFacesCount = sum(list(len(utils.detect_faces(
    face, 24, 24, classifiers_stages, _bbox_to_dlib_rectangle)) for face in facesTestingIntegral))
correctNonFacesCount = len(nonFacesTesting) - sum(list(len(utils.detect_faces(
    nonFace, 24, 24, classifiers_stages, _bbox_to_dlib_rectangle)) for nonFace in nonFacesTestingIntegral))

print('Accuracy:-\nFaces: ' + str(correctFacesCount) + '/' + str(len(facesTesting))
      + '  (' + str((float(correctFacesCount) / len(facesTesting))
                    * 100) + '%)\nNon Faces: '
        + str(correctNonFacesCount) + '/' +
      str(len(nonFacesTesting)) + '  ('
        + str((float(correctNonFacesCount) / len(nonFacesTesting)) * 100) + '%)')


img = cv2.imread('./ViolaJones/test1.jpg')
# frame = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA)
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
rects = utils.detect_faces_non_max_supp(iimage, frameWidth,
                                        frameHeight, classifiers_stages)
modifiedImage = np.copy(frame)
print("rects before non max suppression: ", len(rects))
new_rects = utils.non_max_supp(rects)
print("rects after non max suppression: ", len(new_rects))
for rect in new_rects:
    modifiedImage = cv2.rectangle(
        frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
cv2.imshow("Test", modifiedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
