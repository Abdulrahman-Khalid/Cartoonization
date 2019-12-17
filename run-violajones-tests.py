import violajones.Utils as utils
import violajones.IntegralImage as IImg
import constants as c
from violajones_detector import _bbox_to_dlib_rectangle

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
