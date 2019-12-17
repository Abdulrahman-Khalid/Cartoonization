import violajones.AdaBoost as AdaBoost
import violajones.Utils as utils
import violajones.IntegralImage as IImg
import constants as c


print('Loading faces training data set...')
facesTraining = utils.load_images(c.facesTrainingPath)
facesTrainingIntegral = list(
    map(IImg.get_integral_image, facesTraining))
print(len(facesTraining), 'Training faces data set are loaded!')

print('Loading non-faces training data set...')
nonFacesTraining = utils.load_images(c.nonFacesTrainingPath)
nonFacesTrainingIntegral = list(
    map(IImg.get_integral_image, nonFacesTraining))
print(len(nonFacesTraining), 'Training non-faces data set are loaded!')

classifiers = AdaBoost.adaBoostLearn(facesTrainingIntegral, nonFacesTrainingIntegral,
                                     c.minFeatureHeight, c.maxFeatureHeight, c.minFeatureWidth, c.maxFeatureWidth, c.classifiersNum)

utils.save_classifiers(classifiers, c.classifiersFileName)
print('Saved classifiers')
