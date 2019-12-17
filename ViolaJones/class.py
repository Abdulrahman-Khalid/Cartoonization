
from classifiers import classifiers, stages
from violajones.Feature import FeatureType, Feature
import violajones.Utils as utils
from itertools import islice


def make_sure():
    classifiers_loaded = utils.load_classifiers('./classifiers.pkl')
    print(classifiers_loaded[0][0])
    if(len(stages) == len(classifiers_loaded)):
        print('loaded sucessfully')


def make_classifiers():
    classfiers_new = []
    for c in classifiers:
        topLeft = (c[2], c[1])
        if(c[0] == 1):
            featureType = FeatureType.TWO_HORIZONTAL
            height = c[3]
            width = c[6]*2
        elif(c[0] == 2):
            featureType = FeatureType.THREE_HORIZONTAL
            height = c[3]
            width = c[6]*3
        elif(c[0] == 3):
            featureType = FeatureType.TWO_VERTICAL
            height = c[3]*2
            width = c[6]
        elif(c[0] == 4):
            featureType = FeatureType.THREE_VERTICAL
            height = c[3]*3
            width = c[6]
        elif(c[0] == 5):
            featureType = FeatureType.THREE_VERTICAL
            height = c[3]*2
            width = c[6]*2
        weight = c[7]
        threshold = c[8]
        polarity = c[9]
        cl = Feature(featureType, topLeft, width,
                     height, weight, threshold, polarity)
        classfiers_new.append(cl)

    Inputt = iter(classfiers_new)
    classfiers_new_new = [list(islice(Inputt, num)) for num in stages]
    utils.save_classifiers(classfiers_new_new, './classifiers.pkl')


make_classifiers()
make_sure()
