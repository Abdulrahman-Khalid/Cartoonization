#This function shpuld take a window with size e x e
# apply on it some funcitons:
# 1-Grey Scale
# 2-Integral Image
# 3-Pass that image through multiple stages,
#   for each stage it will be tested on some classifiers
#   these classifiers have the feature index, its threshold, its alpha and its polarity
#   so we should take that feature and apply it on the integral image
# 4- when it fails any Layer of classifiers, we should consider that window as FALSE
# 5- If it passed, then its a face

from auxilaryFunctions import Grey_img,integralImage,Image,np,io,get_integral_image,normalizeImages
from computeFeature import computerFeatureFunc
import time,math,pickle



def cascade(box,iimage):
    start = time.time()
    
    #TODO here read the features and place them in different layers
    #Layers = getLayers()
    
    end2 = time.time()
    print ("Obtaining Layers in: ",end2-start)
    
    #Loop on stages
    for layer in Layers:

        sum_hypotheses = 0.
        sum_alphas = 0.
        #For each feature in that layer:
        for feature in layer:
            #These numbers to be changed
            #feature_id contains the data needed for teh computerFeatureFunc 
            #These lines need to be modified, using enum

            feature_id = feature[0:6]
            feature_value = (computerFeatureFunc(box,feature_id,iimage))
            vote = (np.sign((feature[polarity] * feature[threshold]) - (feature[polarity] * feature_value)) + 1) // 2

            sum_hypotheses += feature[alpha] * vote
            sum_alphas += feature[alpha]
        #Check on predictions
        #predictions_Condition ?
        if (not sum_hypotheses >= .5*sum_alphas):
            #if face, returns true at the end
            return False
    end3 = time.time()
    print ("Time taking for loopin on layers = ",end3 -end2)

    print ("Total time: ",end3 - start)
    return True
    