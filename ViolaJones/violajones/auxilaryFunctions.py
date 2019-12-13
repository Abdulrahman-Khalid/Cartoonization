import skimage.io as io
from skimage.exposure import histogram
from skimage.color import rgb2gray,rgb2hsv
from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny

import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# Convolution: If not needed, delete it later
from scipy.signal import convolve2d
from scipy import fftpack
import math

#Used for Adaboost classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

import numpy as np
from PIL import Image, ImageOps
import os
import cv2




# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    

def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

def Grey_img(image):
    #return image.convert('LA')
    return rgb2gray(image)
    # size = image.shape
    # if (len(size) == 3):
    #     grey_image = rgb2gray(image)
    # else:
    #     grey_image = image/255
    # return grey_image

#Resizing images for datasets
def resizeBy24x24(img_dest):
    img = Image.open(img_dest).convert('L')
    img_down = img.resize((24, 24), Image.ANTIALIAS)
    return img_down
    
#Resizing with maintaining aspect ratios
def RESIZE(image_dst,img_save = None):
    img = Image.open(image_dst)
    img_down = img.thumbnail((480, 320))
    img_down.save(img_save)
    

#Integral Image function
def integralImage (image):
    n,m = image.shape
    II = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            #print (i,j)
            II[i,j] = sum(sum(image[0:i+1,0:j+1]))
    return II


def get_integral_image(imgArr):
    rowAccumelate = np.zeros(imgArr.shape)
    integralImg = np.zeros((imgArr.shape[0] + 1, imgArr.shape[1] + 1))
    for y in range(imgArr.shape[1]):
        for x in range(imgArr.shape[0]):
            rowAccumelate[x, y] = rowAccumelate[x-1, y] + imgArr[x, y]
            integralImg[x + 1, y + 1] = integralImg[x + 1, y] + \
                rowAccumelate[x, y]
    return integralImg 
#another implementation using opensv for higher performance, Doesnt work !
# def integralImage_OpenCv(image):
#     return cv2.integral2(image)
#To get zero mean and unit var. images

def normalizeImages (image):
    max_px = np.max(image)
    min_px = np.min(image)
    std = max_px - min_px
    mean = image.mean()
    image =(image - mean) / std
    return image

def calIntegralImage(integralImg,i,j,w,h): # i,j of the index to be calculated while w,h is the size of the feature image
    topLeft = 0
    top = 0
    left = 0
    if(i - w >= 0 and j - h >= 0):
        topLeft = integralImg[i - w,j - h]
    if(i - w >= 0):
        top = integralImg[i - w,j]
    if(j - h >= 0):
        left = integralImg[i ,j - h]
    #print(integralImg[i,j],' ',left,' ',top,' ',topLeft)
    result = integralImg[i,j] - left - top + topLeft
    return result



def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

