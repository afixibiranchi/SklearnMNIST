#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 23:30:19 2017

@author: sezan92
"""

from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
from skimage.feature import hog as HOG

#Importing Models
from sklearn.svm import NuSVC
Zero = '/home/sezan92/SklearnMNIST/mnist_png/testing/0'
One= '/home/sezan92/SklearnMNIST/mnist_png/testing/1'
Two = '/home/sezan92/SklearnMNIST/mnist_png/testing/2'
Three = '/home/sezan92/SklearnMNIST/mnist_png/testing/3'
Four = '/home/sezan92/SklearnMNIST/mnist_png/testing/4'
Five = '/home/sezan92/SklearnMNIST/mnist_png/testing/5'
Six = '/home/sezan92/SklearnMNIST/mnist_png/testing/6'
Seven = '/home/sezan92/SklearnMNIST/mnist_png/testing/7'
Eight = '/home/sezan92/SklearnMNIST/mnist_png/testing/8'
Nine = '/home/sezan92/SklearnMNIST/mnist_png/testing/9'

testData=[]
ZeroImages = [ f for f in listdir(Zero) if isfile(join(Zero,f)) ]
OneImages = [ f for f in listdir(One) if isfile(join(One,f)) ]
TwoImages = [ f for f in listdir(Two) if isfile(join(Two,f)) ]
ThreeImages = [ f for f in listdir(Three) if isfile(join(Three,f)) ]
FourImages = [ f for f in listdir(Four) if isfile(join(Four,f)) ]
FiveImages = [ f for f in listdir(Five) if isfile(join(Five,f)) ]
SixImages = [ f for f in listdir(Six) if isfile(join(Six,f)) ]
SevenImages = [ f for f in listdir(Seven) if isfile(join(Seven,f)) ]
EightImages = [ f for f in listdir(Eight) if isfile(join(Eight,f)) ]
NineImages = [ f for f in listdir(Nine) if isfile(join(Nine,f)) ]

def ReadImages(ListName,FolderName):
    global NumberList
    global responseData
    global testData
    global hog
    global cv2
    global imutils
    global winSize
    ListName= ListName[0:100]
    for image in ListName:
        img = cv2.imread(join(FolderName,image))
        NumberList.append(img)    
        feature = HOG(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
        testData.append(feature.T)
svm = NuSVC(nu=0.100000000000000001,'linear')
svm.fit()
  