# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 19:52:18 2016

@author: sezan1992
"""
from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
from skimage.feature import hog as HOG

#Importing Models
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Data Preparation

Zero = '/home/sezan92/SklearnMNIST/mnist_png/training/0'
One= '/home/sezan92/SklearnMNIST/mnist_png/training/1'
Two = '/home/sezan92/SklearnMNIST/mnist_png/training/2'
Three = '/home/sezan92/SklearnMNIST/mnist_png/training/3'
Four = '/home/sezan92/SklearnMNIST/mnist_png/training/4'
Five = '/home/sezan92/SklearnMNIST/mnist_png/training/5'
Six = '/home/sezan92/SklearnMNIST/mnist_png/training/6'
Seven = '/home/sezan92/SklearnMNIST/mnist_png/training/7'
Eight = '/home/sezan92/SklearnMNIST/mnist_png/training/8'
Nine = '/home/sezan92/SklearnMNIST/mnist_png/training/9'

trainData = []
responseData = []
NumberList = []
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



def ReadImages(ListName,FolderName,Label):
    global NumberList
    global responseData
    global trainData
    global hog
    global cv2
    global imutils
    global winSize
    ListName= ListName[0:100]
    for image in ListName:
        img = cv2.imread(join(FolderName,image))
        NumberList.append(img)    
        feature = HOG(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
        trainData.append(feature.T)
        responseData.append(Label)


ReadImages(ZeroImages,Zero,0)
ReadImages(OneImages,One,1)
ReadImages(TwoImages,Two,2)
ReadImages(ThreeImages,Three,3)
ReadImages(FourImages,Four,4)
ReadImages(FiveImages,Five,5)
ReadImages(SixImages,Six,6)
ReadImages(SevenImages,Seven,7)
ReadImages(EightImages,Eight,8)
ReadImages(NineImages,Nine,9)

X = np.float32(trainData)
y= np.float32(responseData)
#Real Stuff  Classifier Training

#knn with gridsearch
knn = KNeighborsClassifier()
k_range = list(range(1,31))
leaf_range = list(range(1,40))
weight_options = ['uniform', 'distance']
algorithm_options =  ['auto', 'ball_tree', 'kd_tree', 'brute']
param_gridKnn = dict(n_neighbors = k_range,
                     weights = weight_options,
                     algorithm = algorithm_options
                     #leaf_size = leaf_range
                     )
gridKNN = GridSearchCV(knn,param_gridKnn,cv=10,
                       scoring = 'accuracy') 
gridKNN.fit(X,y)
print "Knn Score "+ str(gridKNN.best_score_)
print "Knn  best Params "+str(gridKNN.best_params_)
#LogReg with gridSearch

logreg = LogisticRegression()
penalty_options =['l1','l2']
solver_options = ['liblinear','newton_cg','lbfgs','sag']
tol_options = [0.0001,0.00001,0.000001,0.000001]
param_gridLog = dict(penalty=penalty_options,
                     tol=tol_options)
gridLog = GridSearchCV(logreg,param_gridLog,cv=10,scoring='accuracy')
gridLog.fit(X,y)

print "LogReg Score "+ str(gridLog.best_score_)
print "LogReg  best Params "+str(gridLog.best_params_)
#NN with gridSearch

NN = MLPClassifier(hidden_layer_sizes=  (45,27,18))
activation_options = ['identity', 'logistic', 'tanh', 'relu']
solver_options =['lbfgs', 'sgd', 'adam']
learning_rate_options = ['constant', 'invscaling', 'adaptive']
param_gridNN = dict(activation=activation_options,
                    solver=solver_options,
                    learning_rate = learning_rate_options)
gridNN = GridSearchCV(NN,param_gridNN,cv=10,
                      scoring = 'accuracy')
gridNN.fit(X,y)
print "NN Score "+ str(gridNN.best_score_)
print "NN  best Params "+str(gridNN.best_params_)

#SVM with SVC
flag = False
if flag is True:
    svm = SVC()
    svmNu = NuSVC()
    nu_options =np.arange(0.1,1,100)
    kernel_options = [ 'linear', 'sigmoid', 'rbf']
    param_gridSVM = dict(kernel = kernel_options)
    param_gridSVMNu = dict(kernel = kernel_options,nu =
                         nu_options)
    gridSVM = GridSearchCV(svm,param_gridSVM,cv=10,
                       scoring = 'accuracy')
    gridSVM.fit(X,y)
    print "SVM Score "+str(gridSVM.best_score_)
    print "SVM best Params"+str(gridSVM.best_params_)
    gridSVMNu = GridSearchCV(svmNu,param_gridSVMNu,cv=10,
                       scoring = 'accuracy')
    gridSVMNu.fit(X,y)
    print "SVM with NuSVC Score "+str(gridSVMNu.best_score_)
    print "SVM with NuSVC best Params"+str(gridSVMNu.best_params_)

#Random Forest
dtree = DecisionTreeClassifier(random_state=0)
criterion_options = ['gini','entropy']
splitter_options =['best','random']

param_gridDtree = dict(criterion =criterion_options,splitter=splitter_options)

gridDtree = GridSearchCV(dtree,param_gridDtree,cv=10,scoring='accuracy')
gridDtree.fit(X,y)

print "Decision Tree Score "+str(gridDtree.best_score_)
print "Decision Tree params "+str(gridDtree.best_params_)

#Random Forest Classifier with GridSearch
random = RandomForestClassifier()
n_estimators_range = list(range(1,31))
criterion_options = ['gini','entropy']
max_features_options =['auto','log2', None]
param_grid = dict(n_estimators =n_estimators_range,
                  criterion= criterion_options,
                  max_features =max_features_options)
gridRandom = GridSearchCV(random,param_grid,cv=10,
                          scoring='accuracy')
gridRandom.fit(X,y)

print "RTrees Score "+str(gridRandom.best_score_)
print "RTrees Best Params " +str(gridRandom.best_params_)


        
