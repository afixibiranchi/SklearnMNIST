# SklearnMNIST
Hello , In this Project, I have compared Six types of Classifiers with their own parameters on MNIST Handwritten Digits Datasets.
The Classifiers are KNN, Logistic Regression, Neural Network, SVM (with SVC and NuSVC) , Decision Trees and Random Forest Classifiers

I have only used 1000 images out of 60000 images i.e. only 1.67% of data. But the result is Astonishing! 

Knn Score 88.3% with Parameters {'n_neighbors': 7, 'weights': 'distance', 'algorithm': 'auto'}

Logistic Regression Score 78.6% with Parameters {'penalty': 'l2', 'tol': 0.0001}

NN Score 91.9% with Parameters {'activation': 'identity', 'learning_rate': 'invscaling', 'solver': 'lbfgs'}

Decision Tree Score 67.3% with Parameters {'splitter': 'best', 'criterion': 'entropy'}

RTrees Score 91.1% with Parameters {'max_features': 'auto', 'n_estimators': 21, 'criterion': 'gini'}

SVM 92.9% with NuSVC and Parameters {'kernel': 'linear','nu':'0.10000000000000001'}

I have tried the SVM with NuSVC and Parameters on the 1000 images from the Test Set. The Result is 919 out of 1000
images were predicted perfectly! 
