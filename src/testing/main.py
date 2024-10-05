import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_wine # use if importing straight from the website instead of from the downloaded data file
from sklearn.model_selection import train_test_split
from utils import incorrectlyClassified, makeCurrentPlot

from logisticRegression import LogisticRegression
from SVM import LSVM
from WidrowHoff import WidrowHoff

#THRESHOLD = 0
LEARNING_RATE = 0.01
NUM_EPOCHS = 100







# Dataset 1: digits
print("digits dataset")
digits = load_digits()
X, y = digits.data, digits.target

# Convert the original dataset consisting of 10 classes to 2 classes (i.e. binary classsification problem)
mask = (y == 0) | (y == 1)

X = X[mask]
y = y[mask]

#print("X shape: ", X.shape)
#print("list of y classes:")
#print(np.unique(y, return_counts = True))

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define the number of features
num_inputs = X_train.shape[1]

#generate perceptrons with input values
lr = LogisticRegression(num_inputs, LEARNING_RATE, NUM_EPOCHS, 0.5, -1, 1) # third to last arg is the threshold
#svm = LSVM(LEARNING_RATE, NUM_EPOCHS)
#wh = WidrowHoff(num_inputs, LEARNING_RATE, NUM_EPOCHS, -1, 1)

#fit perceptrons
lr.fit(X_train, y_train)
#svm.fit(X_train, y_train)
#wh.fit(X_train, y_train)

##Count the number of misclassified data samples and print
lr_misclassified_train = incorrectlyClassified(lr, X_train, y_train, "LR")
lr_misclassified_test = incorrectlyClassified(lr, X_test, y_test, "LR")
#svm_misclassified_train = incorrectlyClassified(svm, X_train, y_train, "SVM")
#svm_misclassified_test = incorrectlyClassified(svm, X_test, y_test, "SVM")
#wh_misclassified_train = incorrectlyClassified(wh, X_train, y_train, "WH")
#wh_misclassified_test = incorrectlyClassified(wh, X_test, y_test, "WH")

# number of samples/rows
num_samples_train = X_train.shape[0]
num_samples_test = X_test.shape[0]

print("num samples train: ", num_samples_train)
print("num samples test: ", num_samples_test)

#results
print("Logistic regression test accuracy: ", lr_misclassified_train)
print("Logistic regression test accuracy: ", lr_misclassified_test)

#print("SVM train accuracy: ", 1 - (svm_misclassified_train / num_samples_train))
#print("SVM test accuracy: ", 1 - (svm_misclassified_test / num_samples_test))

#print("WidrowHoff train accuracy: ", wh_misclassified_train / num_samples_train)
#print("WidrowHoff test accuracy: ", wh_misclassified_test / num_samples_test)






# Dataset 2: wine
##print("")
##print("wine recognition dataset")

##wine = load_wine()
##X, y = wine.data, wine.target

##print("complete dataset X shape: ", X.shape)
##print("list of y classes:")
##print(np.unique(y, return_counts = True))

# convert actual y values from floats to ints (ex. 1.0 to 1)
##y= y.astype(int)

# Convert the original dataset consisting of 10 classes to 2 classes (i.e. binary classsification problem)
##mask = (y == 0) | (y == 1)

##X = X[mask]
##y = y[mask]

##print("X shape: ", X.shape)
##print("list of y classes:")
##print(np.unique(y, return_counts = True))


# split into train and test datasets
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# number of features
##num_inputs = X_train.shape[1]

#generate perceptrons with input values
#lr = LogisticRegression(num_inputs, LEARNING_RATE, NUM_EPOCHS, THRESHOLD, 0, 2)
##svm = LSVM(LEARNING_RATE, NUM_EPOCHS)
#wh = WidrowHoff(num_inputs, LEARNING_RATE, NUM_EPOCHS)

# fit perceptrons
#lr.fit(X_train, y_train)
##svm.fit(X_train, y_train)
#wh.fit(X_train, y_train)

##Count the number of misclassified data samples and print
#lr_misclassified_train = incorrectlyClassified(lr, X_train, y_train, "LR")
#lr_misclassified_test = incorrectlyClassified(lr, X_test, y_test, "LR")
##svm_misclassified_train = incorrectlyClassified(svm, X_train, y_train, "SVM")
##svm_misclassified_test = incorrectlyClassified(svm, X_test, y_test, "SVM")
#wh_misclassified_train = incorrectlyClassified(wh, X_train, y_train, "WH")
#wh_misclassified_test = incorrectlyClassified(wh, X_test, y_test, "WH")

# number of samples (rows)
##num_samples_train = X_train.shape[0]
##num_samples_test = X_test.shape[0]

# results
#print("Logistic regression test accuracy: ", 1 - lr_misclassified_train / num_samples_train)
#print("Logistic regression test accuracy: ", 1 - lr_misclassified_test / num_samples_test)

##print("SVM train accuracy: ", 1 - (svm_misclassified_train / num_samples_train))
##print("SVM test accuracy: ", 1 - (svm_misclassified_test / num_samples_test))

#print("WidrowHoff train accuracy: ", 1 - (wh_misclassified_train / num_samples_train))
#print("WidrowHoff test accuracy: ", 1 - (wh_misclassified_test / num_samples_test))
