import numpy as np
import matplotlib.pyplot as plt
import gzip
from sklearn.datasets import load_digits#, fetch_covtype # use if importing straight from the website instead of from the downloaded data file
from utils import incorrectlyClassified, makeCurrentPlot
from perceptron import Perceptron  # Ensure the class is saved as current_perceptron.py

THRESHOLD = 0
#the magnitude at which the models weights are updated during a single increment
#lower learning rate may lead to a more accurate result but longer to reach that result
LEARNING_RATE = 0.1
#how many times the dataset will be passed through the model
NUM_EPOCHS = 50
NUMSAMPLES = 100
#min value a sample can take
MINSAMPLESPACE = -100
#max value a sample can take
MAXSAMPLESPACE = 100


# Dataset 1: digits
digits = load_digits()
X, y = digits.data, digits.target

#generate Perceptron with input values
print(X.shape)
#input_size_1 = 
maxP = 1
minP = -1
#perceptron_1 = Perceptron(input_size_1,minP,maxP,THRESHOLD)

#perceptron_1.fit(trainingDataX1, trainingDataY1, LEARNING_RATE, NUM_EPOCHS)

# Dataset 2: covtype
# Open the .gz file and load it into a NumPy array
with gzip.open('../data/covtype.data.gz', 'rt') as f:  # 'rt' for reading text
    data = np.loadtxt(f, delimiter=',')  # Adjust the delimiter if needed (e.g., ',' for CSV)

# Check the shape of the loaded array
print(data.shape)
#print(data[,:10])

#covtype = fetch_covtype()
X, y = covtype.data, covtype.target
print(X.shape)

##Generate data for case #1
#trainingDataX1, trainingDataY1 = case1Samples()
#input_size_1 = 2
#maxP = 1
#minP = -1
##generate Perceptron with input values
#perceptron_1 = Perceptron(input_size_1,minP,maxP,THRESHOLD)
##Train perceptron using fit
#perceptron_1.fit(trainingDataX1, trainingDataY1, LEARNING_RATE, NUM_EPOCHS)
#
##generate sample data
#X_test_1, Y_test_1 = case1Samples()
##Count the number of misclassified data samples and print
#misclassified_1 = incorrectlyClassified(perceptron_1, X_test_1, Y_test_1)
#print("The number of samples that have been misclassified for case #1 is: " + str(misclassified_1))
#
##Generate the plot for our Case #1
#makeCurrentPlot(perceptron_1, X_test_1, Y_test_1, title='Case 1: Decision Boundary')
#
#
##Generate Data for Case#2
#trainingDataX2, trainingDataY2 = case2Samples()
#input_size_2 = 2
#maxP = 1
#minP = -1
##create perceptron for case 2
#perceptron_2 = Perceptron(input_size_2,minP,maxP,THRESHOLD)
#
##train perceptron2 from the sample cases
#perceptron_2.fit(trainingDataX2, trainingDataY2, LEARNING_RATE, NUM_EPOCHS)
##plot the data and make test samples
#X_test_2, Y_test_2 = case2Samples()
#misclassified_2 = incorrectlyClassified(perceptron_2, X_test_2, Y_test_2)
#print("The number of samples that have been misclassified for case #2 is: " + str(misclassified_2))
#makeCurrentPlot(perceptron_2, X_test_2, Y_test_2, title='Case 2: Decision Boundary')
#
## Case 3
#
#X_train_3, y_train_3 = case3Samples()
#perceptron_3 = Perceptron(4,-1,1,THRESHOLD)
#
## regular fit function
#
#
##train
#perceptron_3.fit(X_train_3, y_train_3, LEARNING_RATE, NUM_EPOCHS)
#
## Determine count of misclassified samples
#misclassified_3 = incorrectlyClassified(perceptron_3, X_train_3, y_train_3)
#print(f"Case 3a: Misclassified samples reg-fit (train) = {misclassified_3}")
#
##test
#X_test_3, y_test_3 = case3Samples()
#
## Determine count of misclassified samples
#misclassified_3 = incorrectlyClassified(perceptron_3, X_test_3, y_test_3)
#print(f"Case 3b: Misclassified samples reg-fit (test) = {misclassified_3}")
#
#
## GD fit function
#
#
##train
#perceptron_3.fit_GD(X_train_3, y_train_3, LEARNING_RATE, NUM_EPOCHS)
#
## Determine count of misclassified samples
#misclassified_3 = incorrectlyClassified(perceptron_3, X_train_3, y_train_3)
#print(f"Case 3c: Misclassified samples GD-fit (train) = {misclassified_3}")
#
##test
## Determine count of misclassified samples
#misclassified_3 = incorrectlyClassified(perceptron_3, X_test_3, y_test_3)
#print(f"Case 3d: Misclassified samples GD-fit (test) = {misclassified_3}")
