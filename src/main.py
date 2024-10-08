import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_breast_cancer  # Use if importing straight from the website instead of from the downloaded data file
from sklearn.model_selection import train_test_split
from utils import incorrectlyClassified, makeCurrentPlot
from logisticRegression import LogisticRegression
from SVM import LSVM
from WidrowHoff import WidrowHoff
from WestonWatkins import WestonWatkinsSVM  # Import the WestonWatkinsSVM class

# Constants
LEARNING_RATE = 0.01
NUM_EPOCHS = 100

# Dataset 1: digits
print("digits dataset")
digits = load_digits()
X, y = digits.data, digits.target

# Convert the original dataset consisting of 10 classes to 2 classes (i.e., binary classification problem)
mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the number of features
num_inputs = X_train.shape[1]

# Generate perceptrons with input values
lr = LogisticRegression(num_inputs, LEARNING_RATE, NUM_EPOCHS, 0.5, -1, 1)  # Third to last arg is the threshold
svm = LSVM(LEARNING_RATE, NUM_EPOCHS)
wh = WidrowHoff(num_inputs, LEARNING_RATE, NUM_EPOCHS, -1, 1)

# Instantiate the Weston Watkins SVM for binary classification
num_classes = 2  # For binary classification
regularization = 0.01  # Regularization parameter
ww_svm = WestonWatkinsSVM(
    input_size=num_inputs,
    num_classes=num_classes,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    min_value=-1,
    max_value=1,
    regularization=regularization
)

# Fit perceptrons
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
wh.fit(X_train, y_train)
ww_svm.fit(X_train, y_train)  # Note: Updated to remove num_epochs parameter

# Count the number of misclassified data samples and print
lr_misclassified_train = incorrectlyClassified(lr, X_train, y_train, "LR")
lr_misclassified_test = incorrectlyClassified(lr, X_test, y_test, "LR")
svm_misclassified_train = incorrectlyClassified(svm, X_train, y_train, "SVM")
svm_misclassified_test = incorrectlyClassified(svm, X_test, y_test, "SVM")
wh_misclassified_train = incorrectlyClassified(wh, X_train, y_train, "WH")
wh_misclassified_test = incorrectlyClassified(wh, X_test, y_test, "WH")

# For Weston Watkins SVM, get predictions and calculate misclassified samples
ww_predictions_train = ww_svm.predict(X_train)
ww_predictions_test = ww_svm.predict(X_test)
ww_misclassified_train = np.sum(ww_predictions_train != y_train)
ww_misclassified_test = np.sum(ww_predictions_test != y_test)

# Number of samples/rows
num_samples_train = X_train.shape[0]
num_samples_test = X_test.shape[0]

print("num samples train: ", num_samples_train)
print("num samples test: ", num_samples_test)

# Results
print("Logistic Regression train accuracy: ", 1 - (lr_misclassified_train / num_samples_train))
print("Logistic Regression test accuracy: ", 1 - (lr_misclassified_test / num_samples_test))

print("SVM train accuracy: ", 1 - (svm_misclassified_train / num_samples_train))
print("SVM test accuracy: ", 1 - (svm_misclassified_test / num_samples_test))

print("WidrowHoff train accuracy: ", 1 - (wh_misclassified_train / num_samples_train))
print("WidrowHoff test accuracy: ", 1 - (wh_misclassified_test / num_samples_test))

print("Weston Watkins SVM train accuracy: ", 1 - (ww_misclassified_train / num_samples_train))
print("Weston Watkins SVM test accuracy: ", 1 - (ww_misclassified_test / num_samples_test))

# --------------------------------------------------------------
# Part 4: Testing the Weston-Watkins SVM using a scikit-learn dataset for multiclass classification.

print("\nDigits dataset (Multiclass Classification)")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the digits dataset with all classes
digits = load_digits()
X_multi, y_multi = digits.data, digits.target

# Split into train and test datasets
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Number of classes
num_classes_multi = len(np.unique(y_multi))
print("Number of classes:", num_classes_multi)

# Define the number of features
num_inputs_multi = X_train_multi.shape[1]

# Instantiate the Weston Watkins SVM for multiclass classification
ww_svm_multi = WestonWatkinsSVM(
    input_size=num_inputs_multi,
    num_classes=num_classes_multi,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    min_value=-1,
    max_value=1,
    regularization=regularization
)

# Fit the model
ww_svm_multi.fit(X_train_multi, y_train_multi)

# Predict on training and test data
ww_predictions_train_multi = ww_svm_multi.predict(X_train_multi)
ww_predictions_test_multi = ww_svm_multi.predict(X_test_multi)

# Calculate accuracies
ww_train_accuracy_multi = accuracy_score(y_train_multi, ww_predictions_train_multi)
ww_test_accuracy_multi = accuracy_score(y_test_multi, ww_predictions_test_multi)

# Print the results
print("Weston Watkins SVM train accuracy (multiclass):", ww_train_accuracy_multi)
print("Weston Watkins SVM test accuracy (multiclass):", ww_test_accuracy_multi)

# Optionally, print classification report and confusion matrix
print("\nClassification Report (Test Data):")
print(classification_report(y_test_multi, ww_predictions_test_multi))

print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(y_test_multi, ww_predictions_test_multi))






# Dataset 2: breast cancer
print("breast cancer dataset")
bc = load_breast_cancer()
X, y = bc.data, bc.target

# Normalize the input to prevent overflow
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the number of features
num_inputs = X_train.shape[1]

# Generate perceptrons with input values
lr = LogisticRegression(num_inputs, LEARNING_RATE, NUM_EPOCHS, 0.5, -1, 1)  # Third to last arg is the threshold
svm = LSVM(LEARNING_RATE, NUM_EPOCHS)
wh = WidrowHoff(num_inputs, LEARNING_RATE, NUM_EPOCHS, -1, 1)

# Instantiate the Weston Watkins SVM for binary classification
num_classes = 2  # For binary classification
regularization = 0.01  # Regularization parameter
ww_svm = WestonWatkinsSVM(
    input_size=num_inputs,
    num_classes=num_classes,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    min_value=-1,
    max_value=1,
    regularization=regularization
)

# Fit perceptrons
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
wh.fit(X_train, y_train)
ww_svm.fit(X_train, y_train)  # Note: Updated to remove num_epochs parameter

# Count the number of misclassified data samples and print
lr_misclassified_train = incorrectlyClassified(lr, X_train, y_train, "LR")
lr_misclassified_test = incorrectlyClassified(lr, X_test, y_test, "LR")
svm_misclassified_train = incorrectlyClassified(svm, X_train, y_train, "SVM")
svm_misclassified_test = incorrectlyClassified(svm, X_test, y_test, "SVM")
wh_misclassified_train = incorrectlyClassified(wh, X_train, y_train, "WH")
wh_misclassified_test = incorrectlyClassified(wh, X_test, y_test, "WH")

# For Weston Watkins SVM, get predictions and calculate misclassified samples
ww_predictions_train = ww_svm.predict(X_train)
ww_predictions_test = ww_svm.predict(X_test)
ww_misclassified_train = np.sum(ww_predictions_train != y_train)
ww_misclassified_test = np.sum(ww_predictions_test != y_test)

# Number of samples/rows
num_samples_train = X_train.shape[0]
num_samples_test = X_test.shape[0]

print("num samples train: ", num_samples_train)
print("num samples test: ", num_samples_test)

# Results
print("Logistic Regression train accuracy: ", 1 - (lr_misclassified_train / num_samples_train))
print("Logistic Regression test accuracy: ", 1 - (lr_misclassified_test / num_samples_test))

print("SVM train accuracy: ", 1 - (svm_misclassified_train / num_samples_train))
print("SVM test accuracy: ", 1 - (svm_misclassified_test / num_samples_test))

print("WidrowHoff train accuracy: ", 1 - (wh_misclassified_train / num_samples_train))
print("WidrowHoff test accuracy: ", 1 - (wh_misclassified_test / num_samples_test))

print("Weston Watkins SVM train accuracy: ", 1 - (ww_misclassified_train / num_samples_train))
print("Weston Watkins SVM test accuracy: ", 1 - (ww_misclassified_test / num_samples_test))
