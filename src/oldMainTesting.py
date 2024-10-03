
import numpy as np
import matplotlib.pyplot as plt
from SVM import LSVM

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


"""
case1Samples generates the sample data to be used by the perception class
Generates  100 samples in Class 1: {(x1, x2) | −x1 + x2 > 0} 
and 100 samples in Class 2: {(x1, x2) | − x1 + x2 < 0}
Returns two numpy arrays, for x and y
"""
def case1Samples():
    case1Class1 = []
    for i in range(NUMSAMPLES):
        correct = 0
        while correct == 0:
            test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)
            if (-1 * test[0]) + test[1] > 0:
                case1Class1.append(test)
                correct = 1
            else:
                correct = 0
        # print(test)
    case1Class2 = []
    for i in range(NUMSAMPLES):
        correct = 0
        while correct == 0:
            test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)

            if (-1 * test[0]) + test[1] < 0:
                case1Class2.append(test)
                correct = 1

            else:
                correct = 0
    trainingDataX = []
    for i in range(NUMSAMPLES):
        trainingDataX.append(case1Class1[i])
        trainingDataX.append(case1Class2[i])
    trainingDataY = []
    for i in range(len(trainingDataX)):
        if (-1 * trainingDataX[i][0]) + trainingDataX[i][1] > 0:
            trainingDataY.append(1)
        else:
            trainingDataY.append(0)

    trainingDataX = np.array(trainingDataX)
    trainingDataY = np.array(trainingDataY)
    return trainingDataX, trainingDataY


"""
case2Samples generates the sample data to be used by the perception class
Generates 100 samples in Class 1: {(x1, x2) | x1 −
2x2 + 5 > 0} and 100 samples in Class 2: {(x1, x2) | x1 − 2x2 + 5 < 0}
Returns two numpy arrays, for x and y
"""
def case2Samples():
    case2Class1 = []
    for i in range(NUMSAMPLES):
        correct = 0
        while correct == 0:
            test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)

            if test[0] - (2 * test[1]) + 5 > 0:
                case2Class1.append(test)
                correct = 1
            else:
                correct = 0
        # print(test)
    case2Class2 = []
    for i in range(NUMSAMPLES):
        correct = 0
        while correct == 0:
            test = np.random.uniform(MINSAMPLESPACE, MAXSAMPLESPACE, 2)
            if test[0] - (2 * test[1]) + 5 < 0:
                case2Class2.append(test)
                correct = 1
            else:
                correct = 0
    trainingDataX = []
    for i in range(NUMSAMPLES):
        trainingDataX.append(case2Class1[i])
        trainingDataX.append(case2Class2[i])
    trainingDataY = []
    for i in range(len(trainingDataX)):
        if (trainingDataX[i][0] - (2 * trainingDataX[i][1]) + 5) > 0:
            trainingDataY.append(1)
        else:
            trainingDataY.append(0)

    trainingDataX = np.array(trainingDataX)
    trainingDataY = np.array(trainingDataY)

    return trainingDataX, trainingDataY

"""
case3Samples generates the sample data to be used by the perception class
Generates 100 samples in Class 1: {(x1, x2, x3, x4) | 0.5x1 − x2 − 10x3 + x4 + 50 > 0}
and 100 samples in Class 2: {(x1, x2, x3, x4) | 0.5x1 − x2 − 10x3 + x4 + 50 < 0}
Returns two numpy arrays, for x and y
"""
def case3Samples():

    # Initialize empty lists to store valid samples for class 1 and class 2
    class1_samples = []
    # Generate samples for class 1 until we have exactly 100 valid samples
    while len(class1_samples) < 100:
        # Generate random values for X1, X2, X3, X4
        X1 = np.random.uniform(-10, 10)
        X2 = np.random.uniform(-10, 10)
        X3 = np.random.uniform(-10, 10)
        X4 = np.random.uniform(-10, 10)

        # Check if the sample satisfies the condition for class 1
        if 0.5 * X1 - X2 - 10 * X3 + X4 + 50 > 0:
            class1_samples.append([X1, X2, X3, X4])

    class1_samples = np.array(class1_samples)

    # Now generate 100 samples for class 2 where 0.5*X1 - X2 - 10*X3 + X4 + 50 < 0
    class2_samples = []

    while len(class2_samples) < 100:
        # Generate random values for X1, X2, X3, X4
        X1 = np.random.uniform(-10, 10)
        X2 = np.random.uniform(-10, 10)
        X3 = np.random.uniform(-10, 10)
        X4 = np.random.uniform(-10, 10)

        # Check if the sample satisfies the condition for class 2
        if 0.5 * X1 - X2 - 10 * X3 + X4 + 50 < 0:
            class2_samples.append([X1, X2, X3, X4])

    class2_samples = np.array(class2_samples)

    # Perform data maniipulation to combine the samples from the two classes, shuffle them, and return them.

    # Combine both classes
    X = np.vstack((class1_samples, class2_samples))

    # Create labels for the classes (1 for class 1, 0 for class 2)
    y = np.concatenate([np.ones(100), np.zeros(100)])

    # Combine X and y and shuffle, then split back into X and y
    combined = np.hstack((X, y.reshape(-1, 1)))

    np.random.shuffle(combined)

    X_shuffled = combined[:, :-1] # All columns except the laast one are X
    y_shuffled = combined[:, -1] # The last column is y

    return X_shuffled, y_shuffled

"""
incorrectlyClassified examines each data point and determines if it is correctly classified into the correct class
returns an integer number that is the number of data points/samples that have been misclassified
"""
def incorrectlyClassified(svm, xInputs, yInputs):
    incorrect = 0
    for i in range(len(xInputs)):
        nextPrediction = svm.predict(np.array([xInputs[i]]))  # Get prediction for a single point
        correct = 1 if yInputs[i] == 1 else -1  # Convert correct labels to +1 and -1
        if nextPrediction != correct:
            incorrect += 1
    return incorrect



"""
Uses matplot lib to make a plot that shows the data points (colored) and the separation hyperplane
plots in 2d, examining x1 vs x2 in our case.
Key provided to show which class the data is in
"""
def makeCurrentPlot(current_perceptron, xValues, yValues, title):
    # Plot the data points
    class_1 = xValues[yValues == 1]
    class_2 = xValues[yValues == 0]
    graphX = []
    graphY = []
    for i in class_1:
        graphX.append(i[0])
    for i in class_1:
        graphY.append(i[1])

    graphX2 = []
    graphY2 = []
    for i in class_2:
        graphX2.append(i[0])
    for i in class_2:
        graphY2.append(i[1])

    plt.scatter(graphX, graphY, color='black', label='Class 1')
    plt.scatter(graphX2, graphY2, color='green', label='Class 2')

    #samplevalues for number of x data points, should be the same as number of y values
    samplevalues = 100
    x_values = np.linspace(MINSAMPLESPACE, MAXSAMPLESPACE, samplevalues)
    w1 = current_perceptron.weights[0]
    w2 = current_perceptron.weights[1]
    #grab the bias value from the perception class
    bias = current_perceptron.bias
    part1 = -1 * (w1/w2)
    part2 = part1 * x_values
    biasDivWeight2 = bias/w2
    y_values = part2 - biasDivWeight2
    plt.plot(x_values, y_values, 'b--', label='Our Separation Hyperplane')

    #Labeling our plot and setting the tile
    plt.title(title)
    plt.xlabel('X1 values')
    plt.ylabel('X2 values')
    plt.legend()


    plt.show()

#Generate data for case #1
trainingDataX1, trainingDataY1 = case1Samples()
input_size_1 = 2
maxP = 1
minP = -1
#generate Perceptron with input values
perceptron_1 = LSVM(0.01,100)
#Train perceptron using fit

perceptron_1.fit(trainingDataX1, trainingDataY1)

#generate sample data
X_test_1, Y_test_1 = case1Samples()
#Count the number of misclassified data samples and print
misclassified_1 = incorrectlyClassified(perceptron_1, X_test_1, Y_test_1)
print("The number of samples that have been misclassified for case #1 is: " + str(misclassified_1))

#Generate the plot for our Case #1
makeCurrentPlot(perceptron_1, X_test_1, Y_test_1, title='Case 1: Decision Boundary')
