import numpy as np
import matplotlib.pyplot as plt 

"""
incorrectlyClassified examines each data point and determines if it is correctly classified into the correct class
returns an integer number that is the number of data points/samples that have been misclassified
"""
def incorrectlyClassified(perceptron, xInputs, yInputs, p_type): 
    if p_type == "SVM":
        incorrect = 0
        for i in range(len(xInputs)):
            nextPrediction = perceptron.predict(np.array([xInputs[i]]))  # Get prediction for a single point
#            print("current prediction: ", nextPrediction)
            correct = 1 if yInputs[i] == 1 else -1  # Convert correct labels to +1 and -1
            if nextPrediction != correct:
                incorrect += 1
        return incorrect

    else:
        incorrect = 0
        for i in range(len(xInputs)):
            nextPrediction = perceptron.forward(xInputs[i])
###            print("current prediction: ", nextPrediction)
            correct = yInputs[i]
            if nextPrediction != correct:
                incorrect = incorrect + 1
            else:
                incorrect = incorrect + 0
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
    bias = current_perceptron.getBias()
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
