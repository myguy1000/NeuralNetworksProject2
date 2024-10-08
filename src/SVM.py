from utils import incorrectlyClassified
import numpy as np
class LSVM:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

        #regularization param value
        self.rP = 0.01
        self.weights = 0
        self.bias = 0

    #Computes the output of the SVM model
    def forward(self, X):
        #We utilize the dot product here to find the forward prediction value
        total = 0
        index = 0
        #Had some issues where the array would be doubly nested
        if np.array(X).ndim == 2:
            X = X.flatten()

        #dot product calculation, from features and the SVM weights
        for i in X:
            total += i * self.weights[index]
            index += 1
        total += self.bias
        return total

    #Makes a prediction given training data using hinge loss and a regularization parameter
    def fit(self, X, y):
        num_samples = X.shape[0]
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)

        #make classifier be a value of -1 or 1
        y_ = [-1 if classifier <= 0 else 1 for classifier in y]

        # Training loop
        epoch_size = self.epochs
        sample_size = num_samples

        for epoch in range(epoch_size):
            for i in range(sample_size):
                #utalize hinge loss by checking if our data point is correctly classified within a sufficient margin
                data_point = y_[i] * self.forward(X[i])
                condition = (data_point >= 1)

                #This is when we classified within a margin that is greater than or equal to 1,
                #therefore we place no hinge loss penalty
                #only using the regularization value to update the weights (implementation specific)
                if condition == True:
                    update_value = self.rP * self.weights
                    self.weights = self.weights - (self.learning_rate * update_value)

                #hinge loss is applied because this data is misclassified
                else:
                    regularizationVal = self.rP * self.weights
                    gradientVal = np.dot(X[i], y_[i])
                    updateVal = regularizationVal - gradientVal
                    #update weights from gradient and regularization
                    self.weights = self.weights - ( self.learning_rate * updateVal)
                    #update bias from correct y value
                    self.bias -= self.learning_rate * y_[i]

#makes a binary prediction for a classification value by returning a 1 or a 0
    def predict(self, X):
        total = 0
        index = 0
        return_val = 0
        #in case the array is doubly nested
        if np.array(X).ndim == 2:
            X = X.flatten()
        #dot product from weights and features
        for i in X:
            total += i * self.weights[index]
            index += 1
        #Classification prediction
        if total > 0:
            return_val = 1
        elif total <= 0:
            return_val = 0
        else:
            print("ERROR WITH SVM PREDICT")

        return return_val
