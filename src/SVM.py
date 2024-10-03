import numpy as np
class LSVM:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.rP = 0.01
        self.weights = 0
        self.bias = 0


    def forward(self, X):
        #We utalize the dot product here to find the forward prediction value
        total = 0
        index = 0
        #print(X)
        if np.array(X).ndim == 2:
            X = X.flatten()


        for i in X:
            total += i * self.weights[index]
            index += 1
        total += self.bias
        return total

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        # Convert labels to -1 and 1 for SVM
        y_ = np.where(y <= 0, -1, 1)

        # Training loop
        for epoch in range(self.epochs):
            for i in range(num_samples):
                condition = y_[i] * self.forward(X[i]) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.rP * self.weights)
                else:
                    self.weights -= self.learning_rate * (
                            2 * self.rP * self.weights - np.dot(X[i], y_[i]))
                    self.bias -= self.learning_rate * y_[i]

    def predict(self, X):
        total = 0
        index = 0
        return_val = 0
        # print(X)
        if np.array(X).ndim == 2:
            X = X.flatten()


        for i in X:
            total += i * self.weights[index]
            index += 1
        if total > 0:
            return_val = 1
        elif total < 0:
            return_val = -1
        elif total == 0:
            return_val = 0
        else:
            print("ERROR WITH SVM PREDICT")

        return return_val
