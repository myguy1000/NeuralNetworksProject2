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
        #print("INPUT LIST" + str(X))
        for i in X:
            total += i * self.weights[index]
            index += 1
        total += self.bias
        #print(total)
        return np.dot(X, self.weights) + self.bias

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

        return np.sign(self.forward(X))
