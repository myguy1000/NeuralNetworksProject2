import numpy as np
class LinearSVM:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_param = 0.01
        self.weights = None
        self.bias = 0

    def forward(self, X):
        # Calculate the margin score for classification
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
                    self.weights -= self.learning_rate * (2 * self.regularization_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (
                                2 * self.regularization_param * self.weights - np.dot(X[i], y_[i]))
                    self.bias -= self.learning_rate * y_[i]

    def predict(self, X):
        return np.sign(self.forward(X))
