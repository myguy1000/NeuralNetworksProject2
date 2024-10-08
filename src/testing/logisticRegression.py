import numpy as np

class LogisticRegression:
    def __init__(self, input_size, learning_rate, num_epochs, threshold, min, max):
        # Sets the weights and bias to be in the range of -1 to 1.
        # For our standard tests min = -1 and max = 1
        self.weights = np.random.uniform(min, max, input_size)
        self.bias = np.random.uniform(min, max)
        # set threshold to be the threshold value passed in from main
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        linear = np.dot(inputs, self.weights) + self.bias
        pred = self.sigmoid(linear)
        #print("LogReg pred: ", pred)
        #use threshold to determine Label
        if pred > self.threshold:
            return 1
        else:
            return 0

    # Training the model using gradient descent
    def fit(self, inputs, outputs):
        count = 0
        # Gradient descent
        for _ in range(self.num_epochs):
            #Linear Regression: w.T * X + b
            #print(inputs.shape)
            #print(self.weights.shape)
            #print("weights.shape: ", self.weights.shape)
            #reshaped_weights = self.weights.reshape(-1,1)
            #reshaped_weights = self.weights.reshape(1,-1)
            #print("reshaped_weights.shape: ", reshaped_weights.shape)
            linear_model = np.dot(inputs, self.weights) + self.bias
            #Use sigmoid to get Logistic regression
            print(f"activation at {count}: ", linear_model)
            count = count + 1
            pred = self.sigmoid(linear_model)

            #Compute the gradients
            weight_update = (1 / len(inputs)) * np.dot(inputs.T, (pred - outputs))
            bias_update = (1 /  len(inputs)) * np.sum(pred - outputs)

            #Update weights and bias
            self.weights -= self.learning_rate * weight_update
            self.bias -= self.learning_rate * bias_update
