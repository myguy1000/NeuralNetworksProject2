import numpy as np

class WidrowHoff:
    def __init__(self, input_size, learning_rate, num_epochs, min, max):
        self.weights = np.random.uniform(min, max, input_size)
        self.bias = np.random.uniform(min, max)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


    def forward(self, inputs):
        return np.dot(inputs,self.weights) + self.bias

    def fit(self, inputs, output):
        # Train the current_perceptron using the current_perceptron learning algorithm
        z = range(self.num_epochs)
        for i in z:
            for x in inputs:
                prediction = self.forward(x)

                errors = output - prediction
                self.weights += self.learning_rate * np.dot(inputs.T, errors) /len(output)
                self.bias += self.learning_rate * np.mean(errors)
