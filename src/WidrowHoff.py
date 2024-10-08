import numpy as np

class WidrowHoff:
    def __init__(self, input_size, learning_rate, num_epochs, min, max):
        self.weights = np.random.uniform(min, max, input_size)
        self.bias = np.random.uniform(min, max)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return np.where(weighted_sum >= 0, 1, 0)

    def fit(self, inputs, output):
        inputs = inputs.astype(np.float32)  # Ensure inputs are float
        output = output.astype(np.float32)  # Ensure output is float
        z = range(self.num_epochs)

        for i in z:
            for x in inputs:
                prediction = self.forward(x)

                errors = output - prediction

                self.weights += self.learning_rate * np.dot(inputs.T, errors)
                self.bias += self.learning_rate * np.mean(errors)

                # Clip weights to prevent overflow
                self.weights = np.clip(self.weights, -1e10, 1e10)
                self.bias = np.clip(self.bias, -1e10, 1e10)