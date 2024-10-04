import numpy as np

class WestonWatkinsSVM:
    def __init__(self, input_size, num_classes, min_value, max_value, learning_rate, regularization):
        # Initialize weights and biases within the given range
        self.weights = np.random.uniform(min_value, max_value, (num_classes, input_size))
        self.bias = np.random.uniform(min_value, max_value, num_classes)
        self.learning_rate = learning_rate
        self.regularization = regularization

    def forward(self, input_vector):
        # Compute the score for each class for a single input vector
        return np.dot(self.weights, input_vector) + self.bias

    def fit(self, inputs, outputs, num_epochs):
        num_samples, num_features = inputs.shape

        # Training loop
        for epoch in range(num_epochs):
            for i in range(num_samples):
                x_i = inputs[i]
                true_class = outputs[i]
                scores = self.forward(x_i)

                # Compute margins for all classes
                margins = scores - scores[true_class] + 1  # Delta = 1
                margins[true_class] = 0  # Ignore the true class

                # Find classes where the margin is violated
                loss_indices = np.where(margins > 0)[0]

                # Initialize gradients
                dw = np.zeros_like(self.weights)
                db = np.zeros_like(self.bias)

                # Accumulate gradients for classes where margin is violated
                for r in loss_indices:
                    dw[r] += x_i
                    db[r] += 1
                    dw[true_class] -= x_i
                    db[true_class] -= 1

                # Add regularization term to the gradient
                dw += self.regularization * self.weights

                # Update weights and biases
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, inputs):
        # Compute scores for all inputs
        scores = np.dot(inputs, self.weights.T) + self.bias  # Shape: (num_samples, num_classes)
        # Return the class with the highest score for each input
        return np.argmax(scores, axis=1)

