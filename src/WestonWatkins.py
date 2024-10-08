import numpy as np

class WestonWatkinsSVM:
    def __init__(self, input_size, num_classes, learning_rate, num_epochs, min_value, max_value, regularization):
        """
        Initialize the Weston-Watkins SVM model.

        Parameters:
        - input_size (int): Number of input features.
        - num_classes (int): Number of classes for classification.
        - learning_rate (float): Learning rate for weight updates.
        - num_epochs (int): Number of epochs for training.
        - min_value (float): Minimum value for initializing weights and biases.
        - max_value (float): Maximum value for initializing weights and biases.
        - regularization (float): Regularization parameter (lambda) for weight decay.
        """

        self.weights = np.random.uniform(min_value, max_value, (num_classes, input_size))
        self.biases = np.random.uniform(min_value, max_value, num_classes)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.regularization = regularization
        self.num_classes = num_classes

    def forward(self, input_vector):
        """
        Compute the scores for each class for a single input vector.

        Parameters:
        - input_vector (ndarray): Input feature vector of shape (input_size,).

        Returns:
        - scores (ndarray): Scores for each class, shape (num_classes,).
        """

        scores = np.dot(self.weights, input_vector) + self.biases  # Shape: (num_classes,)
        return scores

    def fit(self, inputs, outputs):
        """
        Train the Weston-Watkins SVM model using the provided training data.

        Parameters:
        - inputs (ndarray): Training input data, shape (num_samples, input_size).
        - outputs (ndarray): Training labels, shape (num_samples,).
        """

        num_samples, num_features = inputs.shape

        # Training loop over epochs
        for epoch in range(self.num_epochs):
            for i in range(num_samples):
                x_i = inputs[i]              # Input vector for the i-th sample
                true_class = outputs[i]      # True class label for the i-th sample
                scores = self.forward(x_i)   # Compute scores for all classes

                # Compute margins for all classes (Hinge loss)
                margins = scores - scores[true_class] + 1
                margins[true_class] = 0

                # Find classes where the margin is violated (loss is incurred)
                loss_indices = np.where(margins > 0)[0]

                # Initialize gradients for weights and biases
                dW = np.zeros_like(self.weights)
                dB = np.zeros_like(self.biases)

                # Accumulate gradients for classes where margin is violated
                for j in loss_indices:
                    dW[j] += x_i          # Gradient for incorrect classes
                    dB[j] += 1
                    dW[true_class] -= x_i  # Gradient for the correct class
                    dB[true_class] -= 1

                # Add regularization term to the gradient
                dW += self.regularization * self.weights

                # Update weights and biases using gradient descent
                self.weights -= self.learning_rate * dW
                self.biases -= self.learning_rate * dB

    def predict(self, inputs):
        """
        Predict the class labels for given input data.

        Parameters:
        - inputs (ndarray): Input data, shape (num_samples, input_size) or (input_size,).

        Returns:
        - predictions (ndarray or int): Predicted class labels.
        """

        if inputs.ndim == 1:
            # Single sample prediction
            scores = np.dot(self.weights, inputs) + self.biases
            return np.argmax(scores)
        else:
            # Multiple samples prediction
            scores = np.dot(inputs, self.weights.T) + self.biases
            return np.argmax(scores, axis=1)
