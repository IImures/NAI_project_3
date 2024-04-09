import numpy as np


class Perceptron:
    def __init__(self, num_inputs, learning_rate=1):
        self.weights = np.random.rand(num_inputs)
        self.learning_rate = learning_rate

    def weighted_sum(self, inputs):
        return np.dot(inputs, self.weights)

    def activation(self, weighted_sum):
        return 1 / (1 + np.exp(-self.learning_rate * weighted_sum))

    def correction(self, inputs, prediction, target):
        error = (target - prediction)
        self.weights = self.weights + self.learning_rate * error * inputs

    def predict(self, inputs):
        return self.activation(self.weighted_sum(inputs))

    def __str__(self):
        return (f"\n"
                f"Weights: {self.weights}"
                f"\nBias: {self.bias} "
                f"\nClass: {self.lclass} "
                f"\nLearning rate: {self.learning_rate}")