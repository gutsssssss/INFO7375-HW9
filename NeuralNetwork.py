import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation_function, lambda_val):
        self.layers = layers
        self.activation_function = activation_function
        self.weights = []
        self.biases = []
        self.lambda_val = lambda_val
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]))
            self.biases.append(np.random.randn(layers[i+1]))

    def forward_propagation(self, X):
        self.a = [X]
        self.z = []
        for i in range(len(self.layers) - 1):
            self.z.append(np.dot(self.a[-1], self.weights[i]) + self.biases[i])
            self.a.append(self.activation_function(self.z[-1]))
        return self.a[-1]

    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        loss = -1/m * np.sum(y * np.log(y_hat))
        loss = self.L2_Normalization(loss)
        return loss

    def back_propagation(self, X, y):
        m = X.shape[0]
        self.dz = self.a[-1] - y
        self.dw = 1/m * np.dot(self.a[-2].T, self.dz)
        self.db = 1/m * np.sum(self.dz, axis=0, keepdims=True)
        self.weights[-1] -= self.learning_rate * self.dw
        self.biases[-1] -= self.learning_rate * self.db
        for i in range(len(self.layers)-3, -1, -1):
            self.dz = np.dot(self.dz, self.weights[i+1].T) * (1 - np.power(self.a[i+1], 2))
            self.dw = 1/m * np.dot(self.a[i].T, self.dz)
            self.db = 1/m * np.sum(self.dz, axis=0, keepdims=True)
            self.weights[i] -= self.learning_rate * self.dw
            self.biases[i] -= self.learning_rate * self.db


    def L2_Normalization(self, loss):
        return loss + self.lambda_val * np.sum(self.weights**2)

    def dropout(self, input, p, mode='train'):
        keep_prob = 1 - p
        if mode == 'train':
            input *= np.random.binomial(1, keep_prob, size=input.shape) / keep_prob
        return input
