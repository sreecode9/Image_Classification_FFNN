import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1. - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def forward(self, X):
        activations = [X]
        z_list = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_list.append(z)
            a = self.relu(z)
            activations.append(a)

        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_list.append(z)
        a = self.softmax(z)
        activations.append(a)

        return activations, z_list

    def backward(self, X, y_true, activations, z_list):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Output layer
        delta = activations[-1] - y_true
        grads_w[-1] = activations[-2].T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = (delta @ self.weights[i+1].T) * self.relu_derivative(z_list[i])
            grads_w[i] = activations[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

        return grads_w, grads_b

    def update_params(self, grads_w, grads_b, batch_size):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i] / batch_size
            self.biases[i] -= self.learning_rate * grads_b[i] / batch_size

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def fit(self, X_train, y_train, epochs=10, batch_size=64):
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                activations, z_list = self.forward(X_batch)
                grads_w, grads_b = self.backward(X_batch, y_batch, activations, z_list)
                self.update_params(grads_w, grads_b, batch_size)

            # Compute loss and accuracy
            preds = self.forward(X_train)[0][-1]
            loss = self.cross_entropy_loss(preds, y_train)
            acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_train, axis=1))
            print(f"Epoch {epoch+1} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
