import numpy as np 

class FeedForwardNeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, optimizer_cls=None):
        self.layers = layers
        self.lr = learning_rate
        self.optimizer = optimizer_cls(learning_rate=learning_rate)
        self.params = self.initialize_weights()

    def initialize_weights(self):
        params = []
        for i in range(len(self.layers) - 1):
            W = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2. / self.layers[i])
            b = np.zeros((1, self.layers[i+1]))
            params.append([W, b])
        return params

    def forward(self, X):
        activations = [X]
        z_list = []

        for W, b in self.params[:-1]:
            z = activations[-1] @ W + b
            z_list.append(z)
            a = np.maximum(0, z)
            activations.append(a)

        W, b = self.params[-1]
        z = activations[-1] @ W + b
        z_list.append(z)
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        activations.append(a)

        return activations, z_list

    def compute_loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def backward(self, activations, z_list, y_true):
        grads = []
        delta = activations[-1] - y_true
        for i in reversed(range(len(self.params))):
            a_prev = activations[i]
            dW = a_prev.T @ delta / len(y_true)
            db = np.sum(delta, axis=0, keepdims=True) / len(y_true)
            grads.insert(0, [dW, db])
            if i != 0:
                delta = (delta @ self.params[i][0].T) * (z_list[i-1] > 0)
        return grads

    def train(self, X, y, epochs=10, batch_size=64):
        losses, accuracies = [], []
        for epoch in range(epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                activations, z_list = self.forward(X_batch)
                grads = self.backward(activations, z_list, y_batch)
                for idx, (param, grad) in enumerate(zip(self.params, grads)):
                    self.optimizer.update(param, grad, idx)
            activations, _ = self.forward(X)
            y_pred = activations[-1]
            acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            accuracies.append(acc)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")
        return losses, accuracies

    def predict(self, X):
        a, _ = self.forward(X)
        return a[-1]
