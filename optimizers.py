import numpy as np

class Optimizer:
    def update(self, params, grads, layer_idx):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads, layer_idx):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.v = {}

    def update(self, params, grads, layer_idx):
        if layer_idx not in self.v:
            self.v[layer_idx] = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.v[layer_idx][i] = self.beta * self.v[layer_idx][i] + (1 - self.beta) * grads[i]
            params[i] -= self.lr * self.v[layer_idx][i]


class NAG(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.v = {}

    def update(self, params, grads, layer_idx):
        if layer_idx not in self.v:
            self.v[layer_idx] = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            look_ahead = params[i] - self.beta * self.v[layer_idx][i]
            self.v[layer_idx][i] = self.beta * self.v[layer_idx][i] + self.lr * grads[i]
            params[i] -= self.v[layer_idx][i]


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s = {}

    def update(self, params, grads, layer_idx):
        if layer_idx not in self.s:
            self.s[layer_idx] = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.s[layer_idx][i] = self.beta * self.s[layer_idx][i] + (1 - self.beta) * grads[i]**2
            params[i] -= self.lr * grads[i] / (np.sqrt(self.s[layer_idx][i]) + self.epsilon)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, params, grads, layer_idx):
        if layer_idx not in self.m:
            self.m[layer_idx] = [np.zeros_like(p) for p in params]
            self.v[layer_idx] = [np.zeros_like(p) for p in params]
            self.t[layer_idx] = 0
        self.t[layer_idx] += 1
        for i in range(len(params)):
            self.m[layer_idx][i] = self.beta1 * self.m[layer_idx][i] + (1 - self.beta1) * grads[i]
            self.v[layer_idx][i] = self.beta2 * self.v[layer_idx][i] + (1 - self.beta2) * (grads[i]**2)
            m_hat = self.m[layer_idx][i] / (1 - self.beta1**self.t[layer_idx])
            v_hat = self.v[layer_idx][i] / (1 - self.beta2**self.t[layer_idx])
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class Nadam(Adam):
    def update(self, params, grads, layer_idx):
        if layer_idx not in self.m:
            self.m[layer_idx] = [np.zeros_like(p) for p in params]
            self.v[layer_idx] = [np.zeros_like(p) for p in params]
            self.t[layer_idx] = 0
        self.t[layer_idx] += 1
        for i in range(len(params)):
            self.m[layer_idx][i] = self.beta1 * self.m[layer_idx][i] + (1 - self.beta1) * grads[i]
            self.v[layer_idx][i] = self.beta2 * self.v[layer_idx][i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[layer_idx][i] / (1 - self.beta1 ** self.t[layer_idx])
            v_hat = self.v[layer_idx][i] / (1 - self.beta2 ** self.t[layer_idx])
            nesterov = self.beta1 * m_hat + ((1 - self.beta1) * grads[i]) / (1 - self.beta1 ** self.t[layer_idx])
            params[i] -= self.lr * nesterov / (np.sqrt(v_hat) + self.epsilon)
