import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from feedforward_nn import NeuralNetwork

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

y_train = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model = NeuralNetwork([784, 128, 64, 10], learning_rate=0.01)

model.fit(X_train, y_train, epochs=10, batch_size=64)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")
