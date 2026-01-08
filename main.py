from updatedfeedforward_nn import FeedForwardNeuralNetwork
from optimizers import Adam, Nadam, SGD, Momentum, NAG, RMSprop
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
y_train_oh = one_hot_encode(y_train)
y_test_oh = one_hot_encode(y_test)

model = FeedForwardNeuralNetwork(
    layers=[784, 128, 64, 10],
    learning_rate=0.001,
    optimizer_cls=Adam  
)

losses, accuracies = model.train(X_train, y_train_oh, epochs=10, batch_size=64)

plt.plot(losses, label='Loss')
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.grid(True)
plt.title("Training Loss and Accuracy")
plt.show()

preds = model.predict(X_test)
test_acc = np.mean(np.argmax(preds, axis=1) == y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
