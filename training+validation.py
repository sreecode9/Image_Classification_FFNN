import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train_full = X_train_full.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

y_train_full_oh = one_hot_encode(y_train_full)
y_test_oh = one_hot_encode(y_test)

val_size = int(0.1 * len(X_train_full))
X_val = X_train_full[:val_size]
y_val = y_train_full_oh[:val_size]

X_train = X_train_full[val_size:]
y_train = y_train_full_oh[val_size:]

print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test_oh.shape)
