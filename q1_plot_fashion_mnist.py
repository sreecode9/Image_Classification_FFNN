import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

label_to_class_name = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

(X_train, y_train), _ = fashion_mnist.load_data()

class_examples = {}

for label in np.unique(y_train):
    indices = np.where(y_train == label)[0]
    if len(indices) > 0:
        class_examples[label] = X_train[indices[0]]

plt.figure(figsize=(12, 6))

for i, label in enumerate(sorted(class_examples.keys())):
    plt.subplot(2, 5, i + 1)
    plt.imshow(class_examples[label], cmap='gray')
    plt.title(label_to_class_name[label])
    plt.axis('off')

plt.tight_layout()
plt.show()



