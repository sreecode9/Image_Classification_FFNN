import wandb
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from updatedfeedforward_nn import FeedForwardNeuralNetwork
from sklearn.model_selection import train_test_split

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

def train(config):
    wandb.init(config)  # start a new run
    config = wandb.config

    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)

    # Split into training and validation
    X_train, X_val, y_train_oh, y_val_oh = train_test_split(
        X_train, y_train_oh, test_size=0.1, random_state=42
    )

    # Set architecture
    hidden_layer_sizes = [config.hidden_size] * config.num_hidden_layers
    layers = [784] + hidden_layer_sizes + [10]

    # Create model
    model = FeedForwardNeuralNetwork(
        layers=layers,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        weight_init=config.weight_init,
        activation=config.activation
    )

    # Train
    losses, accuracies = model.train(
        X_train, y_train_oh,
        X_val, y_val_oh,
        epochs=config.epochs,
        batch_size=config.batch_size,
        use_wandb=True
    )

    # Evaluate on test data
    y_pred = model.predict(X_test)
    test_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test_oh, axis=1))
    wandb.log({"test_accuracy": test_accuracy})
    wandb.log({"val_accuracy": val_accuracy})

