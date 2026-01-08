# Image Classification FFNN

A comprehensive feedforward neural network implementation for image classification using the Fashion MNIST dataset.

## Project Description

This project implements a from-scratch feedforward neural network (FFNN) for classifying fashion items from the Fashion MNIST dataset. It includes custom implementations of various optimizers, training utilities, and hyperparameter tuning capabilities using Weights & Biases (W&B) for experiment tracking.

## Features

- **Custom Neural Network Implementation**: Fully implemented feedforward neural network without relying on high-level frameworks
- **Multiple Optimizers**: Support for various optimization algorithms
- **Hyperparameter Tuning**: Integrated with Weights & Biases for sweep configuration and tracking
- **Fashion MNIST Classification**: Train and evaluate on the Fashion MNIST dataset with 10 clothing classes
- **Visualization Tools**: Plotting utilities for analyzing model performance and dataset samples

## Project Structure

```
.
├── feedforward_nn.py          # Core neural network implementation
├── updatedfeedforward_nn.py   # Enhanced version of the neural network
├── train_nn.py                # Training pipeline
├── optimizers.py              # Custom optimizer implementations
├── main.py                    # Main entry point
├── q1_plot_fashion_mnist.py   # Visualization and plotting utilities
├── training+validation.py     # Training and validation logic
├── sweep_training.py          # W&B hyperparameter sweep training
├── sweep_configuration.py     # Sweep configuration settings
├── tempCodeRunnerFile.py      # Temporary test file
├── wandb/                     # Weights & Biases sweep configurations
└── __pycache__/               # Python cache files
```

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib
- scikit-learn (for Fashion MNIST data loading)
- wandb (optional, for experiment tracking)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sreecode9/Image_Classification_FFNN.git
cd Image_Classification_FFNN
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
```

3. Install dependencies:
```bash
pip install numpy matplotlib scikit-learn wandb
```

## Usage

### Basic Training

To train the neural network with default parameters:

```bash
python main.py
```

### Training with W&B Integration

To train with experiment tracking:

```bash
python train_nn.py
```

### Hyperparameter Sweeps

To run hyperparameter sweeps using Weights & Biases:

```bash
python sweep_training.py
```

Configure sweep parameters in `sweep_configuration.py` before running.

### Visualization

To plot Fashion MNIST samples and model performance:

```bash
python q1_plot_fashion_mnist.py
```

## Configuration

### Sweep Configuration

Edit `sweep_configuration.py` to customize:
- Learning rates
- Batch sizes
- Number of hidden layers and neurons
- Activation functions
- Optimization algorithms
- Number of epochs

### Network Architecture

Modify `feedforward_nn.py` or `updatedfeedforward_nn.py` to:
- Change network layers
- Adjust activation functions
- Modify initialization strategies

## Dataset

The project uses the **Fashion MNIST** dataset, which contains:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Optimizers

The `optimizers.py` module includes implementations of various optimization algorithms:
- Stochastic Gradient Descent (SGD)
- Momentum
- Adam
- And more

## Experiment Tracking

This project integrates with **Weights & Biases** for:
- Logging training metrics
- Tracking hyperparameters
- Visualizing model performance
- Running automated hyperparameter sweeps

To enable W&B tracking, log in to your account:
```bash
wandb login
```

## Results

The trained model achieves competitive accuracy on the Fashion MNIST test set. See W&B project reports for detailed results and comparisons across different hyperparameter configurations.

## License

This project is provided as-is for educational purposes.

## Contact

For questions or suggestions, please open an issue on the GitHub repository.
