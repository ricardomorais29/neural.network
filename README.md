# neural.network
A learning project built to understand the underlying mathematics of machine learning algorithms, handling matrix multiplication, activation functions, and backpropagation calculus entirely from scratch using Python and NumPy.

# Pure NumPy Neural Network

This repository contains a feed-forward neural network built completely from scratch using Python and NumPy. 

It is purely a learning project, designed to strip away the abstractions of high-level frameworks (like TensorFlow or PyTorch) to deeply understand the underlying mathematics of machine learning algorithms. By building this, I focused on manually handling matrix multiplications, implementing activation functions, and writing the calculus for backpropagation.

## 🚀 Technical Overview

The code (`neural_network.py`) implements a complete single-hidden-layer network with the following mathematical components:
* **Forward Pass:** Matrix dot products to pass data through the network.
* **Activation Functions:** ReLU (`np.maximum(0, Z)`) for the hidden layer and Sigmoid for the output layer to handle binary classification.
* **Loss Function:** Binary Cross-Entropy (Log Loss) to compute the cost.
* **Backpropagation:** Explicit calculation of gradients (dZ, dW, db) using matrix calculus and the derivative of the ReLU function.
* **Optimization:** Standard gradient descent to update weights and biases based on the learning rate.

## 🛠️ Tech Stack
* **Python 3.x**
* **NumPy**

## ⚙️ Getting Started

The script is self-contained and includes a `__main__` block that automatically generates a synthetic dataset and trains the model, making it instantly runnable.

### Prerequisites
Ensure you have Python and NumPy installed:
```bash
pip install numpy
