# Neural Network from Scratch
This project implements a fully connected neural network for handwritten digit classification from scratch using NumPy, without relying on deep learning frameworks such as PyTorch or TensorFlow.

The goal of the project was to understand the mathematical foundations of neural networks by implementing forward propagation, backpropagation, and gradient descent manually.

## Dataset
The model is trained on the MNIST handwritten digit dataset, which contains:

* 60,000 training images

* 10,000 test images

* Image size: 28 × 28 grayscale pixels

* 10 output classes (digits 0–9)

* Input pixels are normalized to the range [0,1] to prevent extremely large gradients. Keep in mind that test input data must also be normalized to this range.

## Model Architecture

I stuck to a two-layer fully connected neural network: Input (784) -> Linear Layer (784 -> 256) -> ReLU Activtion -> Linear Layer (256 -> 10) -> Softmax for classification.

## Results
Final performance:
* Training accuracy: ~93.09%
* Test accuracy: ~93.39%

## Training Curves
![plots/accvsepoch.png](Accuracy vs. Epoch)
![plots/lossvsepoch.png](Loss vs. Epoch)

## Project Goals
This project was built to gain a deeper understanding of:
* Gradient-based optimization
* Neural network backpropagation
* Numerical stability in machine learning
* Vectorized linear algebra implementations
* Numpy operations, in general
