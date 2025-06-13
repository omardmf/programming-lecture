# MNIST Model: TensorFlow vs PyTorch + Edge Deployment

This project explores the training, evaluation, export, and deployment of a simple neural network for digit classification on the MNIST dataset using both **TensorFlow** and **PyTorch**. The exported models are deployed and tested on a **Raspberry Pi 5** to measure inference performance and runtime behavior on edge devices.

## 🔧 Features

- Train and evaluate models using TensorFlow and PyTorch
- Compare training speed and test accuracy
- Export models to TensorFlow Lite (.tflite) and ONNX (.onnx) formats
- Deploy and run inference on Raspberry Pi (Edge AI)
- Log and compare predictions, inference time, and accuracy

---

## 🧠 Model Architecture

- **Input**: 28×28 grayscale images (flattened to 784 units)
- **Hidden Layer**: Dense layer with 64 ReLU units
- **Output Layer**: Dense layer with 10 softmax/logits units

---
