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

## 🚀 Quick Start

### On PC

```bash


# Clone the repository
git clone https://github.com/yourusername/your-repository.git
cd your-repository

# Create environment and install requirements (example: TensorFlow)
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -r requirements_tf.txt
python tensorflow/train_tf.py

### On Raspberry Pi

```bash

# SSH into your Pi and activate environment
source ~/env1/bin/activate

# Run inference using TFLite or ONNX
python3 run_tflite_pi.py
python3 run_onnx_pi.py


📘 References
TensorFlow Lite Documentation

PyTorch Documentation

ONNX Runtime Docs

Raspberry Pi Official Site

📄 License
This project is released under the MIT License.

