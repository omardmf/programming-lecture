import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import onnxruntime as ort
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and normalize MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MNISTModel().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
start_time = time.time()
for epoch in range(5):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/5], Loss: {epoch_loss:.4f}")
training_time = time.time() - start_time
print(f"✅ Total training time: {training_time:.2f} seconds")

# Load test set
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluation
model.eval()
correct = 0
total = 0

start_infer = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
inference_time = time.time() - start_infer

accuracy = correct / total
print(f"\n📊 Test accuracy: {accuracy * 100:.2f}%")
print(f"⏱️ Inference time (PyTorch): {inference_time:.4f} seconds")

# 6. Export the trained model to ONNX format
dummy_input = torch.randn(1, 784).to(device)
onnx_file = "model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_file,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print(f"\n🧠 Model successfully exported to {onnx_file}")

# 7. ONNX Inference
print("\n🔍 Running inference using ONNX Runtime...")
# Pick one sample from test set
sample_image = test_dataset[0][0].view(-1, 784).numpy().astype(np.float32)
true_label = test_dataset[0][1]

# Load ONNX model
session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])

start = time.time()
outputs = session.run(None, {"input": sample_image})
end = time.time()

predicted_class = np.argmax(outputs[0])
print(f"👁️ True label: {true_label}")
print(f"🤖 ONNX predicted class: {predicted_class}")
print(f"⏱️ ONNX inference time: {end - start:.6f} seconds")