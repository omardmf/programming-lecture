import tensorflow as tf
import numpy as np
import time

# 1. Load and normalize MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),        # Input layer
    tf.keras.layers.Dense(64, activation='relu'),         # Hidden layer
    tf.keras.layers.Dense(10, activation='softmax')       # Output layer
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
print("Training started...")
start_time = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=2)
training_time = time.time() - start_time
print(f"\n✅ Total training time: {training_time:.2f} seconds")

# 5. Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n📊 Test accuracy: {test_acc:.4f}")

# 6. Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to .tflite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("\n📦 Model successfully converted and saved as model.tflite")

# 7. Inference using TensorFlow Lite model
print("\n🔍 Running inference using TFLite model...")

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Use a real test image
test_image = x_test[0].reshape(1, 28, 28).astype(np.float32)
true_label = y_test[0]

# Inference
start_infer = time.time()
interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
inference_time = time.time() - start_infer

# Get prediction
predicted_class = np.argmax(output)

# Display result
print(f"👁️ True label: {true_label}")
print(f"🤖 Predicted class: {predicted_class}")
print(f"⏱️ Inference time: {inference_time:.6f} seconds")