import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction function
def predict(w1, w2, x1, x2, b):
    z = w1 * x1 + w2 * x2 + b
    y_hat = sigmoid(z)
    return y_hat

# Binary cross-entropy loss function
def loss(y, y_hat):
    return -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

# Gradient calculation
def gradient(y, y_hat, x1, x2):
    db = y_hat - y
    dw1 = x1 * (y_hat - y)
    dw2 = x2 * (y_hat - y)
    return db, dw1, dw2

# Parameter update
def update(w1, w2, eta, dw1, dw2, b, db):
    w1 = w1 - eta * dw1
    w2 = w2 - eta * dw2
    b = b - eta * db
    return w1, w2, b

import numpy as np

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction function
def predict(w1, w2, x1, x2, b):
    z = w1 * x1 + w2 * x2 + b
    y_hat = sigmoid(z)
    return y_hat

# Binary cross-entropy loss function
def loss(y, y_hat):
    return -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

# Gradient calculation
def gradient(y, y_hat, x1, x2):
    db = y_hat - y
    dw1 = x1 * (y_hat - y)
    dw2 = x2 * (y_hat - y)
    return db, dw1, dw2

# Parameter update
def update(w1, w2, eta, dw1, dw2, b, db):
    w1 = w1 - eta * dw1
    w2 = w2 - eta * dw2
    b = b - eta * db
    return w1, w2, b


# Initialize weights and bias
w1 = 0.5  # Initial weight for feature x1
w2 = -0.5  # Initial weight for feature x2
b = 0.1  # Initial bias

# Learning rate (hyperparameter)
eta = 0.1

# Training data (example inputs)
x1_train = [0, 0, 1, 1]
x2_train = [0, 1, 0, 1]

# Ground truth labels (example outputs)
y_train = [0, 1, 1, 0]

loss_history = []
epoch_numbers = []

# Training loop
epochs = 1000  # Number of training iterations

for epoch in range(epochs):
    total_loss = 0  # Initialize total loss for this epoch

    for i in range(len(x1_train)):
        # Predict using the current model
        y_hat = predict(w1, w2, x1_train[i], x2_train[i], b)

        # Calculate the loss
        current_loss = loss(y_train[i], y_hat)
        total_loss += current_loss

        # Compute gradients
        db, dw1, dw2 = gradient(y_train[i], y_hat, x1_train[i], x2_train[i])

        # Update weights and bias using gradient descent
        w1, w2, b = update(w1, w2, eta, dw1, dw2, b, db)

    # Print the average loss for this epoch
    if (epoch + 1) % 100 == 0:
        average_loss = total_loss / len(x1_train)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}")

# After training, you can make predictions using the trained model
x1_test = [0.2, 0.3, 0.8, 0.9]
x2_test = [0.3, 0.4, 0.7, 0.8]

for i in range(len(x1_test)):
    y_pred = predict(w1, w2, x1_test[i], x2_test[i], b)
    print(f"Input: ({x1_test[i]}, {x2_test[i]}) => Predicted Output: {y_pred:.4f}")


# Plot the training loss over epochs
plt.plot(epoch_numbers, loss_history, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()