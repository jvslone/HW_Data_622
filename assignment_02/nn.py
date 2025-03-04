import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=";")

# Define input (X) and output (y)
X = data.iloc[:, :-1].values  # Features (all except last column)
y = data.iloc[:, -1].values   # Target (wine quality score)

# Convert target to classification (quality 3-9 -> categories)
# One-hot encode target variable
hot_encoder = OneHotEncoder(sparse_output=False) # Create the one-hot encoder instance for transforming target
y = hot_encoder.fit_transform(y.reshape(-1, 1)) # Perform transformation

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #20% test and 80% train split

# Standardize features
sscaler = StandardScaler() # Initialize the standard scaler
X_train = sscaler.fit_transform(X_train) # Fit the scaler on the training data and transform it
X_test = sscaler.transform(X_test) # Transform the test data

# Define network architecture
input_size = X_train.shape[1] # Number of features
hidden_size = 16  # Hidden layer neurons
output_size = y_train.shape[1]    # Number of classes

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)*0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)*0.01
b2 = np.zeros((1, output_size))

# Activation functions
def relu(Z):
    """Docstring WIP"""
    return np.maximum(0, Z)

def softmax(Z):
    """Docstring WIP"""
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ/np.sum(expZ, axis=1, keepdims=True)

# Forward propagation
def forward_propagation(X):
    """Docstring WIP"""
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Compute loss (categorical cross-entropy)
def compute_loss(y_true, y_pred):
    """Docstring WIP"""
    m = y_true.shape[0]
    epsilon = 1e-8
    loss = -np.sum(y_true*np.log(y_pred + epsilon))/m
    return loss

# Backpropagation
def backward_propagation(X, y, Z1, A1, A2):
    """ Compute the backward pass of the network.
    """
    pass  # TODO: Implement backpropagation

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Perform forward propagation


    # Compute loss


    # Perform backward propagation


    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluate on test set
_, _, _, A2_test = forward_propagation(X_test)
y_pred = np.argmax(A2_test, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")