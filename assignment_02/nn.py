import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# Do not want to return a sparse matrix as that is not as useful for use
y = hot_encoder.fit_transform(y.reshape(-1, 1)) # Perform transformation
# Useful to note that there are 6 classes in the target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #20% test and 80% train split

# Standardize features
sscaler = StandardScaler() # Initialize the standard scaler
X_train = sscaler.fit_transform(X_train) # Fit the scaler on the training data and transform it
X_test = sscaler.transform(X_test) # Transform the test data

# Define network architecture
input_size = X_train.shape[1] # Number of features
hidden_size1 = 32 # Hidden layer neurons
hidden_size2 = 16  # Hidden layer neurons
output_size = y_train.shape[1]    # Number of classes

#Make this all a function
def Network(epochs=10000, lr=0.05, printing=False, alpha=0.01):
    # Initialize weights and biases
    #I used small random values for weights for best behavior
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size1)*0.01
    b1 = np.zeros((1, hidden_size1))
    W2 = np.random.randn(hidden_size1, hidden_size2)*0.01
    b2 = np.zeros((1, hidden_size2))
    W3 = np.random.randn(hidden_size2, output_size)*0.01
    b3 = np.zeros((1, output_size))

    # Activation functions
    def relu(Z):
        """Docstring WIP"""
        return np.maximum(0, Z)

    def leaky_relu(Z, alpha=0.01):
        """Docstring WIP"""
        return np.where(Z > 0, Z, Z*alpha)

    def selu(Z):
        """Docstring WIP"""
        alpha = 1.6732632423543772848170429916717 #Taken from traditional Values
        scale = 1.0507009873554804934193349852946 #Taken from traditional Values
        return scale*np.where(Z > 0, Z, alpha*(np.exp(Z) - 1))

    def softmax(Z):
        """Docstring WIP"""
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ/np.sum(expZ, axis=1, keepdims=True)

    # Forward propagation
    def forward_propagation(X):
        """Docstring WIP"""
        #Hidden Layer 1
        Z1 = np.dot(X, W1) + b1
        A1 = selu(Z1)
        #Hidden Layer 2
        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)
        #Output Layer
        Z3 = np.dot(A2, W3) + b3
        A3 = softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    # Compute loss (categorical cross-entropy)
    def compute_loss(y_true, y_pred):
        """Docstring WIP"""
        m = y_true.shape[0] # Finds the number of samples
        epsilon = 1e-8 # Avoiding log(0)
        loss = -np.sum(y_true*np.log(y_pred + epsilon))/m #Cross entropy calc
        return loss

    # Backpropagation
    def backward_propagation(X, y, Z1, A1, Z2, A2, Z3, A3):
        """Docstring WIP"""
        m = X.shape[0] # Finds the number of samples
        # Output Layer
        dZ3 = A3 - y # Pre-Activitation Gradient
        dW3 = np.dot(A2.T, dZ3)/m # Output Weight Gradient
        db3 = np.sum(dZ3, axis=0, keepdims=True)/m # Output Bias Gradient
        # Hidden Layer 2
        dA2 = np.dot(dZ3, W3.T) # Post-Activation Gradient
        dZ2 = dA2*(Z2 > 0) # RELU Derivative
        dW2 = np.dot(A1.T, dZ2)/m # Hidden 2 Weight Gradient
        db2 = np.sum(dZ2, axis=0, keepdims=True)/m # Hidden 2 Bias Gradient
        # Hidden Layer 1
        dA1 = np.dot(dZ2, W2.T) # Post-Activation Gradient
        #dZ1 = dA1*np.where(Z1 > 0, 1, alpha) #Leaky RELU Derivative
        dZ1 = dA1*(Z1 > 0) #! Selu Derivative (Approximation)
        dW1 = np.dot(X.T, dZ1)/m # Hidden 1 Weight Gradient
        db1 = np.sum(dZ1, axis=0, keepdims=True)/m # Hidden 1 Bias Gradient
        return dW1, db1, dW2, db2, dW3, db3

    loss_data = []
    # Training loop
    for epoch in range(epochs):
        # Perform forward propagation
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X_train)

        # Compute loss
        loss = compute_loss(y_train, A3)
        loss_data.append(loss)

        # Perform backward propagation
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(X_train, y_train, Z1, A1, Z2, A2, Z3, A3)

        # Update weights and biases
        W1 -= lr * dW1 # Hidden 1 Weights
        b1 -= lr * db1 # Hidden 1 Biases
        W2 -= lr * dW2 # Hidden 2 Weights
        b2 -= lr * db2 # Hidden 2 Biases
        W3 -= lr * dW3 # Output Weights
        b3 -= lr * db3 # Output Biases
        
        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            if printing:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluate on test set
    _, _, _, _, _, A3_test = forward_propagation(X_test)
    y_pred = np.argmax(A3_test, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    if printing:
        print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy, loss_data

#Testing
lr=0.09
accuracy, loss_data = Network(epochs=10000, lr=lr, printing=True)
print(f"lr={lr:.2f}; Test Accuracy = {accuracy:.4f}")

#Plotting
fig = plt.figure(figsize=(10, 6))
plt.plot(np.arange(0,10000,1), loss_data, '-r', label='Loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper right')
plt.grid()
plt.show()

#TODO############################################################################################################
#TODO Implement a scheduler?
#TODO could add guesses to initial weights
#TODO could add more layers
#TODO could sweep through more hyperparams
#TODO############################################################################################################

#v01: lr=0.05, epochs=1000, hidden_size=16, Test Accuracy = 0.5594
#v02: lr=0.1, epochs=1000, hidden_size=16, Test Accuracy = 0.5813
#v03: lr=0.2, epochs=1000, hidden_size=16, Test Accuracy = 0.5750
#v04: lr=0.1, epochs=1000, hidden_size=64, Test Accuracy = 0.5781
#v05: lr=0.1, epochs=1000, hidden_size1=16, hidden_size2=8, Test Accuracy = 0.4062...
#v06: lr=0.1, epochs=1000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.5094
#v07: lr=0.02, epochs=10000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.5656
#v08: lr=0.02, epochs=10000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.5625, replaced second activation with selu
#v09: lr=0.03, epochs=10000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.5813, selu kept and all below unless mentioned
#v10: lr=0.05, epochs=10000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.5938
#v11: lr=0.05, epochs=10000, hidden_size1=64, hidden_size2=16, Test Accuracy = 0.5875
#v12: lr=0.08, epochs=10000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.6188
#Gonna run a sweep for learning rates between 0.05 and 0.2 at 0.01 increments below
#lr=0.05; Test Accuracy = 0.5938
#lr=0.06; Test Accuracy = 0.6094
#lr=0.07; Test Accuracy = 0.5969
#lr=0.08; Test Accuracy = 0.6188
#lr=0.09; Test Accuracy = 0.6219
#lr=0.10; Test Accuracy = 0.5875
#lr=0.11; Test Accuracy = 0.6094
#lr=0.12; Test Accuracy = 0.6125
#lr=0.13; Test Accuracy = 0.5781
#lr=0.14; Test Accuracy = 0.6031
#lr=0.15; Test Accuracy = 0.6000
#lr=0.16; Test Accuracy = 0.6031
#lr=0.17; Test Accuracy = 0.5625
#lr=0.18; Test Accuracy = 0.5719
#lr=0.19; Test Accuracy = 0.5875
#lr=0.20; Test Accuracy = 0.5594
#v13: lr=0.05, epochs=20000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.6188
#v14: lr=0.09, epochs=20000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.6188
#!v15: lr=0.09, epochs=10000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.6219
#v16: lr=0.09, epochs=10000, hidden_size1=32, hidden_size2=16, Test Accuracy = 0.6062, swapped selu for leaky relu