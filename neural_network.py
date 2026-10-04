import numpy as np

def initialize_parameters (n_x, n_h):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(1, n_h) * 0.01
    b2 = np.zeros((1, 1))
    
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)
    
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    
    return Z1, A1, Z2, A2

def compute_cost(A2, Y, m):
    cost = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return np.squeeze(cost)

def backward_propagation(X, Y, Z1, A1, Z2, A2, W2, m):
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0) 
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    return W1, b1, W2, b2

def train_model(X, Y, n_h, epochs=1000, learning_rate=0.01):
    n_x = X.shape[0]
    m = X.shape[1]
    W1, b1, W2, b2 = initialize_parameters(n_x, n_h)
    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, Y, m)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W2, m)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 100 == 0:
            print(f"Epoch {i}: Cost = {cost:.6f}")           
    return W1, b1, W2, b2

if __name__ == "__main__":
    np.random.seed(42)
    X_dummy = np.random.randn(3, 100)
    Y_dummy = (np.dot([2, -1, 0.5], X_dummy) + np.random.randn(100) > 0).astype(float)
    Y_dummy = Y_dummy.reshape(1, 100)
    
    print("Starting training...")
    trained_W1, trained_b1, trained_W2, trained_b2 = train_model(
        X_dummy, Y_dummy, n_h=4, epochs=1000, learning_rate=0.05
    )
    print("Training complete!")