import pandas as pd
import numpy as np

# ----------------------------
# Load and preprocess data
# ----------------------------
data = pd.read_csv('./points.csv', index_col=False)

X = data[['Feature_1', 'Feature_2']].to_numpy()
Y = data['Label'].to_numpy()
Y[Y == 0] = -1  # convert 0 -> -1 for logistic regression

# Shuffle data
np.random.seed(42)
perm = np.random.permutation(len(Y))
X = X[perm]
Y = Y[perm]

# Train/validation split (90/10)
N = X.shape[0]
val_idx = np.arange(0, N, 10)
train_idx = np.setdiff1d(np.arange(N), val_idx)

X_train, y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

# Feature scaling (standardization)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# ----------------------------
# Logistic regression functions
# ----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, w, reg=0.0):
    z = y * (X @ w)
    s = sigmoid(z)
    grad = -X.T @ ((1 - s) * y) / len(y)
    grad += reg * w   # L2 regularization
    return grad

def log_loss(X, y, w, reg=0.0):
    z = y * (X @ w)
    return np.mean(np.log(1 + np.exp(-z))) + 0.5 * reg * np.sum(w**2)

def predict(X, w):
    pred = np.sign(X @ w)
    pred[pred == 0] = 1
    return pred

def err_rate(y, y_hat):
    return np.mean(y != y_hat)

# ----------------------------
# Training with tracking
# ----------------------------
def test_lr(lr, X_train, y_train, X_val, Y_val, reg=0.01, iters=5000, print_loss=True):
    w = np.zeros(X_train.shape[1])
    for i in range(iters):
        grad = compute_gradient(X_train, y_train, w, reg)
        w -= lr * grad
        if print_loss and i % 500 == 0:
            print(f"Iter {i}: Train loss = {log_loss(X_train, y_train, w, reg):.4f}, "
                  f"Val loss = {log_loss(X_val, Y_val, w, reg):.4f}")
    y_train_pred = predict(X_train, w)
    y_val_pred = predict(X_val, w)
    return err_rate(y_train, y_train_pred), err_rate(Y_val, y_val_pred)

# ----------------------------
# Test multiple learning rates
# ----------------------------
learning_rates = [0.01, 0.02, 0.05, 0.1, 1.0]
reg_strength = 0.01

#print('Normal:')

#ones = data.loc[data['Label'] == 1]
#zeros = data.loc[data['Label'] == 0]

w = np.zeros(X_train.shape[1])

for lr in learning_rates:
    print(f"\nLearning rate: {lr}")
    E_in, E_out = test_lr(lr, X_train, y_train, X_val, Y_val, reg=reg_strength)
    print(f"Final E_in = {E_in:.4f}, E_out = {E_out:.4f}")

# Compute model outputs for all data