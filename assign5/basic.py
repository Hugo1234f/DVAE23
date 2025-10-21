import pandas as pd
import numpy as np

# Load data
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

# Feature normalization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# Logistic regression functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, w):
    z = y * (X @ w)
    s = sigmoid(z)
    return -X.T @ ((1 - s) * y) / len(y)

def predict(X, w):
    pred = np.sign(X @ w)
    pred[pred == 0] = 1
    return pred

def err_rate(y, y_hat):
    return np.mean(y != y_hat)

def test_lr(lr, X_train, y_train, X_val, Y_val, iters=5000):
    w = np.zeros(X_train.shape[1])
    for i in range(iters):
        grad = compute_gradient(X_train, y_train, w)
        w -= lr * grad
    y_train_pred = predict(X_train, w)
    y_val_pred = predict(X_val, w)
    return err_rate(y_train, y_train_pred), err_rate(Y_val, y_val_pred)

# Test multiple learning rates
learning_rates = [0.01, 0.02, 0.05, 0.07, 0.1, 10, 100, 500, 1000]
for lr in learning_rates:
    E_in, E_out = test_lr(lr, X_train, y_train, X_val, Y_val)
    print(f"Learning rate {lr}: E_in={E_in:.4f}, E_out={E_out:.4f}")
