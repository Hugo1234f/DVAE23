import pandas as pd
import numpy as np

# ----------------------------
# Load and preprocess data
# ----------------------------
data = pd.read_csv('./points.csv', index_col=False)

X = data[['Feature_1', 'Feature_2']].to_numpy()
Y = data['Label'].to_numpy()
Y[Y == 0] = -1  # convert 0 -> -1 for logistic regression

# ----------------------------
# Feature scaling (standardization)
# ----------------------------
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# ----------------------------
# Logistic regression functions
# ----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, w, reg=0.0):
    z = y * (X @ w)
    s = sigmoid(z)
    grad = -X.T @ ((1 - s) * y) / len(y)
    grad += reg * w
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
# Training function
# ----------------------------
def train_lr(X_train, y_train, lr=0.05, reg=0.01, iters=5000):
    w = np.zeros(X_train.shape[1])
    for i in range(iters):
        grad = compute_gradient(X_train, y_train, w, reg)
        w -= lr * grad
    return w

# ----------------------------
# k-Fold cross-validation
# ----------------------------
def k_fold_cv(X, Y, k=5, lr=0.05, reg=0.01, iters=5000):
    N = len(Y)
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = N // k

    E_in_list = []
    E_out_list = []
    Loss_train_list = []
    Loss_val_list = []

    for fold in range(k):
        val_idx = indices[fold*fold_size : (fold+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        w = train_lr(X_train, y_train, lr=lr, reg=reg, iters=iters)

        y_train_pred = predict(X_train, w)
        y_val_pred = predict(X_val, w)

        E_in_list.append(err_rate(y_train, y_train_pred))
        E_out_list.append(err_rate(Y_val, y_val_pred))
        Loss_train_list.append(log_loss(X_train, y_train, w, reg))
        Loss_val_list.append(log_loss(X_val, Y_val, w, reg))

        print(f"Fold {fold+1}: E_in={E_in_list[-1]:.4f}, E_out={E_out_list[-1]:.4f}, "
              f"Train Loss={Loss_train_list[-1]:.4f}, Val Loss={Loss_val_list[-1]:.4f}")

    print("\nAverage across folds:")
    print(f"Mean E_in={np.mean(E_in_list):.4f}, Mean E_out={np.mean(E_out_list):.4f}")
    print(f"Mean Train Loss={np.mean(Loss_train_list):.4f}, Mean Val Loss={np.mean(Loss_val_list):.4f}")

# ----------------------------
# Example usage
# ----------------------------
learning_rates = [0.01, 0.05, 0.1]
reg_strength = 0.01
k_fold = 5

for lr in learning_rates:
    print(f"\n=== Learning rate: {lr} ===")
    k_fold_cv(X, Y, k=k_fold, lr=lr, reg=reg_strength, iters=5000)
