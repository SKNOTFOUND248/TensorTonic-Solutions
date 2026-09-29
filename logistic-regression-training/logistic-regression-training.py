import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """

    N = X.shape[0]

    # Initialize parameters
    w = np.zeros(X.shape[1])
    b = 0.0

    # Gradient descent
    for _ in range(steps):

        # Forward pass
        z = X @ w + b
        p = _sigmoid(z)

        # Gradients
        dw = (X.T @ (p - y)) / N
        db = np.mean(p - y)

        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b