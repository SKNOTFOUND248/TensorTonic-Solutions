import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    sigmoid = 1 / (1 + np.exp(-1 * np.array(x)))
    return sigmoid