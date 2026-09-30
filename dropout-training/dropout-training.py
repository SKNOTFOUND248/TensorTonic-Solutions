import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:

    if not 0 <= p < 1:
        raise ValueError("p must satisfy 0 <= p < 1")

    # Convert input to NumPy array without modifying original input
    x_arr = np.asarray(x, dtype=float)

    # Keep probability
    keep_prob = 1 - p

    # Choose random-number generator
    if rng is None:
        random_values = np.random.random(x_arr.shape)
    else:
        random_values = rng.random(x_arr.shape)

    # Bernoulli mask: 1 where retained, 0 where dropped
    binary_mask = (random_values < keep_prob).astype(float)

    # Inverted dropout scaling
    dropout_pattern = binary_mask / keep_prob

    # Apply dropout
    output = x_arr * dropout_pattern

    return output, dropout_pattern