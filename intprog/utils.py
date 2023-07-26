import numpy as np


def is_integer(x: np.array):
    fractional_parts = get_fractional_parts(x)
    return np.isclose(np.array([min(f, 1 - f) for f in fractional_parts]), 0, atol=1e-4)


def get_fractional_parts(x: np.array):
    return x - np.array(x).astype(np.int32)


def get_index_of_most_fractional_variable(x: np.array):
    fractional_parts = get_fractional_parts(x)
    return np.argmax(
        np.array([max(f, 1 - f) if f > 0.0 else -1.0 for f in fractional_parts])
    )
