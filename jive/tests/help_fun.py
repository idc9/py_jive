import numpy as np


def relative_error(true_value, estimate):
    """
    Relative error under L2/frobenius norm
    """
    # TODO: what if treu value is zero?
    return np.linalg.norm(true_value - estimate) / np.linalg.norm(true_value)


def absolute_error(true_value, estimate):
    """
    Absolute error under L2/frobenius norm
    """
    return np.linalg.norm(true_value - estimate)
