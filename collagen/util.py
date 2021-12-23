import numpy as np


def rand_rot():
    """Returns a random uniform quaternion rotation."""
    q = np.random.normal(size=4)  # sample quaternion from normal distribution
    q = q / np.sqrt(np.sum(q ** 2))  # normalize
    return q
