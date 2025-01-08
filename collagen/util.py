"""Utility functions for collagen."""

import numpy as np  # type: ignore
import time
from typing import List, Set, Union


def rand_rot() -> np.ndarray:
    """Return a random uniform quaternion rotation.

    Returns:
        np.ndarray: A random uniform quaternion rotation.
    """

    # For debugging purposes, we will not do random rotation.
    rot = np.array([1, 0, 0, 0])
    return rot

    # Had to do the below to get the rotation to be different on every rotation
    # during inferance. Note that if two rotations are requested within a
    # microsecond of each other, will return same rotation. But even that
    # wouldn't necessarily be problematic.
    rot_rand_num_gen = np.random.default_rng(int(time.time() * 1000000))

    q = rot_rand_num_gen.normal(size=4)  # sample quaternion from normal distribution
    # q = np.random.normal(size=4)

    # For debugging, if you want a consistent (non-random) rotation.
    # q = np.array([0.5, 0.234, 0.9234, 0.21])

    q = q / np.sqrt(np.sum(q**2))  # normalize

    return q


VDW_RADIUS_BY_NUM = {
    1: 1.1,
    2: 1.4,
    3: 1.82,
    4: 1.53,
    5: 1.92,
    6: 1.7,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    10: 1.54,
    11: 2.27,
    12: 1.73,
    13: 1.84,
    14: 2.1,
    15: 1.8,
    16: 1.8,
    17: 1.75,
    18: 1.88,
    19: 2.75,
    20: 2.31,
    21: 2.15,
    22: 2.11,
    23: 2.07,
    24: 2.06,
    25: 2.05,
    26: 2.04,
    27: 2.0,
    28: 1.97,
    29: 1.96,
    30: 2.01,
    31: 1.87,
    32: 2.11,
    33: 1.85,
    34: 1.9,
    35: 1.85,
    36: 2.02,
    37: 3.03,
    38: 2.49,
    39: 2.32,
    40: 2.23,
    41: 2.18,
    42: 2.17,
    43: 2.16,
    44: 2.13,
    45: 2.1,
    46: 2.1,
    47: 2.11,
    48: 2.18,
    49: 1.93,
    50: 2.17,
    51: 2.06,
    52: 2.06,
    53: 1.98,
    54: 2.16,
    55: 3.43,
    56: 2.68,
    57: 2.43,
    58: 2.42,
    59: 2.4,
    60: 2.39,
    61: 2.38,
    62: 2.36,
    63: 2.35,
    64: 2.34,
    65: 2.33,
    66: 2.31,
    67: 2.3,
    68: 2.29,
    69: 2.27,
    70: 2.26,
    71: 2.24,
    72: 2.23,
    73: 2.22,
    74: 2.18,
    75: 2.16,
    76: 2.16,
    77: 2.13,
    78: 2.13,
    79: 2.14,
    80: 2.23,
    81: 1.96,
    82: 2.02,
    83: 2.07,
    84: 1.97,
    85: 2.02,
    86: 2.2,
    87: 3.48,
    88: 2.83,
    89: 2.47,
    90: 2.45,
    91: 2.43,
    92: 2.41,
    93: 2.39,
    94: 2.43,
    95: 2.44,
    96: 2.45,
    97: 2.44,
    98: 2.45,
    99: 2.45,
    100: 2.45,
    101: 2.46,
    102: 2.46,
    103: 2.46,
}


def get_vdw_radius(num: int) -> float:
    """Return the Van-der Waals radius for a given atomic number or 0.

    Args:
        num (int): Atomic number.

    Returns:
        float: Van-der Waals radius for a given atomic number or 0.
    """
    return VDW_RADIUS_BY_NUM[num] if num in VDW_RADIUS_BY_NUM else 0


def sorted_list(st: Union[Set[str], List[str]]) -> List[str]:
    """Return a sorted list from a set.

    Args:
        st (Union[Set[str], List[str]]): Set to be sorted.

    Returns:
        List[str]: Sorted list from a set.
    """
    return sorted(st)
