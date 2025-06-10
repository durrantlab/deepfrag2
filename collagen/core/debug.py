"""Global funcitons for debugging."""

import os
from typing import Any


def logit(msg: Any, path: str):
    """Log a message to a file.

    Args:
        msg (Any): The message to log.
        path (str): The path to the file.
    """
    with open(os.path.abspath(os.path.expanduser(path)), "a") as f:
        f.write(str(msg) + "\n")
