import numpy as np
import glob
import os


def rand_rot():
    """Returns a random uniform quaternion rotation."""
    q = np.random.normal(size=4)  # sample quaternion from normal distribution
    q = q / np.sqrt(np.sum(q ** 2))  # normalize
    return q


def get_last_checkpoint(args):
    saved_checkpoints = glob.glob(
        args.default_root_dir + os.sep + "**" + os.sep + "last.ckpt", recursive=True
    )

    if len(saved_checkpoints) == 0:
        return None

    if len(saved_checkpoints) == 1:
        return saved_checkpoints[0]

    # Multiple saved checkpoints found. Find the most recent one.
    saved_checkpoints = sorted(saved_checkpoints, key=os.path.getmtime, reverse=True)
    return saved_checkpoints[0]
