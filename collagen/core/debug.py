import os

# Global funcitons for debugging.

def logit(msg, path: str):
    with open(os.path.abspath(os.path.expanduser(path)), "a") as f:
        f.write(str(msg) + "\n")