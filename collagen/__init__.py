"""__init__.py"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deepfrag2")
except PackageNotFoundError:
    # package is not installed, we are probably running in development mode
    __version__ = "2.0.0"


__all__ = ["__version__"]
from collagen.core import *
import collagen.core

__all__ += collagen.core.__all__
