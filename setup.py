from setuptools import setup, find_packages

setup(
    # This is the name of the project. The first time you publish this
    # package, this name will be registered. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install DeepFrag
    #
    # And where it will live on PyPI: https://pypi.org/project/DeepFrag/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='DeepFrag',

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/guides/single-sourcing-package-version/
    version='2.0.0',

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description="A Convolutional Neural Network-driven framework for lead optimization based on chemical fragments",

    # This is the link to project's main homepage.
    url="https://github.com/durrantlab/deepfrag2",

    # Packages to be installed.
    packages=find_packages(),

    # Python versions supported. 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.9, <4",
)
