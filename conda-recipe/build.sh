#!/bin/bash
# Install the package itself without its dependencies,
# as conda will manage all the dependencies listed in meta.yaml
$PYTHON -m pip install . -vv --no-deps --ignore-installed

# Debug: Check pip version and configuration
echo "Python version:"
$PYTHON --version
echo "Pip version:"
$PYTHON -m pip --version
echo "Pip configuration:"
$PYTHON -m pip config list

# Debug: Try to search for the package
echo "Searching for Fancy-aggregations on PyPI:"
$PYTHON -m pip search Fancy-aggregations || echo "Pip search failed or unavailable"

# Debug: Check what indexes pip is using
echo "Available indexes:"
$PYTHON -c "import pip._internal.network.session; print('Network check passed')" || echo "Network issues detected"

# Try installing with verbose output and no cache
echo "Attempting to install Fancy-aggregations:"
$PYTHON -m pip install --no-cache-dir -v Fancy-aggregations

echo "Attempting to install fair-esm:"
$PYTHON -m pip install --no-cache-dir -v fair-esm