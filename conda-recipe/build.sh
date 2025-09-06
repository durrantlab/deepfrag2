#!/bin/bash
# Install the package itself without its dependencies,
# as conda will manage all the dependencies listed in meta.yaml
$PYTHON -m pip install . -vv --no-deps --ignore-installed

# Install additional pip-only packages that aren't available in conda
$PYTHON -m pip install Fancy-aggregations fair-esm