
# Frag Atlas

Frag Atlas (DeepFrag 2) is a collection of machine learning tools for drug discovery & lead optimization applications.

## Overview

The source code is organized into several subdirectoreis in the `atlas` package:

- `convert`: Python scripts for converting between data formats.
- `data`: Utility library for reading different data formats.
- `models`: Model specification (PyTorch) code.
- `tools`: Python scripts for running various tasks.
- `train`: Python scripts for training ML models.
- `viz`: A collection of web-based visualization tools.

## Running Tests

Test suites are created with `unittest`. In the root directory, run:

```sh
$ python3 -m unittest
```

to run all the tests.
