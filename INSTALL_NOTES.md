# Installation Guide

This guide provides instructions for setting up the DeepFrag2 environment.

## Download the Source Code

Clone the DeepFrag2 repository from GitHub.

```bash
git clone git@github.com:durrantlab/deepfrag2.git
```

## Create and Activate the Conda Environment

These instructions assume you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed. The recommended way to install the required dependencies is by using an environment file.

1. Create a Conda environment using the `environment.yml` file. This command will create a new environment named `DeepFrag2`.

    ```bash
    conda env create -f environment.yml
    ```

2. Activate the newly created environment to proceed.

    ```bash
    conda activate DeepFrag2
    ```

## Configuration and Usage

All program configuration, such as file paths for datasets and models, is handled through command-line arguments.
For detailed instructions on how to run DeepFrag2 for training, fine-tuning, testing, and inference, please refer to the [Usage section in README.md](./README.md). The README provides comprehensive examples of the commands and arguments required for each mode of operation.
