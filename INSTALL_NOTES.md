# Installation Guide

This guide provides detailed instructions for setting up the DeepFrag2 environment from the source code using Conda. This installation method is intended for users who wish to train new models, fine-tune existing ones, or contribute to the development of DeepFrag2. It supports both GPU and CPU environments.

For users who only require CPU-based inference with pre-trained models, a simpler installation is available via `pip`. For instructions on how to install the pip package, please refer to the main [README.md](./README.md) file.

## Download the Source Code

Clone the DeepFrag2 repository from GitHub.

```bash
git clone git@github.com:durrantlab/deepfrag2.git
```

## Environment Setup

These instructions assume you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed. We offer two environment setups: a GPU-enabled environment for training and a CPU-only environment that is ideal for inference.

### GPU Environment (for Training and Inference)

The GPU (CUDA) environment is required for training new models. The recommended way to install the required dependencies is by using an environment file.

1. Create a Conda environment using the `environment.yml` file. This command will create a new environment named `DeepFrag2`.

    ```bash
    conda env create -f environment.yml
    ```

2. Activate the newly created environment to proceed.

    ```bash
    conda activate DeepFrag2
    ```

### CPU-Only Environment (for Inference)

For users who only need to run inference (i.e., generate fragment suggestions with pre-trained models), we provide a CPU-only environment. This version is significantly easier to install as it does not require a GPU or the CUDA toolkit. It is also very fast for inference, making it the preferred choice for most users who do not intend to train new models.

Create a Conda environment using the `environment_cpu.yml` file. This will create a new environment named `DeepFrag2CPU`.

```bash
conda env create -f environment_cpu.yaml
```

Activate the newly created environment.

```bash
conda activate DeepFrag2CPU
```

## Configuration and Usage

All program configuration, such as file paths for datasets and models, is handled through command-line arguments.
For detailed instructions on how to run DeepFrag2 for training, fine-tuning, testing, and inference, please refer to the [Usage section in README.md](./README.md). The README provides comprehensive examples of the commands and arguments required for each mode of operation.
