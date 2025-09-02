# Installation Guide

This guide provides instructions for setting up the DeepFrag2 environment and configuring necessary paths.

## Installation

These instructions assume you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Create and Activate the Conda Environment

The recommended way to install the required dependencies is by using an environment file.

1. Create a Conda environment using the `environment.yml` file. This command will create a new environment named `DeepFrag2`.

    ```bash
    conda env create -f environment.yml
    ```

2. Activate the newly created environment to proceed.

    ```bash
    conda activate DeepFrag2
    ```

## Download the Source Code

Clone the DeepFrag2 repository from GitLab.

```bash
git clone git@github.com:durrantlab/deepfrag2.git
```

## Configure `paths.sh`

The `paths.sh` script is used to configure environment variables required by DeepFrag2. You must edit this script before running any training, finetuning, or inference scripts.

Below is a breakdown of the variables in `paths.sh`. An example configuration is provided at the end of this section.

### Common Parameters

- **`CONDA_ENV_NAME`**: The name of the Conda environment you created.

    ```bash
    export CONDA_ENV_NAME="DeepFrag2"
    ```

- **`HOME_DF_WORK`**: The absolute path to the directory where you downloaded DeepFrag2.

    ```bash
    export HOME_DF_WORK="/path/to/your/download/location"
    ```

- **`DEEPFRAG_ROOT`**: The full path to the `deepfrag2` source code directory. This is typically constructed from `HOME_DF_WORK`.

    ```bash
    export DEEPFRAG_ROOT="${HOME_DF_WORK}/deepfrag2"
    ```

### Training on BindingMOAD Parameters

- **`MOAD_PATH`**: The path to the directory where the BindingMOAD database is stored.

    ```bash
    export MOAD_PATH="/path/to/your/moad/database"
    ```

- **`EVERY_CSV`**: The full path to the `every.csv` file from the BindingMOAD database.

    ```bash
    export EVERY_CSV="${MOAD_PATH}/every.csv"
    ```

- **`BINDINGMOAD_DIR`**: The path to the directory containing the BindingMOAD structures.

    ```bash
    export BINDINGMOAD_DIR="${MOAD_PATH}/BindingMOAD_2020"
    ```

- **`OUTPUT_TRAIN_DIR`**: The directory where the outputs of the training process will be saved.

    ```bash
    export OUTPUT_TRAIN_DIR="${HOME_DF_WORK}/output_deepfrag"
    ```

### Finetuning Parameters

- **`DIR_WITH_PDBSDF_FILES`**: The path to your custom dataset of protein-ligand pairs (PDB and SDF files) for finetuning.

    ```bash
    export DIR_WITH_PDBSDF_FILES="/path/to/your/finetuning_dataset"
    ```

- **`INPUT_MODEL_FOR_FINETUNING`**: The path to the pre-trained model file (`.pt`) that will be used as the starting point for finetuning.

    ```bash
    export INPUT_MODEL_FOR_FINETUNING="${OUTPUT_TRAIN_DIR}/model_mean_mean_train.pt"
    ```

- **`OUTPUT_FT_DIR`**: The directory where the outputs of the finetuning process will be saved.

    ```bash
    export OUTPUT_FT_DIR="${HOME_DF_WORK}/output_deepfrag_ft"
    ```

- **`FRACTION_TRAIN`**: The fraction of your custom dataset to use for training during finetuning (e.g., `0.8` for 80%). The rest will be split between validation and testing. Set to `1.0` to use all data for training.

    ```bash
    export FRACTION_TRAIN=0.8
    ```

### Inference Parameters

- **`EXTERNAL_DATA`**: The path to an external dataset (protein-ligand pairs) for inference. These should be different from any data used in training or finetuning.

    ```bash
    export EXTERNAL_DATA="/path/to/your/inference_dataset"
    ```

### Example `paths.sh` Configuration

Here is a complete example of a configured `paths.sh` file. You will need to replace the placeholder paths with the actual paths on your system.

```bash
# Common parameters
export CONDA_ENV_NAME="DeepFrag2"
export HOME_DF_WORK="/usr/local"
export DEEPFRAG_ROOT="${HOME_DF_WORK}/deepfrag2"

# Training on BindingMOAD parameters
# Assumes a 'moad' folder containing the MOAD database.
export MOAD_PATH="/path/to/data/moad"
export EVERY_CSV="${MOAD_PATH}/every.csv"
export BINDINGMOAD_DIR="${MOAD_PATH}/BindingMOAD_2020"
export OUTPUT_TRAIN_DIR="${HOME_DF_WORK}/output_deepfrag"

# Finetuning parameters
# Assumes a folder with protein-ligand pairs for finetuning.
export DIR_WITH_PDBSDF_FILES="/path/to/data/my_pdb_sdf_files/fine_tuning"
export INPUT_MODEL_FOR_FINETUNING="${OUTPUT_TRAIN_DIR}/model_mean_mean_train.pt"
export OUTPUT_FT_DIR="${HOME_DF_WORK}/output_deepfrag_ft"
export FRACTION_TRAIN=0.8

# Inference parameters
# Assumes a folder with protein-ligand pairs for inference.
export EXTERNAL_DATA="/path/to/data/my_pdb_sdf_files/external"
```

## 4. Description of Provided Scripts

After configuration, you can use the following scripts to run DeepFrag2:

- `paths.sh`: Sets the required environment variables. Source this script (`source paths.sh`) before running other scripts.
- `MainDF2.submit_MOAD_train.sh`: Trains a new DeepFrag model from scratch using the BindingMOAD database.
- `MainDF2.submit_MOAD_train_resume.sh`: Resumes a paused or incomplete training process from a checkpoint.
- `MainDF2.submit_MOAD_test`: Performs inference on the test set derived from the BindingMOAD database.
- `MainDF2.submit_PdbSdf_fine_tuning.sh`: Fine-tunes a pre-trained DeepFrag model on a custom dataset of PDB/SDF pairs.
- `MainDF2.submit_PdbSdf_test.sh`: Performs inference on the test set derived from your custom finetuning dataset.
- `MainDF2.submit_inference.sh`: Runs inference on an external dataset not used for training or finetuning.
