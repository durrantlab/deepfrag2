#!/bin/bash
# This bash files defines the absolute paths. Designed to prevent paths from
# ever being uploaded to git respository.

# Common parameters
export CONDA_ENV_NAME="my_deepfrag2_env"
export HOME_DF_WORK="/path/to/where/deepfrag/was/downloaded"
export DEEPFRAG_ROOT="${HOME_DF_WORK}/deepfrag2"

# Training on BindingMOAD parameters
export MOAD_PATH="/path/to/data/moad"
export EVERY_CSV="${MOAD_PATH}/every.csv"
export BINDINGMOAD_DIR="${MOAD_PATH}/BindingMOAD_2020"
export OUTPUT_TRAIN_DIR="${HOME_DF_WORK}/output_deepfrag"

# Finetuning parameters
export DIR_WITH_PDBSDF_FILES="/path/to/my_dir_with_pdb_and_sdf_models"
export INPUT_MODEL_FOR_FINETUNING="${OUTPUT_TRAIN_DIR}/model_mean_mean_train.pt"
export OUTPUT_FT_DIR="${HOME_DF_WORK}/output_deepfrag_ft"
export FRACTION_TRAIN=0.6

# Inferece parameters
export EXTERNAL_DATA="/path/to/my_external_data/in_pdb_and_sdf_files"
