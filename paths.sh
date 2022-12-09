#!/bin/bash
# This bash files defines the absolute paths. Designed to prevent paths from
# ever being uploaded to git respository.

# Common parameters
export CONDA_ENV_NAME="deepfrag2cesar"
export HOME_DF_WORK="/ihome/jdurrant/crg93/test_scripts"
export DEEPFRAG_ROOT="${HOME_DF_WORK}/deepfrag2"

# Training on BindingMOAD parameters
export EVERY_CSV="/ihome/jdurrant/crg93/test_scripts/moad/every.csv"
export BINDINGMOAD_DIR="/ihome/jdurrant/crg93/test_scripts/moad/BindingMOAD_2020"
export OUTPUT_TRAIN_DIR="${HOME_DF_WORK}/output_deepfrag"

# Finetuning parameters
export DIR_WITH_PDBSDF_FILES="/ihome/jdurrant/crg93/test_scripts/prots_and_ligs"
export INPUT_MODEL_FOR_FINETUNING="${OUTPUT_TRAIN_DIR}/model_mean_mean_train.pt"
export OUTPUT_FT_DIR="${HOME_DF_WORK}/output_deepfrag_ft"
