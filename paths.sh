# This bash files defines the absolute paths. Designed to prevent paths from
# ever being uploaded to git respository.

# Common parameters
export CONDA_ENV_NAME="my_deepfrag2_env"
export DEEPFRAG_ROOT="/path/to/deepfrag2/"
export OUTPUT_DIR="/path/to/output_ft_buti04/"

# Training on BindingMOAD parameters
export EVERY_CSV="/path/to/data/moad/every.csv"
export BINDINGMOAD_DIR="/path/to/data/moad/BindingMOAD_2020/"

# Finetuning parameters
export DIR_WITH_MODELS="/path/to/my_dir_with_pdb_and_sdf_models/"
export INPUT_MODEL_FOR_FINETUNING="/path/to/output_deepfrag/model_mean_mean_train.pt"
