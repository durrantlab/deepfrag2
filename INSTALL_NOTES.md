# Clone the repository:

* git clone https://github.com/durrantlab/deepfrag2

# Installation of Python environment via conda

* conda env create -f environment.yml
* conda activate Deepfrag2

# Description of the scripts provided

* `paths.sh`: this script is used to configure the different variables to be
  used by the scripts detailed below (also see Section 4).
* `MainDF2.submit_MOAD_train.sh`: this script is used create (train) a DeepFrag
  model from scratch using the MOAD database
* `MainDF2.submit_MOAD_train_resume.sh`: this script is used to resume the
  training process of a DeepFrag model from a checkpoint. It is recommended when
  the training process started with the previous script was not completed due to
  a time limit on the computational resource used.
* `MainDF2.submit_MOAD_test:` this script is used to perform an inference
  process on the compounds included in the test set created from the MOAD
  database. That test dataset was created before starting the training process
  with the first script above mentioned.
* `MainDF2.submit_PdbSdf_fine_tuning.sh`: this script is used to run a
  fine-tuning process from a dataset comprised of protein-ligand pairs (denoted
  as `PDB_SDF_FILES` in this manual), where every protein and its corresponding
  ligand are into PDB and SDF files, respectively.
* `MainDF2.submit_PdbSdf_test.sh`: this script is used to perform an inference
  process on the compounds included in the test set created from the
  `PDB_SDF_FILES` database. That test dataset was created before starting the
  fine-tunning process if the FRACTION_TRAIN variable is less than 1.0.
* `MainDF2.submit_inference.sh`: this script is to run an inference process on
  an external dataset, that is, on a dataset other than the MOAD and
  `PDB_SDF_FILES` databases that were used for the training and fine-tunning
  process.

4. How to configure the `paths.sh` script
   * This script contains several variables that are required to run DeepFrag.
     See Section 5 for a complete example.

# Common parameters

* `export CONDA_ENV_NAME="my_deepfrag2_env"`
  * `my_deepfrag2_env` must be changed by the name of the conda environment
    created. In this manual, it was used the name of `deepfrag`.
* `export HOME_DF_WORK="/path/to/where/deepfrag/was/downloaded"`
  * `"/path/to/where/deepfrag/was/downloaded"` must be changed by the path where
    the DeepFrag software was downloaded. Here will be used the `/usr/local/`
    path as example.
* `export DEEPFRAG_ROOT="${HOME_DF_WORK}/deepfrag2"`
  * This variable uses the `HOME_DF_WORK` variable to create the path where the
    DeepFrag software was downloaded.

# Training on BindingMOAD parameters

* `export MOAD_PATH="/path/to/data/moad"`
  * `"/path/to/data/moad"` must be changed by the path where the MOAD database
    is stored.
* `export EVERY_CSV="${MOAD_PATH}/every.csv"`
  * This variable uses the `MOAD_PATH` variable to create the path for the
    `every.csv`
* `export BINDINGMOAD_DIR="${MOAD_PATH}/BindingMOAD_2020"`
  * This variable uses the `MOAD_PATH` variable to create the path where the
    compounds of the MOAD database are stored.
* `export OUTPUT_TRAIN_DIR="${HOME_DF_WORK}/output_deepfrag"`
  * This variable uses the `HOME_DF_WORK` variable to create a folder where the
    output of the training process will be saved. In this case, the result of
    the training process will be saved in the `output_deepfrag` folder. This
    folder name can be changed by another one desired by the final user.

# Finetuning parameters

* `export DIR_WITH_PDBSDF_FILES="/path/to/my_dir_with_pdb_and_sdf_models"`
  * `"/path/to/my_dir_with_pdb_and_sdf_models"` must be changed by the path where
    is stored a dataset comprised of protein-ligand pairs (denoted as
    `PDB_SDF_FILES` in this manual), where every protein and its corresponding
    ligand are into PDB and SDF files, respectively. 
* `export INPUT_MODEL_FOR_FINETUNING="${OUTPUT_TRAIN_DIR}/model_mean_mean_train.pt"`
  * This variable uses the `OUTPUT_TRAIN_DIR` variable to create the path where it
    is saved the final model of the training process. This model is the best
    DeepFrag model, and it is the recommended one to fine tuning it.
* `export OUTPUT_FT_DIR="${HOME_DF_WORK}/output_deepfrag_ft"`
  * This variable uses the `HOME_DF_WORK` variable to create a folder where the
    output of the fine-tuning process will be saved. In this case, the result of
    the fine-tuning process will be saved in the `output_deepfrag_ft` folder.
    This folder name can be changed by another one desired by the final user.
* `export FRACTION_TRAIN=0.6`
  * This variable contains the percentage of protein-ligand pairs to be used as
    training set for the fine-tuning process, the remaining protein-ligand pairs
    will be allocated in the validation and test datasets. Remember that those
    protein-ligand pairs are stored in the path specified for the
    `DIR_WITH_PDBSDF_FILES` variable. If all the protein-ligand pairs are used as
    training for the fine-tuning process, then this variable must be changed to
    1.0.

# Inference parameters

* `export EXTERNAL_DATA="/path/to/my_external_data/in_pdb_and_sdf_files"`
  * `"/path/to/my_external_data/in_pdb_and_sdf_files"` must be changed by the
    path containing protein-ligand pairs to be used as external dataset. Note
    that these protein-ligand pairs are different to the protein-ligand pairs
    used to run the fine-tuning process.

5. Example of configuration of the `paths.sh` script

# Common parameters

`export CONDA_ENV_NAME="deepfrag"`
`export HOME_DF_WORK="/usr/local"`
`export DEEPFRAG_ROOT="${HOME_DF_WORK}/deepfrag2"`

# Training on BindingMOAD parameters

`export MOAD_PATH="/usr/local/moad" (here, it is supposed that there is a moad folder containing the MOAD database)`
`export EVERY_CSV="${MOAD_PATH}/every.csv"`
`export BINDINGMOAD_DIR="${MOAD_PATH}/BindingMOAD_2020"`
`export OUTPUT_TRAIN_DIR="${HOME_DF_WORK}/output_deepfrag"`

# Finetuning parameters

`export DIR_WITH_PDBSDF_FILES="/usr/local/my_pdb_sdf_files/fine_tuning"` (here,
it is supposed that there is a folder named `my_pdb_sdf_files` which contain
another folder named `external`, where this last contains the protein-ligand
pairs to be used for fine-tuning)

* `export INPUT_MODEL_FOR_FINETUNING="${OUTPUT_TRAIN_DIR}/model_mean_mean_train.pt"`
* `export OUTPUT_FT_DIR="${HOME_DF_WORK}/output_deepfrag_ft"`
* `export FRACTION_TRAIN=0.6`

# Inference parameters

* `export EXTERNAL_DATA="/usr/local/my_pdb_sdf_files/external"` (here, it is
supposed that there is a folder named `my_pdb_sdf_files` which contain
another folder named `external`, where this last contains the protein-ligand
pairs to be used as external dataset)
