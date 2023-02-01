#!/bin/bash
#SBATCH --job-name=deepfrag_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --time=144:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

module purge
module load python/ondemand-jupyter-python3.8

. paths.sh

source activate ${CONDA_ENV_NAME}

cd ${DEEPFRAG_ROOT}

# This command is slower. It calculates fingerprints from the entire MOAD
# database (test, val, and train sets) for use as the label set.
python MainDF2.py --every_csv ${EVERY_CSV} \
    --data_dir ${BINDINGMOAD_DIR} \
    --custom_test_set_dir ${EXTERNAL_DATA} \
    --default_root_dir ${OUTPUT_FT_DIR} \
    --mode inference_custom_set \
    --rotations 8 \
    --aggregation_3x3_patches mean \
    --aggregation_loss_vector mean \
    --aggregation_rotations mean \
    --load_newest_checkpoint True \
    --inference_label_sets all \
    --fragment_representation ${DEEPFRAG_FRAGMENT_REPRESENTATION} \
    --cache_pdbs_to_disk
