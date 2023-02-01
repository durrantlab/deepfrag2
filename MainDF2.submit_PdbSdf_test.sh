#!/bin/bash
#SBATCH --job-name=deepfrag_test_PdbSdf
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

python MainDF2.py --data_dir ${DIR_WITH_PDBSDF_FILES} \
    --load_splits ${OUTPUT_FT_DIR}/splits.json \
    --default_root_dir ${OUTPUT_FT_DIR} \
    --mode test \
    --rotations 8 \
    --aggregation_3x3_patches mean \
    --aggregation_loss_vector mean \
    --aggregation_rotations mean \
    --load_newest_checkpoint True \
    --inference_label_sets test \
    --fragment_representation ${DEEPFRAG_FRAGMENT_REPRESENTATION} \
    --cache_pdbs_to_disk
