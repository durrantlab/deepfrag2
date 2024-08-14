#!/bin/bash
#SBATCH --job-name=deepfrag_test_MOAD
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

# NOTE: Below not tested yet

python MainDF2.py --default_root_dir ${OUTPUT_TRAIN_DIR} \
    --mode inference \
    --rotations 8 \
    --aggregation_3x3_patches mean \
    --aggregation_loss_vector mean \
    --aggregation_rotations mean \
    --load_checkpoint "path/to/checkpoint.pt" \
    --inference_label_sets "my_file.smi,my_file2.smi" \
    --receptor "/path/to/receptor.pdb" \
    --ligand "/path/to/ligand.sdf" \
    --num_inference_predictions 10 \
    --xyz "5,6,7" \
    --fragment_representation ${DEEPFRAG_FRAGMENT_REPRESENTATION}
