#!/bin/bash
#SBATCH --job-name=deepfrag_test_MOAD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --time=144:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

module purge
module load python/ondemand-jupyter-python3.8

. paths.sh

source activate ${CONDA_ENV_NAME}

cd ${DEEPFRAG_ROOT}

python MainDF2.py --csv ${EVERY_CSV} --data ${BINDINGMOAD_DIR} --load_splits ${OUTPUT_TRAIN_DIR}/splits.json --default_root_dir ${OUTPUT_TRAIN_DIR} --mode test --inference_rotations 8 --aggregation_rotations mean --load_newest_checkpoint True
