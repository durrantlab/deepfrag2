#!/bin/bash
#SBATCH --job-name=deepfrag_train
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

cd ${DEEPFRAG_ROOT}/

python MainDF2.py --csv ${EVERY_CSV} --data ${BINDINGMOAD_DIR}/ --save_splits ${OUTPUT_DIR}/splits.json --default_root_dir ${OUTPUT_DIR}/ --aggregation_3x3_patches mean --aggregation_loss_vector mean --max_epochs 30
