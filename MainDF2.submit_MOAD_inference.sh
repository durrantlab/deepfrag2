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

#This command is recommended to perform a faster inference process
python MainDF2.py --csv ${EVERY_CSV} --data ${BINDINGMOAD_DIR} --external_data ${EXTERNAL_DATA} --default_root_dir ${OUTPUT_TRAIN_DIR} --mode inference  --inference_rotations 8 --aggregation_rotations mean --load_newest_checkpoint True --inference_label_sets test

#This command will perform a slower inference process because it requires to compute fingerprints for fragments into MOAD database
#python MainDF2.py --csv ${EVERY_CSV} --data ${BINDINGMOAD_DIR} --external_data ${EXTERNAL_DATA} --default_root_dir ${OUTPUT_TRAIN_DIR} --mode inference  --inference_rotations 8 --aggregation_rotations mean --load_newest_checkpoint True --inference_label_sets all
