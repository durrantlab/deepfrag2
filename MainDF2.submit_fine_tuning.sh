#!/bin/bash
#SBATCH --job-name=deepfrag-finetune
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

#This command recommended for production. Prevents similar compounds from appearing in both training and val/test sets.
python MainDF2.py --data ${DIR_WITH_PDBSDF_FILES} --save_splits ${OUTPUT_FT_DIR}/splits.json --default_root_dir ${OUTPUT_FT_DIR} --aggregation_3x3_patches mean --aggregation_loss_vector mean --max_epochs 30 --mode warm_starting --model_for_warm_starting ${INPUT_MODEL_FOR_FINETUNING} --save_every_epoch --butina_cluster_division --butina_cluster_cutoff 0.4

#Random split for train/val/test. Good for testing.
#python MainDF2.py --data ${DIR_WITH_PDBSDF_FILES} --save_splits ${OUTPUT_FT_DIR}/splits.json --default_root_dir ${OUTPUT_FT_DIR} --aggregation_3x3_patches mean --aggregation_loss_vector mean --max_epochs 30 --mode warm_starting --model_for_warm_starting ${INPUT_MODEL_FOR_FINETUNING} --save_every_epoch
